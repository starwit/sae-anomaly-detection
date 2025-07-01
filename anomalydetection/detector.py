from collections import defaultdict
import logging
import os
import sys
from typing import List
import torch

from movementpredictor import DataFilterer, makeTorchDataLoader, inference_with_stats, get_unlikely_trajectories, InferenceResult, model_architectures
from movementpredictor.anomalydetection.anomaly_detector import plot_unlikely_samples
from movementpredictor.data.datamanagement import get_downsampled_tensor_img
from visionapi.anomaly_pb2 import AnomalyMessage, Point, Trajectory

from anomalydetection.config import AnomalyDetectionConfig
from anomalydetection.modelinfocollector import ModelInfoCollector


log = logging.getLogger(__name__)


class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class Detector():

    def __init__(self, CONFIG: AnomalyDetectionConfig) -> None:
        log.setLevel(CONFIG.log_level.value)
        self._config = CONFIG
        self._parameters = ModelInfoCollector(CONFIG).model_parameters
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model_architectures.get_model(architecture=self._parameters["model_architecture"], 
                                                   output_prob=self._parameters["output_distribution"], 
                                                   path_model=str(self._config.model.weights_path))
        self.model.eval()

        if self.model is None:
            log.error("Could not find your model!")
            exit(1)

        log.info("starting anomaly detection with parameters:")
        for key in self._parameters.keys():
            log.info(key + ": " + str(self._parameters[key]))
      
    def filter_tracks(self, tracks):
        if len(tracks) != 0:
            if self._config.filtering:
                log.info(f"num single tracks before filtering: {len(tracks)}")
                with SuppressOutput():
                    tracks = DataFilterer().apply_filtering(tracks)
                num_tracks = 0 
                for tr in tracks.values():
                    num_tracks += len(tr)
                log.info(f"num tracks after filtering: {num_tracks}, within {len(tracks)} trajectories")
            else:
                with SuppressOutput():
                    tracks = DataFilterer().only_smoothing(tracks)
                log.info(f"TESTING WITHOUT FILTERING")

        return tracks
    
    def examine_tracks_for_anomalies(self, tracks, frames) -> AnomalyMessage:
        anomalies = []

        if len(tracks) != 0:
            with SuppressOutput():      # suppress tqdm progress bar
                diff = frames[-1].timestamp_utc_ms - frames[0].timestamp_utc_ms
                frame_rate = (1000*(len(frames)-1))/diff
                dataloader = makeTorchDataLoader(tracks, time_diff_prediction=self._parameters["time_diff_prediction"], frame_rate=frame_rate, pixel_per_axis=self._parameters["pixel_per_axis"])
                    
                if dataloader is not None:
                    outputs_with_stats = inference_with_stats(self.model, dataloader)
                    anomalies = get_unlikely_trajectories(outputs_with_stats, self._parameters["anomaly_threshold"] 
                                                        if self._config.model.anomaly_threshold_override is None else self._config.model.anomaly_threshold_override)

            if len(anomalies) != 0:
                log.info("ANOMALY FOUND!\n")

                if self._config.debug_plots:
                    frame_img = get_downsampled_tensor_img(frames[0], self._parameters["pixel_per_axis"])
                    plot_unlikely_samples(frame_img, dataloader, anomalies, "")

        return self._write_anomaly_message(anomalies, frames, tracks)
    
    def _write_anomaly_message(self, anomalies: List[InferenceResult], frames, tracks) -> AnomalyMessage:
        anomaly_msg = AnomalyMessage()
        
        if len(anomalies) > 0:
            anomaly_trajectories = {}
            anomaly_ids = [anomaly.obj_id for anomaly in anomalies]

            for id in anomaly_ids:
                anomaly_trajectories[id] = tracks[id]

            for id in anomaly_trajectories.keys():
                trajectory = anomaly_msg.trajectories.add() 
                trajectory.CopyFrom(self._map_anomaly(anomalies, anomaly_trajectories[id]))
        
            if len(frames) > 0:
                timestamps = [track.capture_ts for traj in anomaly_trajectories.values() for track in traj]
                min_ts_ms, max_ts_ms = min(timestamps), max(timestamps)
                frames_with_anomaly = [frame for frame in frames 
                                       if frame.timestamp_utc_ms >= int(min_ts_ms) - 2000 and 
                                       frame.timestamp_utc_ms <= int(max_ts_ms) + 2000]
                for frame in frames_with_anomaly:
                    anomaly_frame = anomaly_msg.jpeg_frames.add()
                    anomaly_frame.frame_data_jpeg = frame.frame_data_jpeg
                    anomaly_frame.timestamp_utc_ms = frame.timestamp_utc_ms
        
        return anomaly_msg

    def _map_anomaly(self, total_anomalies: List[InferenceResult], anomaly_trajectory) -> Trajectory:
        trajectory = Trajectory()
        trajectory.object_id = bytes.fromhex(anomaly_trajectory[0].uuid)
        trajectory.class_id = anomaly_trajectory[0].class_id

        anomaly_trigger_ts = [a.timestamp for a in total_anomalies if a.obj_id == trajectory.object_id]

        for track in anomaly_trajectory:
            trajectory_point = trajectory.trajectory_points.add()

            center_point = Point()
            center_point.x = track.center[0] 
            center_point.y = track.center[1]

            trajectory_point.detection_center.CopyFrom(center_point)
            if track.capture_ts in anomaly_trigger_ts:
                trajectory_point.anomaly_trigger = True
            else:
                trajectory_point.anomaly_trigger = False
            trajectory_point.timestamp_utc_ms = track.capture_ts

        return trajectory
