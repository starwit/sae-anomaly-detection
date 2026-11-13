from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import logging
import os
import sys
from typing import List
import cv2
import numpy as np
import torch

from movementpredictor import DataFilterer, makeTorchDataLoader, inference_with_stats, get_unlikely_trajectories, InferenceResult, model_architectures
from movementpredictor.anomalydetection.anomaly_detector import plot_unlikely_samples, calculate_trajectory_anomaly_scores
from movementpredictor.data.datamanagement import get_downsampled_tensor_img
from visionapi.anomaly_pb2 import AnomalyMessage, Point, Trajectory

from anomalydetection.config import AnomalyDetectionConfig
from anomalydetection.modelinfocollector import ModelInfoCollector


log = logging.getLogger(__name__)

PLACEHOLDER_GREY = 128
PLACEHOLDER_DEFAULT_SHAPE = (1920, 1080)


@lru_cache(maxsize=8)
def _grey_jpeg(width: int, height: int) -> bytes:
    """Uniform grey JPEG, used as frame background if the input message did not contain any frame data (useful for testing)."""
    img = np.full((height, width, 3), PLACEHOLDER_GREY, dtype=np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


# TODO: workaround for movementpredictor writing tqdm progress bars / print() output to stdout/stderr; remove once it reports via logging upstream
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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.parameters = ModelInfoCollector(CONFIG).model_parameters if CONFIG.model.parameters_path.exists() else None
        if self.parameters is None and self._config.model.start_training is False:
            log.error("Could not find your model parameters! If you want to train a new model, set start_training to True in your config.")
            exit(1)

        if not self._config.model.start_training:
            self.model = model_architectures.get_model(architecture=self.parameters["model_architecture"], 
                                                   output_prob=self.parameters["output_distribution"], 
                                                   path_model=str(self._config.model.weights_path))
            self.model.eval()

            if self.model is None:
                log.error("Could not find your model!")
                exit(1)

            log.info("starting anomaly detection with parameters:")
            for key in self.parameters.keys():
                log.info(key + ": " + str(self.parameters[key]))
      
    def filter_tracks(self, tracks):
        if len(tracks) != 0:
            num_tracks_before = len(tracks)
            # TODO: workaround for movementpredictor (tqdm progress bar in DataFilterer); remove once fixed upstream
            with SuppressOutput():
                if self._config.filtering:
                    tracks = DataFilterer().apply_filtering(tracks)
                else:
                    tracks = DataFilterer().only_smoothing(tracks)
            num_tracks = sum(len(tr) for tr in tracks.values())
            log.debug(f"{'Filtered' if self._config.filtering else 'Smoothed (filtering disabled)'} tracks: "
                      f"{num_tracks_before} -> {num_tracks} tracks in {len(tracks)} trajectories")

        return tracks
    
    def examine_tracks_for_anomalies(self, tracks, frames) -> AnomalyMessage:
        anomalies = []

        if len(frames) < 2 or frames[-1].timestamp_utc_ms == frames[0].timestamp_utc_ms:
            log.warning(f"Cannot determine frame rate from {len(frames)} frame(s), skipping anomaly detection for {len(tracks)} tracks")
            return AnomalyMessage()

        threshold = self.parameters["anomaly_threshold"] if self._config.model.anomaly_threshold_override is None else self._config.model.anomaly_threshold_override

        if len(tracks) != 0:
            # TODO: workaround for movementpredictor (tqdm progress bar in inference_with_stats); remove once fixed upstream
            with SuppressOutput():      # suppress tqdm progress bar
                diff = frames[-1].timestamp_utc_ms - frames[0].timestamp_utc_ms
                frame_rate = (1000*(len(frames)-1))/diff
                dataloader = makeTorchDataLoader(tracks, time_diff_prediction=self.parameters["time_diff_prediction"], frame_rate=frame_rate, pixel_per_axis=self.parameters["pixel_per_axis"])
                    
                if dataloader is not None:
                    outputs_with_stats = inference_with_stats(self.model, dataloader)
                    anomalies = get_unlikely_trajectories(outputs_with_stats, threshold)

            if len(anomalies) != 0:
                log.info(self._describe_anomalies(anomalies, outputs_with_stats, tracks, threshold))

                if self._config.debug_plots:
                    log.info(f"Plotting {len(anomalies)} anomalies")
                    frame_img = get_downsampled_tensor_img(frames[0], self.parameters["pixel_per_axis"])
                    plot_unlikely_samples(frame_img, dataloader, anomalies, "")

        return self._write_anomaly_message(anomalies, frames, tracks)
    
    def _describe_anomalies(self, anomalies: List[InferenceResult], samples_with_stats: List[InferenceResult], tracks, threshold: float) -> str:
        """Compact one-line summary of all anomalous trajectories (for logging)"""
        # TODO: workaround for movementpredictor; get_unlikely_trajectories() computes but does not return trajectory scores, so they are recomputed here
        scores = dict(calculate_trajectory_anomaly_scores(samples_with_stats))
        triggers_per_id = Counter(anomaly.obj_id for anomaly in anomalies)

        descriptions = []
        for obj_id, num_triggers in triggers_per_id.items():
            trajectory = tracks[obj_id]
            start = datetime.fromtimestamp(trajectory[0].capture_ts / 1000, tz=timezone.utc)
            duration_s = (trajectory[-1].capture_ts - trajectory[0].capture_ts) / 1000
            descriptions.append(f"id={obj_id} class={trajectory[0].class_id} score={scores.get(obj_id, float('nan')):.4f} "
                                f"start={start:%Y-%m-%dT%H:%M:%S}Z duration={duration_s:.1f}s points={len(trajectory)} triggers={num_triggers}")

        return f"Anomaly found: {len(triggers_per_id)} trajectories (threshold={threshold:.4f}): " + "; ".join(descriptions)

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
                    anomaly_frame.frame_data_jpeg = frame.frame_data_jpeg if len(frame.frame_data_jpeg) > 0 else self._placeholder_jpeg(frame)
                    anomaly_frame.timestamp_utc_ms = frame.timestamp_utc_ms
        
        return anomaly_msg

    def _placeholder_jpeg(self, frame) -> bytes:
        width, height = frame.shape.width, frame.shape.height
        if width <= 0 or height <= 0:
            width, height = PLACEHOLDER_DEFAULT_SHAPE
        return _grey_jpeg(width, height)

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
