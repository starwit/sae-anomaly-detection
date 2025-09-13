import json
import logging
from datetime import datetime, timedelta
import os
import random
from typing import Any, Dict, List, Tuple, Optional

from prometheus_client import Counter, Histogram, Summary
from visionapi.sae_pb2 import SaeMessage

from anomalydetection.detector import Detector
from anomalydetection.modelinfocollector import ModelInfoCollector
from anomalydetection.trackcollector import TimedTrajectories

from movementpredictor.cnn import training, inferencing
from movementpredictor.data.dataset import store_data, getTorchDataSet, getTorchDataLoader
from movementpredictor.anomalydetection.anomaly_detector import calculate_trajectory_threshold
from movementpredictor import TrackedObjectPosition, makeTorchDataLoader

from .config import AnomalyDetectionConfig, ModelTrainingConfig


logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('anomaly_detection_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
OBJECT_COUNTER = Counter('anomaly_detection_object_counter', 'How many detections have been transformed')
PROTO_SERIALIZATION_DURATION = Summary('anomaly_detection_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('anomaly_detection_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class AnomalyDetection:

    def __init__(self, CONFIG: AnomalyDetectionConfig) -> None:
        logger.setLevel(CONFIG.log_level.value)
        self._config = CONFIG
        self.model_info = ModelInfoCollector(CONFIG).model_info if not CONFIG.model.start_training else None
        self._detector = Detector(self._config)
        self._timed_data_collector = TimedTrajectories(self._config.log_level.value) 

        # ---- training control state ----
        self._train_start_time: Optional[datetime] = None  
        # store snapshots to later hand full interval to trainer
        self._train_buffer: List[TrackedObjectPosition] = []  
 
    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)
   
    @GET_DURATION.time()
    def get(self, input_proto):
        sae_msg = self._unpack_proto(input_proto)
        self._timed_data_collector.add(sae_msg)

        # collect data for eventual model training
        if self._config.model.start_training:
            finished_training = self._collect_and_train()
            if finished_training:
                self._detector = Detector(self._config)
                self.model_info = ModelInfoCollector(self._config).model_info
            else:
                return None

        tracks, frames = self._timed_data_collector.get_latest_data()
        if len(tracks) == 0:
            return None
        filtered_data = self._detector.filter_tracks(tracks)
        if len(filtered_data) == 0:
            return None
        anomaly_message = self._detector.examine_tracks_for_anomalies(filtered_data, frames)

        if len(anomaly_message.trajectories) == 0:
            return None
        
        # Forward camera geo location
        if sae_msg.frame.HasField('camera_location'):
            anomaly_message.camera_location.CopyFrom(sae_msg.frame.camera_location)

        return self._create_output(anomaly_message)
    
    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)
        return sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, output_anomaly_msg):

        output_anomaly_msg.model_info.CopyFrom(self.model_info)
        return output_anomaly_msg.SerializeToString()
    
    def _collect_and_train(self) -> None:
        """
        If model_info.train is True, keep collecting data until the configured
        hour interval elapses. Then trigger model training once and flip
        model_info.train to False. Resets internal buffers afterward.
        """
        if not self._config.model.start_training:
            # if someone turned training off mid-way, reset state
            if self._train_start_time is not None or self._train_buffer:
                self._reset_training_state()
            return False

        # initialize the window start when training turns on
        if self._train_start_time is None:
            self._train_start_time = datetime.utcnow()
            logger.info(f"Started collecting data for model training at {self._train_start_time.isoformat()} UTC")
            logger.info(f"Will trigger training in {self._config.model.time_interval_train_data} hours.")

        # append current latest snapshot to the buffer
        tracks, _ = self._timed_data_collector.get_latest_data()
        if tracks:  # only store non-empty snapshots
            self._train_buffer.extend(tracks)

        # check elapsed time against configured interval (in hours)
        hours = self._config.model.time_interval_train_data
        if hours <= 0:
            # no valid interval -> train immediately once and flip off
            logger.error("Configured training interval is non-positive; aborting.")
            exit(1)

        if datetime.utcnow() - self._train_start_time >= timedelta(hours=hours):
            self._trigger_training()
            return True
        
        return False

    def _trigger_training(self) -> None:
        """Runs the training hook with buffered data, then flips train=False."""
        # Prepare a flat view for your trainer; adapt as needed

        try:
            self._train_anoamly_detector()
        finally:
            # Always flip & reset, even if training raised, to avoid infinite loops.
            self._config.model.start_training = False
            self._reset_training_state()

    def _reset_training_state(self) -> None:
        self._train_start_time = None
        self._train_buffer.clear()

    
    def _train_anoamly_detector(self) -> None:
        """
        Hook: start your actual model training here.
        Replace with your real trainer (sync or async as fits your system).
        Example:
            self._detector.train(tracks=tracks, frames=frames, config=self._config)
        """
        tracks = self._train_buffer
        logger.info(
            f"Preparing model training with {len(tracks)} tracks collected over {self._config.model.time_interval_train_data}.",
        )
        params = ModelTrainingConfig()

        filtered_data = self._detector.filter_tracks(tracks)
        train, val = random_split_validation_data_from_train_data(filtered_data, val_frac=0.2)

        path_store_prefix = self._config.model.path_train_data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        path_store = f"{path_store_prefix}_{timestamp}"

        logger.info(f"Store training data at {path_store}")
        store_data(train, path_store, time_diff_prediction=params.time_diff_prediction, folder="train", frame_rate=params.framerate)
        store_data(val, path_store, time_diff_prediction=params.time_diff_prediction, folder="test", frame_rate=params.framerate)

        ds = getTorchDataSet(os.path.join(path_store, "train"), pixel_per_axis=params.pixel_per_axis)
        total_train_len = len(ds)
        batch_interval = min(total_train_len//16 -1, 10000)
        
        logger.info(f"Start training with {len(train)} training trajectories and {len(val)} validation trajectories")

        model, _ = training.train_model(path_store, str(self._config.model.weights_path.parent), architecture=params.model_architecture, batch_interval=batch_interval,
                            output_prob=params.output_distribution, pixel_per_axis=params.pixel_per_axis) 
        model.eval()

        logger.info("Inferencing to determine anomaly threshold")
        ds = getTorchDataSet(os.path.join(path_store, "test"), pixel_per_axis=params.pixel_per_axis)
        test = getTorchDataLoader(ds, shuffle=False)
        samples_with_stats = inferencing.inference_with_stats(model, test)
        score_thr = calculate_trajectory_threshold(samples_with_stats, percentage_p=params.percentage_anomaly)

        # Save parameters
        json_dict = params.model_dump(mode="json")

        json_dict.update({
            "time": timestamp,                         # ensure it's a string if needed
            "training_data": str(path_store),          # Path -> string
            "anomaly_threshold": float(score_thr),     # make sure it's JSON-safe
        })

        # Write to file
        with open(self._config.model.parameters_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)


    

def random_split_validation_data_from_train_data(filtered_tracks: Dict[str, list[TrackedObjectPosition]], val_frac: float = 0.2) -> Tuple[List[TrackedObjectPosition], List[TrackedObjectPosition]]:
    """
    Take a total fraction `val_frac` from the training data as validation,
    randomly sampled across the entire dataset. Original order is preserved.

    Returns:
        (train_data, val_data)
    """
    assert 0 < val_frac < 1, "val_frac must be in (0, 1)"

    N = len(filtered_tracks)
    if N == 0:
        raise ValueError("Dataset is empty")

    # Target number of validation entries (rounded, at least 1)
    n_val_total = max(1, round(val_frac * N))

    # Sample indices (not keys) to preserve the original key order on reconstruction
    val_indices = set(random.sample(range(N), n_val_total))

    # Build splits while keeping insertion order
    train_data: Dict[str, List["TrackedObjectPosition"]] = {}
    val_data: Dict[str, List["TrackedObjectPosition"]] = {}

    for i, (k, v) in enumerate(filtered_tracks.items()):
        if i in val_indices:
            val_data[k] = v
        else:
            train_data[k] = v

    return train_data, val_data