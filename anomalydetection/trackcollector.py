import logging
from movementpredictor import TrackingDataManager


class TrackCollector:
    border_threshold = None

    def __init__(self, log_level):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)
        self.tracks = []
        self.frames = []
        self.out_time_in_ms = 10000

    def add(self, proto):
        frame = proto.frame
        self.frames.append(frame)
        
        trackManager = TrackingDataManager()
        self.tracks = trackManager.extract_tracked_objects(proto, self.tracks, inferencing=True)

    def get_latest_data(self):
        if self.frames[-1].timestamp_utc_ms - self.frames[0].timestamp_utc_ms < self.out_time_in_ms:
            return [], []
        
        track_list = self.tracks.copy()
        frame_list = self.frames.copy()

        self.frames = self.frames[-5:]
        timestamps = [frame.timestamp_utc_ms for frame in self.frames]
        self.tracks = [track for track in self.tracks if track.capture_ts in timestamps]

        return track_list, frame_list

    def calc_frame_border_threshold(self, frame):
        width_height = frame.shape.width / frame.shape.height
        return [0.01, width_height * 0.01]