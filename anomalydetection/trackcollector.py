from dataclasses import dataclass
import time
import logging
from datetime import datetime, timedelta
from typing import List
from visionapi.sae_pb2 import SaeMessage, VideoFrame
from movementpredictor import TrackingDataManager, TrackedObjectPosition


@dataclass
class FrameWithTracks():
    frame: VideoFrame                       # frame object from SaeMessage
    tracks: List[TrackedObjectPosition]     # all detections+trackingIDdetected on this frame 


class TimedTrajectories:
    def __init__(self, log_level=logging.INFO, timeout=5):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)

        # timeout in seconds: if a track id hasn't been updated within this, we close the trajectory
        self.timeout = timeout

        # per‐id list of track objects seen so far
        self.data = {}          # { track_id: [track1, track2, ...] }
        # per‐id first_seen and last_seen times (epoch seconds)
        self.timestamps = {}    # { track_id: (first_seen, last_seen) }

        # completed trajectories, ready to be consumed by get_latest_data()
        self.out = []           # [ [track, track, ...], ... ]

        # list of frame objects with their tracks, in chronological order
        self.frames_with_tracks: List[FrameWithTracks] = []   # [ [frame1, [tracks1]], ...]    

        # milliseconds waiting intervall until self.out will be returnd when calling self.get_lastest_data
        self.out_time = timedelta(milliseconds=10000)
        # timestamp of last self.out return in milliseconds
        self.last_out = datetime.min 

    def add(self, proto: SaeMessage):
        """Call this for each incoming proto (with frame + raw detections)."""
        current_time = time.time()
        frame = proto.frame

        # extract new tracks from this frame
        trackManager = TrackingDataManager()
        tracks = trackManager.extract_tracked_objects(proto, extracted_tracks=[], inferencing=True)

        self.frames_with_tracks.append(FrameWithTracks(frame=frame, tracks=tracks))

        # update per‐track data & timestamps
        for tr in tracks:
            tid = tr.uuid
            self.data.setdefault(tid, []).append(tr)

            if tid not in self.timestamps:
                # first time we see this ID
                self.timestamps[tid] = (current_time, current_time)
            else:
                first_seen, _ = self.timestamps[tid]
                # update last_seen only
                self.timestamps[tid] = (first_seen, current_time)

        # evict any trajectories that have timed out
        self._evict_expired(current_time)

    def _evict_expired(self, current_time):
        """Move any timed-out or over-long trajectories to self.out, then clean up frames."""
        # Find IDs that either haven’t been seen within `timeout` s or have existed > 60 s
        expired = [
            tid
            for tid, (first_seen, last_seen) in self.timestamps.items()
            if (current_time - last_seen) > self.timeout
            or (current_time - first_seen) > 60
        ]

        # Move them to self.out, remove from data+timestamps
        for tid in expired:
            self.out.append(self.data.pop(tid))
            self.timestamps.pop(tid)

        # Clean up old frames
        self._cleanup_frames()

    def _cleanup_frames(self):
        """
        Retain frames needed by active tracks and always keep the last 2 seconds of frames.
        """

        if not self.timestamps and not self.out:
            # no active tracks -> no frames needed, but keep last 2 seconds
            now_ms = time.time() * 1_000
            cutoff_ms = now_ms - 2_0000
        
        else:
            #Determine the earliest frame time required by active tracks - 2 seconds
            cutoff_s = None
            if self.timestamps:
                cutoff_s = min(first for first, _ in self.timestamps.values())
            if self.out:
                cutoff_out_s = min(track.capture_ts/1000 for trajectory in self.out for track in trajectory) 
                cutoff_s = cutoff_out_s if cutoff_s is None else min(cutoff_s, cutoff_out_s)

            cutoff_ms = cutoff_s * 1_000 - 2_000


        # Drop frames older than cutoff_ms
        for i, frame_with_tracks in enumerate(self.frames_with_tracks):
            if frame_with_tracks.frame.timestamp_utc_ms >= cutoff_ms:
                self.frames_with_tracks = self.frames_with_tracks[i:]
                return

        # If none remain, clear all
        self.frames_with_tracks.clear()

    def get_latest_data(self):
        """
        Returns:
          - track_list: a flat list of all completed tracks (from self.out) and context vehicle tracks
          - frames:       the list of remaining frames at this point
        """
        if datetime.now() - self.last_out < self.out_time:
            return [], []
        
        self.last_out = datetime.now()

        if len(self.out) == 0:
            return [], []

        frames_with_tracks = self.frames_with_tracks.copy()
        frame_list = []

        track_list = [trk for traj in self.out for trk in traj]
        target_uuids = { trk.uuid for trk in track_list }
        target_timestamps = { trk.capture_ts for trk in track_list }
        target_min_ts = min(target_timestamps)
        target_max_ts = max(target_timestamps)

        # add context vehicles, change their object type to -1 to not be confused a target vehicle later
        for frame_with_tracks in frames_with_tracks:
            # only return frames and context data in between 2 sec before and after target trajectory
            if frame_with_tracks.frame.timestamp_utc_ms > target_min_ts - 2_000 and  frame_with_tracks.frame.timestamp_utc_ms < target_max_ts + 2_000:         
                frame_list.append(frame_with_tracks.frame)

                for track in frame_with_tracks.tracks:
                    if track.uuid not in target_uuids:
                        track_cp = track.model_copy()
                        track_cp.class_id = -1
                        track_list.append(track_cp)

        # clear out for next round
        self.out.clear()

        return track_list, frame_list

    def calc_frame_border_threshold(self, frame):
        wh = frame.shape.width / frame.shape.height
        return [0.01, wh * 0.01]