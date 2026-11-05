from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple
from visionapi.sae_pb2 import SaeMessage, VideoFrame
from movementpredictor import TrackingDataManager, TrackedObjectPosition


@dataclass
class FrameWithTracks():
    frame: VideoFrame                       # frame object from SaeMessage
    tracks: List[TrackedObjectPosition]     # all detections+trackingIDdetected on this frame


class TimedTrajectories:
    """
    Collects trajectories per track id. All times are frame time (frame.timestamp_utc_ms),
    so behaviour is independent of playback speed. Frame time is assumed to be strictly
    monotonic; a frame that is not newer than the previous one is treated as a discontinuity.
    """

    def __init__(self, log_level=logging.INFO, timeout=5, max_length=60):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)

        # timeout in seconds of frame time: if a track id hasn't been updated within this, we close the trajectory
        self.timeout_ms = timeout * 1_000
        # trajectories existing longer than this (seconds of frame time) are closed
        self.max_length_ms = max_length * 1_000

        # per‐id list of track objects seen so far
        self.data = {}          # { track_id: [track1, track2, ...] }
        # per‐id first_seen and last_seen frame times (ms)
        self.timestamps = {}    # { track_id: (first_seen, last_seen) }

        # completed trajectories, ready to be consumed by get_latest_data()
        self.out = []           # [ [track, track, ...], ... ]

        # list of frame objects with their tracks, in chronological order
        self.frames_with_tracks: List[FrameWithTracks] = []   # [ [frame1, [tracks1]], ...]

        # frame time interval (ms) until self.out will be returned when calling self.get_latest_data
        self.out_interval_ms = 10_000
        # frame time (ms) of last self.out return, None -> return on next call
        self.last_out_ms: Optional[int] = None

        # frame time (ms) of the newest frame
        self.current_ts_ms: Optional[int] = None

        # batch flushed because of a frame time discontinuity, returned on next get_latest_data()
        self._pending: Optional[Tuple[list, list]] = None

    def add(self, proto: SaeMessage):
        """Call this for each incoming proto (with frame + raw detections)."""
        frame = proto.frame
        ts = frame.timestamp_utc_ms

        if self.current_ts_ms is not None and ts <= self.current_ts_ms:
            self.log.warning(f"Frame time not strictly monotonic ({self.current_ts_ms} -> {ts}), resetting trajectories")
            self._flush_on_discontinuity()
        self.current_ts_ms = ts

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
                self.timestamps[tid] = (ts, ts)
            else:
                first_seen, _ = self.timestamps[tid]
                # update last_seen only
                self.timestamps[tid] = (first_seen, ts)

        # evict any trajectories that have timed out
        self._evict_expired(ts)

    def _evict_expired(self, now_ms):
        """Move any timed-out or over-long trajectories to self.out, then clean up frames."""
        # Find IDs that either haven’t been seen within the timeout or have existed longer than max length
        expired = [
            tid
            for tid, (first_seen, last_seen) in self.timestamps.items()
            if (now_ms - last_seen) > self.timeout_ms
            or (now_ms - first_seen) > self.max_length_ms
        ]

        # Move them to self.out, remove from data+timestamps
        for tid in expired:
            self.out.append(self.data.pop(tid))
            self.timestamps.pop(tid)

        # Clean up old frames
        self._cleanup_frames()

    def _cleanup_frames(self):
        """
        Retain frames needed by active and completed tracks (plus 2 seconds before them),
        otherwise keep the last 2 seconds of frames.
        """

        if not self.timestamps and not self.out:
            # no active tracks -> no frames needed, but keep last 2 seconds
            cutoff_ms = self.current_ts_ms - 2_000

        else:
            # Determine the earliest frame time required by active and completed tracks - 2 seconds
            candidates = [first for first, _ in self.timestamps.values()]
            candidates += [track.capture_ts for trajectory in self.out for track in trajectory]
            cutoff_ms = min(candidates) - 2_000

        # Drop frames older than cutoff_ms
        for i, frame_with_tracks in enumerate(self.frames_with_tracks):
            if frame_with_tracks.frame.timestamp_utc_ms >= cutoff_ms:
                self.frames_with_tracks = self.frames_with_tracks[i:]
                return

        # If none remain, clear all
        self.frames_with_tracks.clear()

    def _flush_on_discontinuity(self):
        """Close all active trajectories, stash them as pending batch and reset all state."""
        self.out.extend(self.data.values())
        track_list, frame_list = self._build_batch()
        if self._pending is not None:
            track_list = self._pending[0] + track_list
            frame_list = self._pending[1] + frame_list
        self._pending = (track_list, frame_list) if track_list else None

        self.data.clear()
        self.timestamps.clear()
        self.frames_with_tracks.clear()
        self.last_out_ms = None

    def _build_batch(self):
        """Build the output from self.out plus context vehicles and frames, then clear self.out."""
        if len(self.out) == 0:
            return [], []

        frame_list = []

        track_list = [trk for traj in self.out for trk in traj]
        target_uuids = { trk.uuid for trk in track_list }
        target_timestamps = { trk.capture_ts for trk in track_list }
        target_min_ts = min(target_timestamps)
        target_max_ts = max(target_timestamps)

        # add context vehicles, change their object type to -1 to not be confused a target vehicle later
        for frame_with_tracks in self.frames_with_tracks:
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

    def get_latest_data(self):
        """
        Returns:
          - track_list: a flat list of all completed tracks (from self.out) and context vehicle tracks
          - frames:       the list of frames belonging to these tracks
        """
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self.last_out_ms = self.current_ts_ms
            return pending

        if self.last_out_ms is not None and self.current_ts_ms - self.last_out_ms < self.out_interval_ms:
            return [], []

        self.last_out_ms = self.current_ts_ms

        return self._build_batch()

    def calc_frame_border_threshold(self, frame):
        wh = frame.shape.width / frame.shape.height
        return [0.01, wh * 0.01]
