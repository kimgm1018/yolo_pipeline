"""
독립 NumPy ByteTrack — Ultralytics/PyTorch 없이 동작.
설정: trackers/bytetrack_stable.yaml 키를 그대로 읽음.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4) xyxy, b (M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clip(0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def _linear_assignment(cost: np.ndarray, thresh: float):
    """Greedy IoU matching (SciPy 없이). returns matches, u_a, u_b"""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    # cost is 1-iou style; convert from iou matrix
    iou = 1.0 - cost
    matches = []
    used_a, used_b = set(), set()
    # sort pairs by iou desc
    flat = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    flat.sort(reverse=True)
    for v, i, j in flat:
        if v < thresh:
            break
        if i in used_a or j in used_b:
            continue
        matches.append((i, j))
        used_a.add(i)
        used_b.add(j)
    u_a = [i for i in range(cost.shape[0]) if i not in used_a]
    u_b = [j for j in range(cost.shape[1]) if j not in used_b]
    return matches, u_a, u_b


class _STrack:
    _count = 0

    def __init__(self, xyxy: np.ndarray, score: float, class_id: int):
        self.xyxy = xyxy.astype(np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        _STrack._count += 1
        self.track_id = _STrack._count
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0
        self.state = "new"  # new / tracked / lost

    def update(self, xyxy, score, class_id, frame_id: int):
        self.xyxy = xyxy.astype(np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        self.frame_id = frame_id
        self.time_since_update = 0
        self.state = "tracked"

    def mark_lost(self):
        self.state = "lost"
        self.time_since_update += 1


class ByteTrackAdapter:
    def __init__(self, tracker_config: str | Path | None = None, **overrides):
        cfg = {
            "track_high_thresh": 0.20,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.15,
            "track_buffer": 30,
            "match_thresh": 0.7,
            "fuse_score": True,
        }
        if tracker_config is not None:
            path = Path(tracker_config)
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for k in cfg:
                    if k in data:
                        cfg[k] = data[k]
        cfg.update(overrides)
        self.track_high_thresh = float(cfg["track_high_thresh"])
        self.track_low_thresh = float(cfg["track_low_thresh"])
        self.new_track_thresh = float(cfg["new_track_thresh"])
        self.track_buffer = int(cfg["track_buffer"])
        self.match_thresh = float(cfg["match_thresh"])
        self.fuse_score = bool(cfg["fuse_score"])

        self.tracked: list[_STrack] = []
        self.lost: list[_STrack] = []
        self.frame_id = 0

    def reset(self):
        self.tracked.clear()
        self.lost.clear()
        self.frame_id = 0
        _STrack._count = 0

    def _match(self, tracks: list[_STrack], dets_xyxy, dets_scores):
        if not tracks or len(dets_xyxy) == 0:
            return [], list(range(len(tracks))), list(range(len(dets_xyxy)))
        tboxes = np.stack([t.xyxy for t in tracks], axis=0)
        iou = _iou_batch(tboxes, dets_xyxy)
        if self.fuse_score:
            iou = iou * dets_scores[None, :]
        cost = 1.0 - iou
        return _linear_assignment(cost, thresh=self.match_thresh)

    def update(self, detections: list[dict], frame_shape=None) -> list[dict]:
        """
        detections: yolo_trt 결과 (track_id 없음)
        return: track_id 포함 detection (확정 track만)
        """
        self.frame_id += 1

        if not detections:
            for t in self.tracked:
                t.mark_lost()
            self.lost.extend(self.tracked)
            self.tracked = []
            self._remove_old()
            return []

        xyxy = np.array([d.get("xyxy", d["bbox"]) for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)
        cls_ids = np.array([d["class_id"] for d in detections], dtype=np.int32)

        high_mask = scores >= self.track_high_thresh
        low_mask = (scores >= self.track_low_thresh) & (~high_mask)

        # pool = tracked + lost
        pool = self.tracked + self.lost

        # 1st association with high score dets
        matches, u_tr, u_det = self._match(pool, xyxy[high_mask], scores[high_mask])
        high_idx = np.where(high_mask)[0]

        activated = []
        for it, idet in matches:
            tr = pool[it]
            di = high_idx[idet]
            tr.update(xyxy[di], scores[di], cls_ids[di], self.frame_id)
            activated.append(tr)

        # remaining tracks
        rem_tracks = [pool[i] for i in u_tr]

        # 2nd association with low score dets
        low_idx = np.where(low_mask)[0]
        matches2, u_tr2, _u_low = self._match(
            rem_tracks, xyxy[low_idx] if len(low_idx) else np.zeros((0, 4), np.float32), scores[low_idx] if len(low_idx) else np.zeros((0,), np.float32)
        )
        for it, idet in matches2:
            tr = rem_tracks[it]
            di = low_idx[idet]
            tr.update(xyxy[di], scores[di], cls_ids[di], self.frame_id)
            activated.append(tr)

        unmatched_tracks = [rem_tracks[i] for i in u_tr2]
        for tr in unmatched_tracks:
            tr.mark_lost()

        # new tracks from unmatched high dets
        for j in u_det:
            di = high_idx[j]
            if scores[di] < self.new_track_thresh:
                continue
            tr = _STrack(xyxy[di], scores[di], cls_ids[di])
            tr.start_frame = self.frame_id
            tr.frame_id = self.frame_id
            tr.state = "tracked"
            activated.append(tr)

        # rebuild lists
        self.tracked = [t for t in activated if t.state == "tracked"]
        lost_now = [t for t in unmatched_tracks if t.state == "lost"]
        # keep previous lost that weren't rematched
        still_lost = []
        activated_ids = {t.track_id for t in self.tracked}
        for t in self.lost:
            if t.track_id in activated_ids:
                continue
            if t not in lost_now:
                t.mark_lost()
                still_lost.append(t)
        self.lost = lost_now + still_lost
        self._remove_old()

        # output tracked only (with track_id)
        name_of = {d["class_id"]: d["class_name"] for d in detections}
        out = []
        for t in self.tracked:
            box = t.xyxy.astype(int).tolist()
            out.append(
                {
                    "track_id": int(t.track_id),
                    "class_id": int(t.class_id),
                    "class_name": name_of.get(t.class_id, str(t.class_id)),
                    "confidence": float(t.score),
                    "bbox": box,
                    "xyxy": box,
                }
            )
        return out

    def _remove_old(self):
        self.lost = [t for t in self.lost if t.time_since_update <= self.track_buffer]
