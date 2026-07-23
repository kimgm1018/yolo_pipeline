"""이벤트 생성 계층 — 대상 클래스 필터, 중복 제거, API JSON 생성."""

from datetime import datetime

from config import (
    ALLOWED_PLATES,
    DEFER_EVENT_CLASSES,
    EVENT_INFO,
    UNMATCHED_PLATE_EVENT,
    URGENT_CLASSES,
)


class EventManager:
    def __init__(self):
        self.created = set()

    def create_events(self, detections, x, y, robot_id):
        """
        Returns:
            batch_events: 일괄 전송용 이벤트 목록 (robotId 없음)
            urgent_events: 즉시 단건 전송용 이벤트 목록 (robotId 포함)
        """
        batch_events = []
        urgent_events = []

        for detection in detections:
            class_name = detection["class_name"]
            track_id = detection["track_id"]

            if class_name in DEFER_EVENT_CLASSES:
                continue
            if class_name not in EVENT_INFO or track_id is None:
                continue

            event_key = (class_name, track_id)
            if event_key in self.created:
                continue

            info = EVENT_INFO[class_name]
            occurred_at = datetime.now().astimezone().isoformat()

            if class_name in URGENT_CLASSES:
                urgent_events.append(
                    {
                        "robotId": robot_id,
                        "eventType": info["eventType"],
                        "eventTitle": info["eventTitle"],
                        "eventDetails": info["eventDetails"],
                        "occurredAt": occurred_at,
                        "xCoordinate": x,
                        "yCoordinate": y,
                        "riskLevel": info["riskLevel"],
                    }
                )
            else:
                batch_events.append(
                    {
                        "eventType": info["eventType"],
                        "eventTitle": info["eventTitle"],
                        "eventDetails": info["eventDetails"],
                        "occurredAt": occurred_at,
                        "xCoordinate": x,
                        "yCoordinate": y,
                        "riskLevel": info["riskLevel"],
                    }
                )

            self.created.add(event_key)

        return batch_events, urgent_events

    def create_unmatched_plate_events(self, plates, x, y):
        """
        OCR로 모은 번호판 중 허용 목록에 없는 것만 배치 이벤트로 생성.

        plates: [{"track_id", "plate", "confidence"}, ...]
        """
        events = []
        info = UNMATCHED_PLATE_EVENT

        for item in plates:
            plate = item["plate"]
            track_id = item["track_id"]

            if not plate:
                continue
            if plate in ALLOWED_PLATES:
                continue

            # 같은 번호 문자열은 한 번만
            event_key = ("unmatched_plate", plate)
            if event_key in self.created:
                continue

            events.append(
                {
                    "eventType": info["eventType"],
                    "eventTitle": info["eventTitle"],
                    "eventDetails": f"{info['eventDetails']} (번호: {plate}, track: {track_id})",
                    "occurredAt": datetime.now().astimezone().isoformat(),
                    "xCoordinate": x,
                    "yCoordinate": y,
                    "riskLevel": info["riskLevel"],
                }
            )
            self.created.add(event_key)

        return events

    def reset(self):
        self.created.clear()
