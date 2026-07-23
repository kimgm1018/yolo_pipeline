"""
MQTT 클라이언트 스텁 — 긴급 단건 / 배치 이벤트 publish.

실제 브로커 연결 전: dry_run=True 이면 콘솔만 출력.
의존성: pip install paho-mqtt  (Jetson에서)
"""

from __future__ import annotations

import json
from typing import Any


class MqttEventClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        urgent_topic: str = "patrol/events/urgent",
        batch_topic: str = "patrol/events/batch",
        dry_run: bool = True,
    ):
        self.host = host
        self.port = port
        self.urgent_topic = urgent_topic
        self.batch_topic = batch_topic
        self.dry_run = dry_run
        self._client = None

    def connect(self) -> None:
        if self.dry_run:
            print(f"[MQTT dry-run] connect {self.host}:{self.port}")
            return
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client()
        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

    def publish_urgent(self, event: dict[str, Any]) -> None:
        """긴급 lying 등 — robotId 포함 단건 Request."""
        payload = json.dumps(event, ensure_ascii=False)
        if self.dry_run or self._client is None:
            print(f"[MQTT dry-run] {self.urgent_topic}")
            print(payload)
            return
        self._client.publish(self.urgent_topic, payload, qos=1)

    def publish_batch(self, events: list[dict[str, Any]]) -> None:
        """일반 이벤트 일괄 — {\"events\": [...]}."""
        body = {"events": events}
        payload = json.dumps(body, ensure_ascii=False)
        if self.dry_run or self._client is None:
            print(f"[MQTT dry-run] {self.batch_topic}")
            print(payload)
            return
        self._client.publish(self.batch_topic, payload, qos=1)


if __name__ == "__main__":
    c = MqttEventClient(dry_run=True)
    c.connect()
    c.publish_urgent(
        {
            "robotId": 1,
            "eventType": "EMERGENCY_PERSON",
            "eventTitle": "응급상황 의심",
            "eventDetails": "바닥에 누워 있는 사람이 감지되었습니다.",
            "occurredAt": "2026-07-23T00:00:00+09:00",
            "xCoordinate": 10.0,
            "yCoordinate": 20.0,
            "riskLevel": "HIGH",
        }
    )
    c.publish_batch(
        [
            {
                "eventType": "BIN_OVERFLOW",
                "eventTitle": "쓰레기통 포화 감지",
                "eventDetails": "쓰레기통이 가득 찬 상태입니다.",
                "occurredAt": "2026-07-23T00:00:00+09:00",
                "xCoordinate": 10.0,
                "yCoordinate": 20.0,
                "riskLevel": "MEDIUM",
            }
        ]
    )
    c.close()
