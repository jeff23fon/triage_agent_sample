#  type: ignore
import requests


def test_triage_endpoint_integration():
    url = "http://localhost:8082/v1/agents/triage"
    payload = {
        "messages": [{"role": "user", "content": "What is your name?"}],
        "settings": {},
        "conversation_id": "abc123",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "conversation_id" in data
    assert data["conversation_id"] == "abc123"
