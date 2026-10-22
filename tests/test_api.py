import pytest

from fastapi.testclient import TestClient

from genai_assistant.api import app, histories


client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }

def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        json={
            "session_id": "test-session",
            "message": ""
        }
    )

    assert response.status_code == 422

def test_chat(monkeypatch):
    def fake_conversational_rag(question, history):
        return "Mocked answer"

    monkeypatch.setattr(
        "genai_assistant.api.conversational_rag",
        fake_conversational_rag
    )

    response = client.post(
        "/chat",
        json={
            "session_id": "test-session",
            "message": "What is Amazon S3?"
        }
    )

    assert response.status_code == 200

    assert response.json() == {
        "session_id": "test-session",
        "answer": "Mocked answer"
    }

def test_delete_session(monkeypatch):
    def fake_conversational_rag(question, history):
        history.append(question)
        return "Mocked answer"

    monkeypatch.setattr(
        "genai_assistant.api.conversational_rag",
        fake_conversational_rag
    )

    session_id = "delete-test"

    client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "Hello"
        }
    )

    response = client.delete(
        f"/sessions/{session_id}"
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": f"Session '{session_id}' deleted."
    }

    response = client.delete(
        f"/sessions/{session_id}"
    )

    assert response.status_code == 404   

@pytest.fixture(autouse=True)
def clear_histories():
    histories.clear()
    yield
    histories.clear()