import httpx

from genai_assistant.api_client import (
    check_backend,
    delete_session,
)


def test_check_backend_success(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            pass

    def fake_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        "genai_assistant.api_client.httpx.get",
        fake_get
    )

    assert check_backend("http://test") is True


def test_check_backend_failure(monkeypatch):
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(
        "genai_assistant.api_client.httpx.get",
        fake_get
    )

    assert check_backend("http://test") is False

def test_delete_session_success(monkeypatch):
    class MockResponse:
        status_code = 200

        def raise_for_status(self):
            raise AssertionError(
                "raise_for_status should not be called"
            )

    def fake_delete(url, timeout):
        assert url == (
            "http://test/sessions/test-session"
        )
        assert timeout == 10.0

        return MockResponse()

    monkeypatch.setattr(
        "genai_assistant.api_client.httpx.delete",
        fake_delete
    )

    delete_session(
        "http://test",
        "test-session"
    )


def test_delete_session_not_found(monkeypatch):
    class MockResponse:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError(
                "raise_for_status should not be called for 404"
            )

    def fake_delete(url, timeout):
        assert url == (
            "http://test/sessions/missing-session"
        )
        assert timeout == 10.0

        return MockResponse()

    monkeypatch.setattr(
        "genai_assistant.api_client.httpx.delete",
        fake_delete
    )

    delete_session(
        "http://test",
        "missing-session"
    )