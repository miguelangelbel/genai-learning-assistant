import httpx


def check_backend(api_base_url):
    """Return whether the backend is reachable."""
    try:
        response = httpx.get(
            f"{api_base_url}/health",
            timeout=5.0
        )

        response.raise_for_status()
        return True

    except httpx.HTTPError:
        return False


def delete_session(api_base_url, session_id):
    """Delete a backend conversation session."""
    response = httpx.delete(
        f"{api_base_url}/sessions/{session_id}",
        timeout=10.0
    )

    if response.status_code not in (200, 404):
        response.raise_for_status()