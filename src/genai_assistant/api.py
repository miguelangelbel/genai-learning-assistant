"""FastAPI backend for the GenAI Learning Assistant.

Provides chat, health-check, and conversation-session endpoints.
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)

from genai_assistant.conversation import conversational_rag

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenAI Learning Assistant"
)

histories = {}


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100
    )

    message: str = Field(
        min_length=1,
        max_length=2000
    )

class ChatResponse(BaseModel):
    session_id: str
    answer: str

class SessionDeleteResponse(BaseModel):
    message: str

class HealthResponse(BaseModel):
    status: str


@app.get("/")
def root():
    return {
        "message": "GenAI Learning Assistant API"
    }

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):
    """
    Generate a conversational RAG response for a session.
    """

    history = histories.setdefault(
        request.session_id,
        []
    )

    try:
        answer = conversational_rag(
            request.message,
            history
        )

    except ContextOverflowError as exc:
        logger.exception("Model context limit exceeded")

        raise HTTPException(
            status_code=400,
            detail="The conversation is too long for the model context window."
        ) from exc

    except ModelRateLimitError as exc:
        logger.exception("Model provider rate limit exceeded")

        raise HTTPException(
            status_code=429,
            detail="The model provider rate limit was exceeded. Try again later."
        ) from exc

    except ModelTimeoutError as exc:
        logger.exception("Model provider timed out")

        raise HTTPException(
            status_code=504,
            detail="The model provider took too long to respond."
        ) from exc

    except (ModelConnectionError, ModelAPIError) as exc:
        logger.exception("Model provider unavailable")

        raise HTTPException(
            status_code=503,
            detail="The model provider is temporarily unavailable."
        ) from exc

    except (
        ModelAuthenticationError,
        ModelPermissionDeniedError,
        ModelNotFoundError,
        ModelInvalidRequestError,
    ) as exc:
        logger.exception("Model configuration or provider request error")

        raise HTTPException(
            status_code=500,
            detail="The model service is incorrectly configured."
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error while generating response")

        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred."
        ) from exc

    return ChatResponse(
        session_id=request.session_id,
        answer=answer
    )

@app.delete(
    "/sessions/{session_id}",
    response_model=SessionDeleteResponse
)
def delete_session(session_id: str):
    """
    Delete the in-memory history associated with a session.
    """

    if session_id not in histories:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    del histories[session_id]

    return SessionDeleteResponse(
        message=f"Session '{session_id}' deleted."
    )

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():
    """
    Return the API health status.
    """
    return HealthResponse(
        status="ok"
    )