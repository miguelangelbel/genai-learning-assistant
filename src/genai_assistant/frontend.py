"""Streamlit frontend for the GenAI Learning Assistant.

Provides the chat interface and communicates with the FastAPI backend
for health checks, conversational responses, and session management.
"""


import os
import uuid

import httpx
import streamlit as st

from genai_assistant.api_client import (
    check_backend,
    delete_session,
)


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)



def reset_conversation(backend_available):
    """Reset frontend state and backend conversation history."""
    if backend_available:
        try:
            delete_session(
                API_BASE_URL,
                st.session_state.session_id
            )
        except httpx.HTTPError:
            pass

    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.last_error = None

    st.rerun()

    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.last_error = None

    st.rerun()


st.set_page_config(
    page_title="GenAI Learning Assistant",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)


# Session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_error" not in st.session_state:
    st.session_state.last_error = None


backend_available = check_backend(API_BASE_URL)


# Sidebar

with st.sidebar:
    st.title("GenAI Assistant")

    if st.button(
        "New conversation",
        use_container_width=True
    ):
        reset_conversation(backend_available)

    st.divider()

    if backend_available:
        st.markdown("🟢 **Backend connected**")
    else:
        st.markdown("🔴 **Backend unavailable**")

    st.caption(
        "Conversational RAG assistant built with "
        "Streamlit, FastAPI and LangChain."
    )


# Main page

st.title("GenAI Learning Assistant")

st.caption(
    "Conversational assistant powered by RAG."
)


# Empty state

if not st.session_state.messages:
    st.info(
        "Ask me something about the AWS knowledge base. "
        "For example: **What is Amazon S3?**"
    )


# Existing conversation

for message in st.session_state.messages:
    avatar = (
        "🧑"
        if message["role"] == "user"
        else "🤖"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(message["content"])


# Previous frontend error

if st.session_state.last_error:
    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):
        st.error(st.session_state.last_error)


# Chat input

placeholder = (
    "Ask something about AWS"
    if backend_available
    else "Backend unavailable"
)

prompt = st.chat_input(
    placeholder,
    disabled=not backend_available
)


if prompt:
    # Clear any previous frontend error.
    st.session_state.last_error = None

    # Store and render the user's message.
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message(
        "user",
        avatar="🧑"
    ):
        st.markdown(prompt)

    # Request assistant response.
    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):
        try:
            with st.spinner("Thinking..."):
                response = httpx.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt
                    },
                    timeout=60.0
                )

                response.raise_for_status()
                answer = response.json()["answer"]

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except httpx.ConnectError:
            st.session_state.last_error = (
                "The backend is currently unavailable. "
                "Please try again later."
            )
            st.error(st.session_state.last_error)

        except httpx.TimeoutException:
            st.session_state.last_error = (
                "The backend took too long to respond."
            )
            st.error(st.session_state.last_error)

        except httpx.HTTPStatusError:
            st.session_state.last_error = (
                "The backend returned an error while "
                "processing the request."
            )
            st.error(st.session_state.last_error)

        except httpx.HTTPError:
            st.session_state.last_error = (
                "An unexpected communication error occurred."
            )
            st.error(st.session_state.last_error)