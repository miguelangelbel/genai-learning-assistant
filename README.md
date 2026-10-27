# GenAI Learning Assistant

End-to-end GenAI learning project focused on building and understanding the main components of modern LLM applications.

The project evolves progressively from direct LLM API usage to RAG, LangChain, conversational assistants, APIs, agents and fine-tuning.

## Objectives

The main goal is to reinforce and connect concepts related to:

- LLM APIs
- Prompt engineering
- Embeddings
- Semantic similarity
- Vector search
- Retrieval-Augmented Generation (RAG)
- LangChain
- Conversational memory
- FastAPI
- Chat interfaces
- Tool calling and agents
- Fine-tuning
- LLM and RAG evaluation

Core concepts are implemented manually whenever possible before introducing higher-level frameworks.

## Roadmap

| Version | Topic | Status |
| ------- | ----- | ------ |
| v0.1 | LLM API Basics | Finished |
| v0.2 | Embeddings and Semantic Similarity | Finished |
| v0.3 | Manual RAG | Finished |
| v0.4 | RAG with LangChain | Finished |
| v0.5 | Conversational Assistant | Finished |
| v0.6 | FastAPI Backend | Finished |
| v0.7 | Chat Interface | Finished |
| v0.8 | Tools and Agents | Next |
| v0.9 | Fine-tuning | |
| v1.0 | Evaluation and Final Integration | |

## Tech Stack

- Python
- Google Gemini
- Sentence Transformers
- LangChain
- FastAPI
- Streamlit
- HTTPX
- Hugging Face
- PEFT / LoRA
- GitHub Codespaces
- Google Colab

## Repository Structure

```text
genai-learning-assistant/
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- notebooks/
|
|-- src/
|   `-- genai_assistant/
|       |-- __init__.py
|       |-- api.py
|       |-- api_client.py
|       |-- conversation.py
|       `-- frontend.py
|
|-- tests/
|   |-- test_api.py
|   `-- test_api_client.py
|
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- requirements-dev.txt
|-- LICENSE
`-- README.md
```

## Development Environment

Application development is performed using VS Code and GitHub Codespaces, while Google Colab is used for notebook-based experimentation and GPU-dependent workloads.

GitHub is used for source control through a feature branch, pull request and merge workflow.

The project aims to remain at zero monetary cost by prioritizing open-source tools and free-tier services.

## Current Status

### v0.6 — FastAPI Backend - Finished

The conversational RAG pipeline is exposed through a FastAPI backend with:

- Pydantic request and response validation
- Session-specific conversational memory
- Session deletion
- Health checks
- Model and provider error handling
- Lazy initialization of RAG components
- Automated API tests

Conversation state and vector storage are currently maintained in memory and are intended for local development and learning.

### v0.7 — Chat Interface - Finished

The FastAPI backend is now connected to a Streamlit frontend with:

- Chat-style user interface
- Frontend session state
- Session-specific backend conversations
- Backend health monitoring
- Conversation reset
- Loading states
- Graceful HTTP error handling
- Markdown-formatted responses

The frontend and backend run as separate applications and communicate through HTTP.

**Next:** v0.8 — Tools and Agents.

## License

MIT License.