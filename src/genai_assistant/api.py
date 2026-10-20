from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="GenAI Learning Assistant"
)

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000
    )

class ChatResponse(BaseModel):
    answer: str

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
    return ChatResponse(
        answer=f"You said: {request.message}"
    )


# @app.post("/chat")
# def chat(request: ChatRequest):
#     return {
#         "answer": f"You said: {request.message}"
#     }

# @app.post("/chat")
# def chat(request: ChatRequest):

#     print(type(request))
#     print(request)
#     print(request.message)

#     return {
#         "answer": f"You said: {request.message}"
#     }