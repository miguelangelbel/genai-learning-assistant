from fastapi import FastAPI


app = FastAPI(
    title="GenAI Learning Assistant"
)

@app.get("/")
def root():
    return {
        "message": "GenAI Learning Assistant API"
    }