from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    version="1.0.0",
    description="Enterprise RAG system built with FastAPI"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Enterprise RAG Assistant is running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }
