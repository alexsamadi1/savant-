from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, ingest, analyze, dashboard, chat

app = FastAPI(
    title="Savant API",
    version="2.0.0",
    description="AI-assisted consulting platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, tags=["Ingest"])
app.include_router(analyze.router, tags=["Analysis"])
app.include_router(dashboard.router, tags=["Dashboard"])
app.include_router(chat.router, tags=["Chat"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
