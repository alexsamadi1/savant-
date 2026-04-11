from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import query, documents, health, admin

app = FastAPI(
    title="Savant API",
    version="1.0.0",
    description="RAG-powered knowledge retrieval for GovCon organizations"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(query.router, tags=["Query"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(admin.router, tags=["Admin"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
