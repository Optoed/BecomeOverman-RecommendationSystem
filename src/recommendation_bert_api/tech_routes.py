import logging

import torch
from fastapi import HTTPException

from internal.pydantic_models.pydantic_models import HealthResponse
from src.recommendation_bert_api.main import app, storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tech Routes


@app.get("/api/health")
async def health() -> HealthResponse:
    """Проверка здоровья API"""
    return HealthResponse(
        status="healthy",
        model="paraphrase-multilingual-MiniLM-L12-v2",
        device=app.state.device,
        quests_count=len(app.state.quest_embeddings)
    )


@app.get("/api/stats")
async def get_stats():
    """Статистика"""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
        else:
            memory_allocated = memory_reserved = 0

        db_stats = storage.get_stats()

        return {
            "status": "success",
            "db_stats": db_stats,
            "quests_count": len(app.state.quest_embeddings),
            "embedding_dimension": 384,  # для выбранной модели
            "gpu_memory_allocated_mb": round(memory_allocated, 2),
            "gpu_memory_reserved_mb": round(memory_reserved, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

