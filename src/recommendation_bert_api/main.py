# bert_api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import torch
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import logging
from contextlib import asynccontextmanager

# Настройка логирования
from internal.pydantic_models.pydantic_models import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NLP_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Модель загружается при старте и живет в памяти
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загрузка модели при старте
    logger.info("Загружаем модель BERT...")
    app.state.model = SentenceTransformer(NLP_MODEL)
    app.state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    app.state.model.to(app.state.device)
    logger.info(f"Модель загружена на {app.state.device}")

    # Кэш эмбеддингов квестов
    app.state.quest_embeddings = {}
    app.state.quests_data = {}

    yield

    # Очистка при выключении
    logger.info("Выключаем API...")


app = FastAPI(title="Recommendation BERT API", lifespan=lifespan)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React frontend
        "http://localhost:8080",  # Go backend (если запущен локально)
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
        "*"
        # Добавь домены для продакшена
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы: GET, POST, PUT, DELETE и т.д.
    allow_headers=["*"],  # Разрешаем все заголовки
    expose_headers=["*"],  # Показываем все заголовки в ответе
    max_age=3600,  # Кэширование preflight запросов на 1 час
)


# Эндпоинты
@app.post("/api/quests/add")
async def add_quests(request: AddQuestsRequest):
    """Добавить квесты для поиска"""
    try:
        quest_texts = []
        for quest in request.quests:
            text = f"{quest.title}. {quest.description}"
            if quest.category:
                text += f". {quest.category}"
            quest_texts.append(text)
            app.state.quests_data[quest.id] = quest.dict()

        # Создаем эмбеддинги
        embeddings = app.state.model.encode(
            quest_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        # Сохраняем в кэш
        for idx, quest in enumerate(request.quests):
            app.state.quest_embeddings[quest.id] = embeddings[idx]

        logger.info(f"Добавлено {len(request.quests)} квестов")
        return {"status": "success", "added": len(request.quests)}

    except Exception as e:
        logger.error(f"Ошибка добавления квестов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search(request: SearchRequest) -> SearchResponse:
    """Поиск квестов по семантической близости"""
    import time
    start_time = time.time()

    try:
        if not app.state.quest_embeddings:
            raise HTTPException(status_code=400, detail="Сначала добавьте квесты")

        # Кодируем запрос
        query_embedding = app.state.model.encode(
            request.query,
            convert_to_tensor=True
        )

        # Сравниваем со всеми квестами
        results = []
        for quest_id, quest_embedding in app.state.quest_embeddings.items():
            # Фильтр по категории
            if request.category:
                quest_data = app.state.quests_data.get(quest_id)
                if quest_data and quest_data.get('category') != request.category:
                    continue

            # Вычисляем схожесть
            score = util.cos_sim(query_embedding, quest_embedding).item()

            if score > 0.1:  # Порог релевантности
                quest_data = app.state.quests_data.get(quest_id, {})
                results.append({
                    **quest_data,
                    "similarity_score": float(score),
                    "id": quest_id
                })

        # Сортируем по убыванию схожести
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results[:request.top_k],
            query_embedding_size=query_embedding.shape[0],
            search_time_ms=round(search_time, 2)
        )

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/similar")
async def find_similar(request: SimilarQuestsRequest):
    """Найти похожие квесты"""
    try:
        if request.quest_id not in app.state.quest_embeddings:
            raise HTTPException(status_code=404, detail="Квест не найден")

        target_embedding = app.state.quest_embeddings[request.quest_id]

        results = []
        for quest_id, quest_embedding in app.state.quest_embeddings.items():
            if quest_id == request.quest_id:
                continue

            score = util.cos_sim(target_embedding, quest_embedding).item()

            if score > 0.3:  # Порог для похожих
                quest_data = app.state.quests_data.get(quest_id, {})
                results.append({
                    **quest_data,
                    "similarity_score": float(score),
                    "id": quest_id
                })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return {"similar_quests": results[:request.top_k]}

    except Exception as e:
        logger.error(f"Ошибка поиска похожих: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
    else:
        memory_allocated = memory_reserved = 0

    return {
        "quests_count": len(app.state.quest_embeddings),
        "embedding_dimension": 384,  # для выбранной модели
        "gpu_memory_allocated_mb": round(memory_allocated, 2),
        "gpu_memory_reserved_mb": round(memory_reserved, 2)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )