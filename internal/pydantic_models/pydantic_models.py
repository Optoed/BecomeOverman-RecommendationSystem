# Модели данных Pydantic
from pydantic import BaseModel
from typing import List, Optional


class Quest(BaseModel):
    id: int
    title: str
    description: str
    category: Optional[str] = None
    # TODO: потом можно добавить rarity, difficulty, price


class AddQuestsRequest(BaseModel):
    quests: List[Quest]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[dict]
    query_embedding_size: int
    search_time_ms: float


class SimilarQuestsRequest(BaseModel):
    quest_id: int
    top_k: int = 5


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    quests_count: int
