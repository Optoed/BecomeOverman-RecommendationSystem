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

# Users
# ---->
class User(BaseModel):
    user_id: int
    quest_ids: List[int]


class AddUsersRequest(BaseModel):
    users: List[User]

# <----


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


class RecommendQuestsRequest(BaseModel):
    user_quest_ids: List[int] # ID квестов, которые уже есть у пользователя
    top_k: int = 10
    category: Optional[str] = None # Опциональная фильтрация по категории


class RecommendUsersRequest(BaseModel):
    user_id: int  # можем quests_ids не указывать
    top_k: int = 10


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    quests_count: int
