# bert_api/main.py
import http.client

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
import psycopg2
from tqdm import tqdm

from internal.pydantic_models.pydantic_models import *
from internal.repo.db import SQLiteBlobStorage
from src.recommendation_bert_api.routes_utils import _get_recommendation_explanation, \
    _get_user_recommendation_explanation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage = SQLiteBlobStorage()

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

    # Кэш эмбеддингов пользователей
    app.state.users_data = {} # k : user_id, v : dict(user_id, [quest_id_1, quest_id_2, ..., quest_id_n])
    app.state.profile_embeddings = {} # k : user_id, v : profile_embeddings - то есть mean-всех эмбеддингов квестов этого пользователя

    # ЗАГРУЖАЕМ ДАННЫЕ ИЗ БАЗЫ ПРИ СТАРТЕ
    logger.info("Загружаем данные из базы данных...")

    app.state.quests_data, app.state.quest_embeddings = storage.get_all_quests()
    logger.info(f"Загружено квестов: {len(app.state.quests_data)}")

    app.state.users_data, app.state.profile_embeddings = storage.get_all_users()
    logger.info(f"Загружено пользователей: {len(app.state.users_data)}")

    # Статистика
    stats = storage.get_stats()
    logger.info(f"Статистика БД: {stats}")

    yield

    # Закрываем БД при выключении
    logger.info("Выключаем API...")
    storage.close()


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
            show_progress_bar=False  # <-- ОТКЛЮЧАЕМ ПРОГРЕСС БАР
        )

        # Сохраняем в кэш эмбеддинги
        for idx, quest in enumerate(request.quests):
            app.state.quest_embeddings[quest.id] = embeddings[idx]

        logger.info(f"Добавлено в кеш {len(request.quests)} квестов")

        for quest in request.quests:
            storage.save_quest(quest, app.state.quest_embeddings[quest.id])
            logger.info(f"Добавлен квест в БД {quest}")

        return {"status": "success", "added": len(request.quests)}

    except Exception as e:
        logger.error(f"Ошибка добавления квестов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/add")
async def add_users(request: AddUsersRequest):
    """Добавление пользователей с их quest_ids в кеш + создание эмбеддингов"""
    try:
        for user in request.users:
            user_id = user.user_id

            if user_id in app.state.users_data:
                existing_user = app.state.users_data[user_id]
                existing_quest_ids = existing_user.get("quest_ids", [])

                # Если квесты одинаковые - пропускаем
                if sorted(existing_quest_ids) == sorted(user.quest_ids):
                    logger.info(f"Пользователь {user_id} уже существует с такими же quest_ids, пропускаем")
                    continue
                else:
                    # Квесты изменились - обновляем
                    logger.info(f"Пользователь {user_id} существует, но quest_ids изменились. Обновляем.")

            app.state.users_data[user_id] = {
                "user_id": user_id,
                "quest_ids": user.quest_ids
            }

            # Если у пользователя есть квесты - создаем/обновляем профиль
            if user.quest_ids:
                user_embeddings = []
                valid_quests = []

                # Собираем эмбеддинги существующих квестов
                for quest_id in user.quest_ids:
                    if quest_id in app.state.quest_embeddings:
                        user_embeddings.append(app.state.quest_embeddings[quest_id])
                        valid_quests.append(quest_id)

                # Обновляем список квестов пользователя только валидными (по логике такая ситуация никогда не произойдет)
                if len(valid_quests) != len(user.quest_ids):
                    app.state.users_data[user_id]["quest_ids"] = valid_quests
                    logger.warning(f"У пользователя {user_id} найдено {len(valid_quests)} из {len(user.quest_ids)} квестов")

                # Если пустой список - пропускаем
                if len(user_embeddings) == 0:
                    logger.warning(f"У пользователя {user_id} нет валидных эмбеддингов квестов")
                    continue

                try:
                    # Усредняем эмбеддинги (mean pooling) - Преобразуем список тензоров в один тензор
                    user_embeddings_tensor = torch.stack(user_embeddings)
                    user_profile_embedding = torch.mean(user_embeddings_tensor, dim=0)
                    app.state.profile_embeddings[user_id] = user_profile_embedding

                    # Перезаписываем в БД
                    storage.save_user(user, user_profile_embedding)

                except Exception as e:
                    logger.error(f"Ошибка создания профиля для пользователя {user_id}: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

        return {
            "status": "success",
            "total_users": len(app.state.users_data),
            "total_profiles": len(app.state.profile_embeddings),
            "message": f"Обработано {len(request.users)} пользователей"
        }

    except Exception as e:
        logger.error(f"Ошибка добавления пользователя: {e}")
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
            convert_to_tensor=True,
            show_progress_bar=False  # <-- ОТКЛЮЧАЕМ ПРОГРЕСС БАР
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

@app.post("/api/quests/recommend")
async def recommend_quests(request: RecommendQuestsRequest):
    """Рекомендации квестов на основе истории пользователя"""
    try:
        # Получаем ID квестов пользователя из go-backend (их достает из БД)
        # user_quest_ids = [1, 3, 5, 8] - ID квестов, которые у пользователя уже есть

        if not request.user_quest_ids:
            return {"recommendations": [], "message": "У пользователя нет квестов для рекомендаций"}

        # 1. Получаем эмбеддинги квестов пользователя
        user_embeddings = []
        for quest_id in request.user_quest_ids:
            if quest_id in app.state.quest_embeddings:
                user_embeddings.append(app.state.quest_embeddings[quest_id])

        if not user_embeddings:
            return {"recommendations": [], "message": "Не найдены эмбеддинги для квестов пользователя"}

        # 2. Усредняем эмбеддинги (mean pooling)
        # Преобразуем список тензоров в один тензор
        user_embeddings_tensor = torch.stack(user_embeddings)
        user_profile_embedding = torch.mean(user_embeddings_tensor, dim=0)

        # 3. Ищем похожие квесты (которые пользователь еще не имеет)
        results = []
        for quest_id, quest_embedding in app.state.quest_embeddings.items():
            # Пропускаем квесты, которые уже есть у пользователя
            if quest_id in request.user_quest_ids:
                continue

            # Фильтр по категории, если указана
            if request.category:
                quest_data = app.state.quests_data.get(quest_id)
                if quest_data and quest_data.get('category') != request.category:
                    continue

            # Вычисляем схожесть с профилем пользователя
            score = util.cos_sim(user_profile_embedding, quest_embedding).item()

            if score > 0.2:  # Порог можно настроить
                quest_data = app.state.quests_data.get(quest_id, {})
                results.append({
                    **quest_data,
                    "similarity_score": float(score),
                    "id": quest_id
                })

        # 4. Сортируем и возвращаем топ-K
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # 5. Добавляем объяснения рекомендаций
        enhanced_results = []
        for result in results[:request.top_k]:
            explanation = _get_recommendation_explanation(
                result,
                request.user_quest_ids,
                app.state.quests_data
            )
            result["explanation"] = explanation
            enhanced_results.append(result)

        return {
            "recommendations": enhanced_results,
            "user_profile_info": {
                "quests_count": len(request.user_quest_ids),
                "embedding_dim": user_profile_embedding.shape[0],
                "method": "mean_pooling"
            }
        }

    except Exception as e:
        logger.error(f"Ошибка рекомендаций: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/recommend")
async def recommend_users(request: RecommendUsersRequest):
    cur_user_id = request.user_id

    if cur_user_id not in app.state.users_data:
        raise HTTPException(status_code=http.client.BAD_REQUEST,
                            detail=f"user (with user_id={cur_user_id}) not found in data")

    if cur_user_id not in app.state.profile_embeddings:
        raise HTTPException(status_code=http.client.INTERNAL_SERVER_ERROR,
                            detail=f"user (with user_id={cur_user_id}) profile_embedding not found in data")

    cur_user_profile_embedding = app.state.profile_embeddings[cur_user_id]
    cur_user_quests = app.state.users_data.get(cur_user_id, {}).get("quest_ids", [])

    results = []

    for user_id, profile_embedding in app.state.profile_embeddings.items():
        # пропускаем самого себя
        if user_id == cur_user_id:
            continue

        # Вычисляем схожесть
        score = util.cos_sim(cur_user_profile_embedding, profile_embedding).item()

        if score > 0.2:  # Порог можно настроить
            other_user_quests = app.state.users_data.get(user_id, {}).get("quest_ids", [])

            # Анализируем причины схожести
            explanation = _get_user_recommendation_explanation(
                cur_user_id=cur_user_id,
                cur_user_quests=cur_user_quests,
                other_user_id=user_id,
                other_user_quests=other_user_quests,
                similarity_score=score,
                quests_data=app.state.quests_data
            )

            results.append({
                "user_id": user_id,
                "similarity_score": float(score),
                "explanation": explanation
            })

    # Сортируем и возвращаем топ-K
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_k = min(max(request.top_k, 1), len(results))
    top_k_results = results[:top_k]

    return {
        "status": "success",
        "user_id": cur_user_id,
        "results": top_k_results,
        "total_users_analyzed": len(app.state.profile_embeddings) - 1  # исключая текущего
    }


@app.post("/sync-db")
async def syncDB():
    """Простая миграция данных"""
    try:
        # Конфигурация PostgreSQL
        PG_CONFIG = {
            'host': 'localhost',
            'port': 5432,
            'database': 'becomeoverman',
            'user': 'postgres',
            'password': 'postgres'
        }

        # Подключаемся к PostgreSQL
        logger.info("Подключаемся к PostgreSQL...")
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor()

        # Шаг 1: Мигрируем квесты
        logger.info("Мигрируем квесты...")
        cursor.execute("SELECT id, title, description, category FROM quests")
        quests = cursor.fetchall()

        for quest_id, title, description, category in tqdm(quests, desc="Квесты"):
            # Создаем текст для эмбеддинга
            text = f"{title}. {description or ''}"
            if category:
                text += f". {category}"

            # Создаем эмбеддинг
            embedding = app.state.model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False  # <-- ОТКЛЮЧАЕМ ПРОГРЕСС БАР
            )
            # Сохраняем в SQLite
            quest = Quest(
                id=quest_id,
                title=title,
                description=description,
                category=category
            )

            storage.save_quest(quest, embedding)

            # Сохраняем в кеш
            app.state.quests_data[quest_id] = quest.dict()
            app.state.quest_embeddings[quest_id] = embedding

        # Шаг 2: сохраняем пользователей и подсчитываем их профили
        logger.info("Мигрируем пользователей...")
        cursor.execute("""
            SELECT user_id,
                COALESCE(ARRAY_AGG(quest_id), ARRAY[]::integer[]) as quest_ids
            FROM user_quests
            GROUP BY user_id
        """)

        user_quests_data = cursor.fetchall()

        logger.info(f"Найдено {len(user_quests_data)} пользователей с квестами")

        successful_users = 0
        failed_users = 0

        for user_id, quest_ids in tqdm(user_quests_data, desc="Пользователи"):
            try:
                user = User(
                    user_id=user_id,
                    quest_ids=quest_ids
                )

                # Добавляем в кеш
                app.state.users_data[user_id] = user.dict()

                user_embeddings = []
                for quest_id in quest_ids:
                    if quest_id in app.state.quest_embeddings:
                        user_embeddings.append(app.state.quest_embeddings[quest_id])
                    else:
                        # Если квеста нет в кэше (он был удален или что-то пошло не так)
                        logger.warning(f"Квест {quest_id} пользователя {user_id} не найден в кэше")
                        continue

                if len(user_embeddings) == 0:
                    logger.warning(f"Пользователь {user_id} не получил профиль (нет эмбеддингов)")
                    storage.save_user(user, None)
                    failed_users += 1
                    continue

                # Сохраняем профиль в КЕШ
                # Усредняем эмбеддинги (mean pooling) - Преобразуем список тензоров в один тензор
                user_embeddings_tensor = torch.stack(user_embeddings)
                user_profile_embedding = torch.mean(user_embeddings_tensor, dim=0)
                app.state.profile_embeddings[user_id] = user_profile_embedding

                # Сохраняем юзера и профиль в БД
                storage.save_user(user, user_profile_embedding)
                successful_users += 1

            except Exception as e:
                logger.error(f"Ошибка обработки пользователя {user_id}: {e}")
                failed_users += 1
                continue

        # Закрываем соединения
        cursor.close()
        conn.close()

        # Выводим статистику
        stats = storage.get_stats()
        logger.info(f"\n📊 Статистика после миграции:")
        logger.info(f"   Квестов: {stats['quests']}")
        logger.info(f"   Квестов с эмбеддингами: {stats['quests_with_embeddings']}")
        logger.info(f"   Пользователей: {stats['users']}")
        logger.info(f"   Пользователей с профилями: {stats['users_with_profiles']}")
        logger.info(f"   Размер БД: {stats['db_size_mb']} MB")
        logger.info(f"   Успешных профилей: {successful_users}")
        logger.info(f"   Неудачных профилей: {failed_users}")

        logger.info("\n✅ Миграция завершена!")

        return {
            "status": "success",
            "message": "Миграция данных успешно завершена",
            "stats": {
                "total_quests": stats['quests'],
                "total_users": len(user_quests_data),
                "successful_user_profiles": successful_users,
                "failed_user_profiles": failed_users,
                "quests_with_embeddings": stats['quests_with_embeddings'],
                "users_with_profiles": stats['users_with_profiles']
            }
        }

    except psycopg2.Error as e:
        logger.error(f"Ошибка подключения к PostgreSQL: {e}")
        return {
            "status": "error",
            "message": f"Ошибка подключения к PostgreSQL: {str(e)}",
            "error_type": "database_connection_error"
        }

    except Exception as e:
        logger.error(f"Критическая ошибка при миграции: {e}")
        return {
            "status": "error",
            "message": f"Критическая ошибка при миграции: {str(e)}",
            "error_type": "migration_error"
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )