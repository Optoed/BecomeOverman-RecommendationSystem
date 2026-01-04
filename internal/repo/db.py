# sqlite_blob_storage.py
import logging
import sqlite3
import pickle
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor

from internal.pydantic_models.pydantic_models import Quest, User
from src.recommendation_bert_api.main import logger


class SQLiteBlobStorage:
    """SQLite с BLOB для хранения эмбеддингов и данных"""

    def __init__(self, db_path: str = "recommendation_service_become_overman_storage.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode = WAL")  # Для лучшей производительности
        self._init_tables()

    def _init_tables(self):
        """Создаем оптимизированные таблицы"""
        cursor = self.conn.cursor()

        # Таблица квестов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quests (
                id INTEGER PRIMARY KEY,
                -- Метаданные как JSON (легко расширять)
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT -- может быть пустым
                -- Эмбеддинг как BLOB (бинарный, компактный)
                embedding_blob BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Таблица пользователей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                -- Квесты пользователя как JSON
                quest_ids_json TEXT NOT NULL DEFAULT '[]',
                -- Профиль как BLOB
                profile_blob BLOB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    # ========== CRUD операции ==========

    def get_all_quests(self) -> Tuple[Dict[int, Dict], Dict[int, torch.Tensor]]:
        """Получает все квесты"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, description, category, embedding_blob FROM quests")

        quests_data = {}
        quest_embeddings = {}

        for row in cursor.fetchall():
            quest_id, title, description, category, embedding_blob = row

            quests_data[quest_id] = {
                'id': quest_id,
                'title': title,
                'description': description,
                'category': category
            }

            if embedding_blob:
                try:
                    embedding_np = pickle.loads(embedding_blob)
                    quest_embeddings[quest_id] = torch.tensor(embedding_np)
                except Exception as e:
                    logger.error(f"Error loading embedding for quest {quest_id}: {e}")

        logger.info(f"Loaded {len(quests_data)} quests from database")
        return quests_data, quest_embeddings

    def save_quest(self, quest: Quest, embedding: Optional[torch.Tensor] = None):
        """Сохраняет или обновляет квест"""
        cursor = self.conn.cursor()

        embedding_blob = pickle.dumps(embedding) if embedding is not None else None

        cursor.execute("""
            INSERT OR REPLACE INTO quests 
            (id, title, description, category, embedding_blob) 
            VALUES (?, ?, ?, ?, ?)
        """, (
            quest.id,
            quest.title,
            quest.description,
            quest.category,
            embedding_blob
        ))

        self.conn.commit()

    def get_all_users(self) -> tuple[dict[Any, dict[str, list[Any] | Any]], dict[Any, Tensor]]:
        """Получает всех пользователей"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id, quest_ids_json, profile_blob FROM users")

        users_data = {}  # user_id -> список квестов
        profile_embeddings = {}  # user_id -> профиль

        for row in cursor.fetchall():
            user_id, quest_ids_json, profile_blob = row

            # Восстанавливаем список квестов
            try:
                quest_ids = json.loads(quest_ids_json) if quest_ids_json else []
            except (json.JSONDecodeError, TypeError):
                quest_ids = []
                logger.warning(f"Invalid JSON for user {user_id}")

            users_data[user_id] = {
                "user_id": user_id,
                "quest_ids": quest_ids
            }

            # Восстанавливаем профиль
            if profile_blob:
                try:
                    profile_np = pickle.loads(profile_blob)
                    profile_embeddings[user_id] = torch.tensor(profile_np)
                except Exception as e:
                    logger.error(f"Error loading profile for user {user_id}: {e}")

        logger.info(f"Loaded {len(users_data)} users from database")
        return users_data, profile_embeddings

    def save_user(self, user: User, profile: Optional[torch.Tensor] = None):
        """Сохраняет или обновляет пользователя"""
        cursor = self.conn.cursor()

        profile_blob = pickle.dumps(profile) if profile is not None else None
        quest_ids_json = json.dumps(user.quest_ids) if user.quest_ids is not None else '[]'

        cursor.execute("""
            INSERT OR REPLACE INTO users 
            (user_id, quest_ids_json, profile_blob) 
            VALUES (?, ?, ?)
        """, (
            user.user_id,
            quest_ids_json,
            profile_blob
        ))

        self.conn.commit()

    # TODO: unused пока что - тк save_user(user, new_profile)
    # def update_user_quests_and_profile(self, user: User):
    #     """Добавляет квест пользователю и пересчитывает профиль"""
    #     cursor = self.conn.cursor()
    #
    #     # 1. Получаем текущие данные пользователя
    #     cursor.execute(
    #         "SELECT quest_ids_json, profile_blob FROM users WHERE user_id = ?",
    #         (user.user_id,)
    #     )
    #
    #     result = cursor.fetchone()
    #     if not result:
    #         # Пользователь не найден
    #         logger.info(f"User {user.user_id} not found")
    #         return False
    #
    #     # 2. Обновляем список квестов
    #     if result[0] is not None:
    #         current_quests = json.loads(result[0])
    #     else:
    #         current_quests = []
    #
    #     new_quests = user.quest_ids
    #     if sorted(current_quests) == sorted(new_quests):
    #         return True  # Уже есть
    #
    #     # 3. Получаем эмбеддинги ВСЕХ квестов пользователя
    #     cursor.execute("""
    #         SELECT embedding_blob FROM quests
    #         WHERE id IN ({})
    #     """.format(','.join(['?'] * len(new_quests))), new_quests)
    #
    #     embeddings = []
    #     for (blob,) in cursor.fetchall():
    #         if blob:
    #             embeddings.append(pickle.loads(blob))
    #         else:
    #             logger.warning(f"quest_embedding_blob is NULL while updating user ({user}) quest and profile")
    #
    #     # 4. Пересчитываем профиль
    #     if embeddings:
    #         # Усредняем эмбеддинги (mean pooling) - Преобразуем список тензоров в один тензор
    #         user_embeddings_tensor = torch.stack(embeddings)
    #         new_user_profile = torch.mean(user_embeddings_tensor, dim=0)
    #         profile_blob = pickle.dumps(new_user_profile)
    #     else:
    #         profile_blob = None
    #
    #     # 5. Сохраняем обновленного пользователя
    #     cursor.execute("""
    #         UPDATE users
    #         SET quest_ids_json = ?, profile_blob = ?, updated_at = CURRENT_TIMESTAMP
    #         WHERE user_id = ?
    #     """, (json.dumps(new_quests), profile_blob, user.user_id))
    #
    #     self.conn.commit()
    #     return True

    def get_stats(self) -> Dict:
        """Статистика базы"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM quests")
        quest_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM quests WHERE embedding_blob IS NOT NULL")
        embeddings_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE profile_blob IS NOT NULL")
        profiles_count = cursor.fetchone()[0]

        import os
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        return {
            'quests': quest_count,
            'users': user_count,
            'quests_with_embeddings': embeddings_count,
            'users_with_profiles': profiles_count,
            'db_size_mb': round(db_size / (1024 * 1024), 2)
        }

    def close(self):
        """Закрывает соединение с БД"""
        self.conn.close()
        logger.info("Database connection closed")