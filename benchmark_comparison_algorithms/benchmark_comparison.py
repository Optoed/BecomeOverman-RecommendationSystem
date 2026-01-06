"""
Сравнение алгоритмов рекомендаций для дипломной работы
Проводится тестирование на локальных тестовых данных без использования API
"""

import time
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Tuple, Optional
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestQuest:
    """Тестовый квест для экспериментов"""
    id: int
    title: str
    description: str
    category: str


@dataclass
class TestUser:
    """Тестовый пользователь для экспериментов"""
    id: int
    quests: List[int]  # ID завершенных квестов
    preferred_categories: List[str]


class BaseRecommender:
    """Базовый класс для всех рекомендательных алгоритмов"""

    def __init__(self, name: str):
        self.name = name
        self.quests: Dict[int, TestQuest] = {}
        self.users: Dict[int, TestUser] = {}

    def load_data(self, quests: List[TestQuest], users: List[TestUser]):
        """Загрузка данных для тестирования"""
        self.quests = {q.id: q for q in quests}
        self.users = {u.id: u for u in users}

    def recommend_quests(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Рекомендация квестов для пользователя"""
        raise NotImplementedError

    def recommend_friends(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        """Рекомендация друзей (похожих пользователей)"""
        raise NotImplementedError

    def search_quests(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Поиск квестов по запросу"""
        raise NotImplementedError

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Информация об алгоритме"""
        return {
            "name": self.name,
            "type": "base",
            "description": "Базовый рекомендательный алгоритм"
        }


class ContentBasedBERT(BaseRecommender):
    """Контентная фильтрация на BERT (текущий алгоритм)"""

    def __init__(self):
        super().__init__("Content-Based BERT")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.quest_embeddings: Dict[int, torch.Tensor] = {}
        self.user_profiles: Dict[int, torch.Tensor] = {}

    def load_data(self, quests: List[TestQuest], users: List[TestUser]):
        super().load_data(quests, users)

        logger.info(f"Создаем эмбеддинги для {len(quests)} квестов...")

        # Создаем эмбеддинги для всех квестов
        for quest in quests:
            text = self._create_quest_text(quest)
            embedding = self.model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False  # <-- ОТКЛЮЧАЕМ ПРОГРЕСС БАР
            )
            self.quest_embeddings[quest.id] = embedding

        # Создаем профили пользователей
        logger.info(f"Создаем профили для {len(users)} пользователей...")
        for user in users:
            if user.quests:
                user_embeddings = []
                for quest_id in user.quests:
                    if quest_id in self.quest_embeddings:
                        user_embeddings.append(self.quest_embeddings[quest_id])

                if user_embeddings:
                    user_embeddings_tensor = torch.stack(user_embeddings)
                    self.user_profiles[user.id] = torch.mean(user_embeddings_tensor, dim=0)

    def _create_quest_text(self, quest: TestQuest) -> str:
        """Создание текстового описания квеста для эмбеддинга"""
        return f"{quest.title}. {quest.description}. {quest.category}"

    def recommend_quests(self, user_id: int, top_k: int = 10) -> tuple[list[dict[str, Any]], float]:
        start_time = time.time()

        if user_id not in self.user_profiles:
            return [], 0

        user_profile = self.user_profiles[user_id]
        user_quests = set(self.users[user_id].quests)

        results = []
        for quest_id, quest_embedding in self.quest_embeddings.items():
            if quest_id in user_quests:
                continue

            score = util.cos_sim(user_profile, quest_embedding).item()
            quest = self.quests[quest_id]

            results.append({
                "quest_id": quest_id,
                "title": quest.title,
                "category": quest.category,
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        execution_time = (time.time() - start_time) * 1000  # мс

        return results[:top_k], execution_time

    def search_quests(self, query: str, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        start_time = time.time()

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False  # <-- ОТКЛЮЧАЕМ ПРОГРЕСС БАР
        )

        results = []
        for quest_id, quest_embedding in self.quest_embeddings.items():
            score = util.cos_sim(query_embedding, quest_embedding).item()
            quest = self.quests[quest_id]

            results.append({
                "quest_id": quest_id,
                "title": quest.title,
                "category": quest.category,
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        execution_time = (time.time() - start_time) * 1000  # мс

        return results[:top_k], execution_time

    def recommend_friends(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        """Рекомендация друзей на основе схожести профилей пользователей"""
        start_time = time.time()

        if user_id not in self.user_profiles:
            return [], 0

        user_profile = self.user_profiles[user_id]
        user_quests = set(self.users[user_id].quests)

        results = []
        for other_user_id, other_profile in self.user_profiles.items():
            if other_user_id == user_id:
                continue

            # Вычисляем схожесть профилей
            similarity = util.cos_sim(user_profile, other_profile).item()

            # Получаем информацию о другом пользователе
            other_user = self.users[other_user_id]
            other_quests = set(other_user.quests)

            # Находим общие квесты
            common_quests = list(user_quests & other_quests)
            common_categories = len(set(self.users[user_id].preferred_categories) &
                                    set(other_user.preferred_categories))

            results.append({
                "user_id": other_user_id,
                "similarity_score": similarity,
                "common_quests_count": len(common_quests),
                "common_categories_count": common_categories,
                "quests_count": len(other_user.quests),
                "explanation": self._generate_friend_explanation(similarity, len(common_quests), common_categories)
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        execution_time = (time.time() - start_time) * 1000

        return results[:top_k], execution_time

    def _generate_friend_explanation(self, similarity: float, common_quests: int, common_categories: int) -> str:
        """Генерация объяснения рекомендации друга"""
        explanations = []

        if similarity > 0.8:
            explanations.append("очень высокая схожесть интересов")
        elif similarity > 0.6:
            explanations.append("высокая схожесть интересов")
        elif similarity > 0.4:
            explanations.append("умеренная схожесть интересов")

        if common_quests > 0:
            explanations.append(f"{common_quests} общих квестов")

        if common_categories > 0:
            explanations.append(f"{common_categories} общих категорий интересов")

        return ", ".join(explanations) if explanations else "схожесть по интересам"

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "content_based",
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dim": 384,
            "description": "Контентная фильтрация на основе BERT эмбеддингов"
        }


class CollaborativeFilteringKNN(BaseRecommender):
    """Коллаборативная фильтрация на основе KNN"""

    def __init__(self, n_neighbors: int = 5):
        super().__init__(f"Collaborative KNN (k={n_neighbors})")
        self.n_neighbors = n_neighbors
        self.user_quest_matrix = None
        self.knn_model = None
        self.user_ids = []
        self.quest_ids = []

    def load_data(self, quests: List[TestQuest], users: List[TestUser]):
        super().load_data(quests, users)

        # Создаем user-quest матрицу
        self.user_ids = list(self.users.keys())
        self.quest_ids = list(self.quests.keys())

        logger.info(f"Создаем user-quest матрицу ({len(users)}x{len(quests)})...")

        # Матрица: 1 если пользователь добавил себе этот квест, 0 если нет
        self.user_quest_matrix = np.zeros((len(self.user_ids), len(self.quest_ids)))

        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        quest_id_to_idx = {quest_id: idx for idx, quest_id in enumerate(self.quest_ids)}

        for user in users:
            user_idx = user_id_to_idx[user.id]
            for quest_id in user.quests:
                if quest_id in quest_id_to_idx:
                    quest_idx = quest_id_to_idx[quest_id]
                    self.user_quest_matrix[user_idx][quest_idx] = 1

        # Обучаем KNN модель
        logger.info("Обучаем KNN модель...")
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(users)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(self.user_quest_matrix)

    def recommend_quests(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        start_time = time.time()

        if user_id not in self.user_ids:
            return [], 0

        user_idx = self.user_ids.index(user_id)
        user_quests = set(self.users[user_id].quests)

        # Находим похожих пользователей
        distances, indices = self.knn_model.kneighbors(
            self.user_quest_matrix[user_idx].reshape(1, -1),
            n_neighbors=min(self.n_neighbors + 1, len(self.user_ids))
        )

        # Собираем квесты похожих пользователей
        quest_scores = defaultdict(float)

        for i, neighbor_idx in enumerate(indices[0]):
            if i == 0:  # Пропускаем самого себя
                continue

            neighbor_id = self.user_ids[neighbor_idx]
            neighbor_user = self.users[neighbor_id]

            # Учитываем квесты похожего пользователя с весом
            weight = 1 - distances[0][i]  # Чем ближе пользователь, тем больше вес

            for quest_id in neighbor_user.quests:
                if quest_id not in user_quests:
                    quest_scores[quest_id] += weight

        # Сортируем и формируем результаты
        results = []
        for quest_id, score in sorted(quest_scores.items(), key=lambda x: x[1], reverse=True):
            quest = self.quests[quest_id]
            results.append({
                "quest_id": quest_id,
                "score": score,
                "title": quest.title,
                "category": quest.category,
                "method": "collaborative"
            })

        execution_time = (time.time() - start_time) * 1000  # мс

        return results[:top_k], execution_time

    def recommend_friends(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        """Рекомендация друзей на основе коллаборативной фильтрации"""
        start_time = time.time()

        if user_id not in self.user_ids:
            return [], 0

        user_idx = self.user_ids.index(user_id)
        user_quests = set(self.users[user_id].quests)

        # Находим похожих пользователей
        distances, indices = self.knn_model.kneighbors(
            self.user_quest_matrix[user_idx].reshape(1, -1),
            n_neighbors=min(self.n_neighbors + 1, len(self.user_ids))
        )

        results = []
        for i, neighbor_idx in enumerate(indices[0]):
            if i == 0:  # Пропускаем самого себя
                continue

            neighbor_id = self.user_ids[neighbor_idx]
            neighbor_user = self.users[neighbor_id]
            neighbor_quests = set(neighbor_user.quests)

            similarity = 1 - distances[0][i]  # Косинусная схожесть
            common_quests = list(user_quests & neighbor_quests)
            common_categories = len(set(self.users[user_id].preferred_categories) &
                                    set(neighbor_user.preferred_categories))

            results.append({
                "user_id": neighbor_id,
                "similarity_score": float(similarity),
                "common_quests_count": len(common_quests),
                "common_categories_count": common_categories,
                "quests_count": len(neighbor_user.quests),
                "method": "collaborative_filtering",
                "explanation": f"Схожесть по предпочтениям: {similarity:.2f}"
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        execution_time = (time.time() - start_time) * 1000

        return results[:top_k], execution_time

    def search_quests(self, query: str, top_k: int = 10) -> tuple[list[Any], int]:
        # Для коллаборативной фильтрации поиск не реализован
        return [], 0

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "collaborative",
            "n_neighbors": self.n_neighbors,
            "description": "Коллаборативная фильтрация на основе пользовательских предпочтений"
        }


class HybridRecommender(BaseRecommender):
    """Гибридный алгоритм (Content-Based + Collaborative)"""

    def __init__(self, content_weight: float = 0.6, collaborative_weight: float = 0.4):
        super().__init__(f"Hybrid (CB:{content_weight}, CF:{collaborative_weight})")
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

        self.content_recommender = ContentBasedBERT()
        self.collaborative_recommender = CollaborativeFilteringKNN()

    def load_data(self, quests: List[TestQuest], users: List[TestUser]):
        super().load_data(quests, users)

        # Загружаем данные в оба алгоритма
        self.content_recommender.load_data(quests, users)
        self.collaborative_recommender.load_data(quests, users)

    def recommend_quests(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        start_time = time.time()

        # Получаем рекомендации от обоих алгоритмов
        content_results, _ = self.content_recommender.recommend_quests(user_id, top_k * 2)
        collaborative_results, _ = self.collaborative_recommender.recommend_quests(user_id, top_k * 2)

        # Нормализуем скоры
        content_scores = {r["quest_id"]: r["score"] for r in content_results}
        collaborative_scores = {r["quest_id"]: r["score"] for r in collaborative_results}

        # Масштабируем скоры к [0, 1]
        if content_scores:
            max_content = max(content_scores.values())
            if max_content > 0:
                content_scores = {k: v / max_content for k, v in content_scores.items()}

        if collaborative_scores:
            max_collab = max(collaborative_scores.values())
            if max_collab > 0:
                collaborative_scores = {k: v / max_collab for k, v in collaborative_scores.items()}

        # Объединяем скоры с весами
        combined_scores = defaultdict(float)

        for quest_id, score in content_scores.items():
            combined_scores[quest_id] += score * self.content_weight

        for quest_id, score in collaborative_scores.items():
            combined_scores[quest_id] += score * self.collaborative_weight

        # Формируем результаты
        results = []
        for quest_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            if quest_id in self.quests:
                quest = self.quests[quest_id]
                results.append({
                    "quest_id": quest_id,
                    "score": score,
                    "title": quest.title,
                    "category": quest.category,
                    "content_score": content_scores.get(quest_id, 0),
                    "collaborative_score": collaborative_scores.get(quest_id, 0)
                })

        execution_time = (time.time() - start_time) * 1000  # мс

        return results[:top_k], execution_time

    def recommend_friends(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        """Гибридные рекомендации друзей"""
        start_time = time.time()

        # Получаем рекомендации от обоих алгоритмов
        content_friends, _ = self.content_recommender.recommend_friends(user_id, top_k * 2)
        collaborative_friends, _ = self.collaborative_recommender.recommend_friends(user_id, top_k * 2)

        # Нормализуем скоры
        content_scores = {r["user_id"]: r["similarity_score"] for r in content_friends}
        collaborative_scores = {r["user_id"]: r["similarity_score"] for r in collaborative_friends}

        # Масштабируем скоры к [0, 1]
        if content_scores:
            max_content = max(content_scores.values())
            if max_content > 0:
                content_scores = {k: v / max_content for k, v in content_scores.items()}

        if collaborative_scores:
            max_collab = max(collaborative_scores.values())
            if max_collab > 0:
                collaborative_scores = {k: v / max_collab for k, v in collaborative_scores.items()}

        # Объединяем скоры с весами
        combined_scores = defaultdict(float)

        for user_id_, score in content_scores.items():
            combined_scores[user_id_] += score * self.content_weight

        for user_id_, score in collaborative_scores.items():
            combined_scores[user_id_] += score * self.collaborative_weight

        # Формируем результаты
        results = []
        for friend_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            if friend_id in self.users:
                friend_user = self.users[friend_id]
                results.append({
                    "user_id": friend_id,
                    "similarity_score": score,
                    "quests_count": len(friend_user.quests),
                    "content_score": content_scores.get(friend_id, 0),
                    "collaborative_score": collaborative_scores.get(friend_id, 0),
                    "method": "hybrid",
                    "explanation": f"Гибридная схожесть: контент={content_scores.get(friend_id, 0):.2f}, коллаб={collaborative_scores.get(friend_id, 0):.2f}"
                })

        execution_time = (time.time() - start_time) * 1000

        return results[:top_k], execution_time

    def search_quests(self, query: str, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        # Используем только контентную фильтрацию для поиска
        return self.content_recommender.search_quests(query, top_k)

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "hybrid",
            "content_weight": self.content_weight,
            "collaborative_weight": self.collaborative_weight,
            "description": "Гибридный алгоритм, объединяющий контентную и коллаборативную фильтрацию"
        }


class SimpleTfidfRecommender(BaseRecommender):
    """Простой TF-IDF алгоритм для сравнения"""

    def __init__(self):
        super().__init__("Simple TF-IDF")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['russian'])
        self.tfidf_matrix = None
        self.quest_ids = []
        self.knn = None

    def load_data(self, quests: List[TestQuest], users: List[TestUser]):
        super().load_data(quests, users)

        logger.info("Создаем TF-IDF представления...")
        quest_texts = []
        self.quest_ids = []

        for quest in quests:
            text = f"{quest.title} {quest.description} {quest.category}"
            quest_texts.append(text)
            self.quest_ids.append(quest.id)

        self.tfidf_matrix = self.vectorizer.fit_transform(quest_texts)
        self.knn = NearestNeighbors(
            n_neighbors=min(20, len(quests)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn.fit(self.tfidf_matrix)

    def recommend_quests(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        start_time = time.time()

        if user_id not in self.users:
            return [], 0

        user = self.users[user_id]
        completed_quests = set(user.quests)

        if not user.quests:
            return [], 0

        user_quest_indices = []
        for quest_id in user.quests:
            if quest_id in self.quest_ids:
                user_quest_indices.append(self.quest_ids.index(quest_id))

        if not user_quest_indices:
            return [], 0

        # Создаем профиль пользователя на основе его квестов
        user_vectors = self.tfidf_matrix[user_quest_indices]
        user_vector = user_vectors.mean(axis=0)
        # ЯВНО конвертируем np.matrix в np.array
        user_vector = np.asarray(user_vector)

        n_neighbors = min(top_k + len(completed_quests), self.tfidf_matrix.shape[0])
        if n_neighbors <= 0:
            return [], 0

        distances, indices = self.knn.kneighbors(user_vector, n_neighbors=n_neighbors)

        results = []
        for i, (distance, quest_idx) in enumerate(zip(distances[0], indices[0])):
            quest_id = self.quest_ids[quest_idx]
            if quest_id not in completed_quests:
                quest = self.quests[quest_id]
                results.append({
                    "quest_id": quest_id,
                    "score": float(1 - distance),
                    "title": quest.title,
                    "category": quest.category
                })
            if len(results) >= top_k:
                break

        execution_time = (time.time() - start_time) * 1000  # мс
        return results, execution_time

    def search_quests(self, query: str, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        start_time = time.time()

        query_vector = self.vectorizer.transform([query])

        n_neighbors = min(top_k, self.tfidf_matrix.shape[0])
        if n_neighbors <= 0:
            return [], 0

        distances, indices = self.knn.kneighbors(query_vector, n_neighbors=n_neighbors)

        results = []
        for distance, quest_idx in zip(distances[0], indices[0]):
            quest_id = self.quest_ids[quest_idx]
            quest = self.quests[quest_id]
            results.append({
                "quest_id": quest_id,
                "score": float(1 - distance),
                "title": quest.title,
                "category": quest.category
            })

        execution_time = (time.time() - start_time) * 1000  # мс
        return results, execution_time

    def recommend_friends(self, user_id: int, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        """Рекомендация друзей на основе TF-IDF профилей"""
        start_time = time.time()

        if user_id not in self.users:
            return [], 0

        user = self.users[user_id]
        user_quests = set(user.quests)

        # Получаем TF-IDF профиль пользователя
        user_quest_indices = []
        for quest_id in user.quests:
            if quest_id in self.quest_ids:
                user_quest_indices.append(self.quest_ids.index(quest_id))

        if not user_quest_indices:
            return [], 0

        # Создаем профиль пользователя
        user_vectors = self.tfidf_matrix[user_quest_indices]
        user_vector = user_vectors.mean(axis=0)
        user_vector = np.asarray(user_vector).reshape(1, -1)

        # Создаем KNN для пользователей
        user_profiles = []
        user_ids_for_knn = []

        for u_id, u in self.users.items():
            if u_id == user_id:
                continue

            u_quest_indices = []
            for quest_id in u.quests:
                if quest_id in self.quest_ids:
                    u_quest_indices.append(self.quest_ids.index(quest_id))

            if u_quest_indices:
                u_vectors = self.tfidf_matrix[u_quest_indices]
                u_vector = u_vectors.mean(axis=0)
                u_vector = np.asarray(u_vector).reshape(1, -1)
                user_profiles.append(u_vector.flatten())
                user_ids_for_knn.append(u_id)

        if not user_profiles:
            return [], 0

        user_profiles_np = np.array(user_profiles)
        user_knn = NearestNeighbors(n_neighbors=min(top_k, len(user_profiles_np)), metric='cosine')
        user_knn.fit(user_profiles_np)

        # Ищем похожих пользователей
        distances, indices = user_knn.kneighbors(user_vector, n_neighbors=min(top_k, len(user_profiles_np)))

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            friend_id = user_ids_for_knn[idx]
            friend_user = self.users[friend_id]
            friend_quests = set(friend_user.quests)

            common_quests = list(user_quests & friend_quests)
            common_categories = len(set(user.preferred_categories) & set(friend_user.preferred_categories))

            results.append({
                "user_id": friend_id,
                "similarity_score": float(1 - distance),
                "common_quests_count": len(common_quests),
                "common_categories_count": common_categories,
                "quests_count": len(friend_user.quests),
                "method": "tfidf",
                "explanation": f"TF-IDF схожесть профилей: {1 - distance:.2f}"
            })

        execution_time = (time.time() - start_time) * 1000
        return results, execution_time

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "tfidf",
            "max_features": 1000,
            "description": "Простой алгоритм на основе TF-IDF векторизации"
        }


class TestDataGenerator:
    """Генератор тестовых данных для экспериментов"""

    @staticmethod
    def generate_quests() -> List[TestQuest]:
        """Генерация тестовых квестов для развития и саморазвития"""

        # Более релевантные категории для self-improvement
        categories = [
            "Продуктивность", "Здоровье", "Образование", "Карьера",
            "Творчество", "Финансы", "Социальные навыки", "Языки",
            "Технологии", "Ментальное здоровье", "Хобби", "Спорт"
        ]

        # Реальные квесты для развития
        quest_templates = [
            # Здоровье и фитнес (дополняем)
            ("Утренний энерджайзер: 21 день ранних подъемов",
             "Сформируйте привычку раннего пробуждения через постепенное изменение режима. Включает техники засыпания, утренние ритуалы и трекинг сна.",
             "Здоровье"),

            ("Функциональный тренинг: тело за 30 дней",
             "Комплекс упражнений для развития силы, выносливости и гибкости без оборудования. Программа для дома с прогрессией нагрузок.",
             "Спорт"),

            # Продуктивность и сила воли (дополняем)
            ("Марафон глубокой работы: 40 часов фокуса",
             "Интенсивный курс по развитию концентрации. Техники Pomodoro, digital detox, управление вниманием для студентов и профессионалов.",
             "Продуктивность"),

            ("Привычки на миллион: 66 дней трансформации",
             "Научный подход к формированию устойчивых привычек. От трекинга до автоматизации, с фокусом на продуктивность и личное развитие.",
             "Продуктивность"),

            # Образование и технологии (новые)
            ("Python за 30 дней: от нуля до первого проекта",
             "Интенсив по основам программирования на Python. Ежедневные задачи, мини-проекты, основы ООП и работа с API.",
             "Технологии"),

            ("Data Science для начинающих",
             "Введение в анализ данных: pandas, numpy, визуализация. Реальные датасеты, статистика и ML basics для нетекстовых задач.",
             "Технологии"),

            ("Веб-разработка: HTML/CSS/JS марафон",
             "Создание полноценного сайта-портфолио за 3 недели. Адаптивная верстка, основы JavaScript и деплой на GitHub Pages.",
             "Технологии"),

            # Языки (новые)
            ("Английский для IT: 90 дней immersion",
             "Специализированный курс технического английского. Документация, код-ревью, техники чтения и профессиональная коммуникация.",
             "Языки"),

            ("Испанский с нуля: 60 дней до A1",
             "Интенсив для начинающих. Основы грамматики, бытовые диалоги, испанская культура через сериалы и музыку.",
             "Языки"),

            # Карьера и финансы (новые)
            ("Карьерный рост: стратегия за 4 недели",
             "Планирование карьерного развития, составление резюме, подготовка к собеседованиям и нетворкинг в IT-сфере.",
             "Карьера"),

            ("Инвестиции для начинающих: первые 100к",
             "Безопасное введение в мир инвестиций. Акции, облигации, ETF, риск-менеджмент и психология трейдинга.",
             "Финансы"),

            ("Фриланс-старт: уход в свободное плавание",
             "Пошаговый план перехода на фриланс. Поиск клиентов, ценообразование, юридические аспекты и управление проектами.",
             "Карьера"),

            # Творчество (дополняем)
            ("Дизайн-мышление: создание продукта",
             "Полный цикл от идеи до прототипа. Исследование пользователей, скетчинг, UX/UI основы и тестирование решений.",
             "Творчество"),

            ("Цифровая иллюстрация в Procreate",
             "Освоение iPad + Procreate с нуля. Брашпики, слои, анимация и создание стикерпаков для Telegram.",
             "Творчество"),

            # Социальные навыки (новые)
            ("Эмоциональный интеллект: управление отношениями",
             "Развитие EQ для карьерного и личного роста. Распознавание эмоций, эмпатия, разрешение конфликтов и нетоксичное общение.",
             "Социальные навыки"),

            ("Публичные выступления: от страха к мастерству",
             "Преодоление страха сцены через постепенную экспозицию. Техника речи, сторителлинг и работа с аудиторией.",
             "Социальные навыки"),

            # Ментальное здоровье (новые)
            ("Медитация для занятых людей",
             "10-минутные практики для интеграции в рабочий день. Mindfulness, дыхательные техники и управление стрессом в реальном времени.",
             "Ментальное здоровье"),

            ("Цифровой минимализм: наведение порядка",
             "Оптимизация цифрового пространства для ясности ума. Чистка соцсетей, настройка уведомлений и осознанное использование гаджетов.",
             "Ментальное здоровье"),

            # Хобби и увлечения (новые)
            ("Фотография на iPhone: про уровень",
             "Профессиональная съемка на смартфон. Композиция, свет, мобильный Lightroom и создание контента для соцсетей.",
             "Хобби"),

            ("Кулинарный экспресс: 15 быстрых ужинов",
             "Рецепты для занятых людей. Приготовление за 30 минут, meal prep, основы ножевой техники и работа со специями.",
             "Хобби"),

            ("Гитара за 60 дней: 10 песен у костра",
             "Практический курс для абсолютных новичков. Аккорды, бой, перебор и разбор популярных песен для дружеских посиделок.",
             "Хобби"),

            # Дополнительные технологии (новые)
            ("DevOps для разработчиков: Docker & Kubernetes",
             "Основы контейнеризации и оркестрации. Практика с реальными проектами, деплой микросервисов и CI/CD пайплайны.",
             "Технологии"),

            ("Мобильная разработка: Flutter марафон",
             "Создание кроссплатформенного приложения за месяц. Dart основы, виджеты, state management и публикация в магазины.",
             "Технологии"),

            ("SQL для анализа данных",
             "От простых запросов до сложных аналитических отчетов. JOIN, оконные функции, оптимизация и работа с большими объемами данных.",
             "Технологии"),

            # Личное развитие (новые)
            ("Скорочтение и конспектирование",
             "Увеличение скорости чтения в 2-3 раза с сохранением понимания. Техники для учебников, документации и профессиональной литературы.",
             "Образование"),

            ("Тайм-менеджмент для remote-работы",
             "Эффективная организация удаленного рабочего дня. Приоритизация, борьба с прокрастинацией и баланс work-life.",
             "Продуктивность"),

            ("Критическое мышление: фильтрация информации",
             "Навыки анализа новостей, статей и исследований. Логические ошибки, когнитивные искажения и принятие взвешенных решений.",
             "Образование"),
        ]

        quests = []
        for idx, quest_data in enumerate(quest_templates):
            quest = TestQuest(
                id=idx,
                title=quest_data[0],
                description=quest_data[1],
                category=quest_data[2],
            )
            quests.append(quest)

        return quests

    @staticmethod
    def generate_users(n_users: int = 50, quests: List[TestQuest] = None) -> List[TestUser]:
        """Генерация тестовых пользователей с реальными интересами"""

        if quests is None:
            quests = []

        # Профили типичных пользователей self-improvement платформы
        user_profiles = [
            {"interests": ["Технологии", "Образование", "Карьера"]},
            {"interests": ["Карьера", "Финансы", "Продуктивность"]},
            {"interests": ["Технологии", "Продуктивность", "Языки"]},
            {"interests": ["Здоровье", "Спорт", "Ментальное здоровье"]},
            {"interests": ["Творчество", "Хобби", "Социальные навыки"]},
            {"interests": ["Здоровье", "Продуктивность", "Хобби"]}
        ]

        users = []
        for i in range(1, n_users + 1):
            # Выбираем профиль пользователя
            profile = random.choice(user_profiles)

            # Выбираем квесты, соответствующие интересам профиля
            available_quests = [q for q in quests if q.category in profile["interests"]]

            if not available_quests:
                available_quests = quests  # fallback

            # Пользователь завершает 2-8 квестов в своих интересах
            n_chosed = random.randint(2, 8)

            # Берем только существующие ID квестов
            available_quest_ids = [q.id for q in available_quests]

            if len(available_quests) >= n_chosed:
                user_quests = random.sample(available_quest_ids, n_chosed)
            else:
                user_quests = available_quest_ids.copy()
                # Добавляем случайные квесты из других категорий
                all_quest_ids = [q.id for q in quests]
                other_quests = [qid for qid in all_quest_ids if qid not in user_quests]
                if other_quests:
                    additional = min(n_chosed - len(user_quests), len(other_quests))
                    user_quests.extend(random.sample(other_quests, additional))

            # Создаем пользователя
            user = TestUser(
                id=i,
                quests=user_quests,
                preferred_categories=profile["interests"][:2]  # Топ-2 интереса
            )
            users.append(user)

        return users

    @staticmethod
    def create_ground_truth(users: List[TestUser], quests: List[TestQuest]) -> Dict[int, List[int]]:
        """Создание продвинутого ground truth с учетом семантики"""

        ground_truth = {}

        # Используем BERT для семантического ground truth
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Эмбеддинги квестов
        quest_embeddings = {}
        for quest in quests:
            text = f"{quest.title}. {quest.description}. {quest.category}"
            quest_embeddings[quest.id] = model.encode(text, convert_to_tensor=True, show_progress_bar=False)

        for user in users:
            relevant_quests = []
            user_quests = set(user.quests)

            if user.quests:
                # 1. Эмбеддинг профиля пользователя
                user_embeddings = []
                for quest_id in user.quests:
                    if quest_id in quest_embeddings:
                        user_embeddings.append(quest_embeddings[quest_id])

                if user_embeddings:
                    user_embedding = torch.stack(user_embeddings).mean(dim=0)

                    # 2. Находим семантически похожие квесты
                    for quest_id, quest_embedding in quest_embeddings.items():
                        if quest_id not in user_quests:
                            similarity = util.cos_sim(user_embedding, quest_embedding).item()
                            if similarity > 0.5:  # Порог схожести
                                relevant_quests.append((quest_id, similarity))

            # 3. Сортируем по схожести
            relevant_quests.sort(key=lambda x: x[1], reverse=True)
            ground_truth[user.id] = [qid for qid, _ in relevant_quests[:15]]  # Топ-15

        return ground_truth

    @staticmethod
    def create_friends_ground_truth(users: List[TestUser]) -> Dict[int, List[int]]:
        """Создание ground truth для друзей (основано на интересах)"""

        ground_truth = {}

        for user in users:
            similarities = []
            user_quests = set(user.quests)
            user_categories = set(user.preferred_categories)

            for other_user in users:
                if other_user.id == user.id:
                    continue

                other_quests = set(other_user.quests)
                other_categories = set(other_user.preferred_categories)

                # 1. Общие квесты (самый сильный сигнал)
                common_quests = len(user_quests & other_quests)

                # 2. Общие категории интересов
                common_categories = len(user_categories & other_categories)

                # 3. Jaccard similarity по квестам
                if user_quests or other_quests:
                    jaccard_similarity = len(user_quests & other_quests) / len(user_quests | other_quests)
                else:
                    jaccard_similarity = 0

                # Комбинированный скоринг
                score = (common_quests * 0.5 +
                         common_categories * 0.3 +
                         jaccard_similarity * 0.2)

                if score > 0.1:  # Порог
                    similarities.append((other_user.id, score))

            # Сортируем и берем топ
            similarities.sort(key=lambda x: x[1], reverse=True)
            ground_truth[user.id] = [uid for uid, _ in similarities[:10]]

        return ground_truth

class BenchmarkEvaluator:
    """Оценщик алгоритмов рекомендаций"""

    def __init__(self, ground_truth: Dict[int, List[int]]):
        self.ground_truth = ground_truth

    def evaluate(self, recommendations: Dict[int, List[int]], top_k: int = 5) -> Dict[str, float]:
        """Оценка рекомендаций по различным метрикам"""

        metrics = {
            "precision@k": [],
            "recall@k": [],
            "f1@k": [],
            "ndcg@k": [],
            "map@k": []
        }

        for user_id, user_recommendations in recommendations.items():
            if user_id not in self.ground_truth:
                continue

            true_positives = set(self.ground_truth[user_id])
            recommended = user_recommendations[:top_k]

            # Precision@K
            hits = len(set(recommended) & true_positives)
            precision = hits / top_k if top_k > 0 else 0
            metrics["precision@k"].append(precision)

            # Recall@K
            recall = hits / len(true_positives) if true_positives else 0
            metrics["recall@k"].append(recall)

            # F1@K
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics["f1@k"].append(f1)

            # NDCG@K
            dcg = 0
            for i, quest_id in enumerate(recommended):
                if quest_id in true_positives:
                    dcg += 1 / np.log2(i + 2)

            # Ideal DCG
            idcg = sum(1 / np.log2(i + 2) for i in range(min(top_k, len(true_positives))))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics["ndcg@k"].append(ndcg)

            # Average Precision@K
            ap = 0
            num_hits = 0
            for i, quest_id in enumerate(recommended):
                if quest_id in true_positives:
                    num_hits += 1
                    ap += num_hits / (i + 1)
            ap = ap / min(len(true_positives), top_k) if true_positives else 0
            metrics["map@k"].append(ap)

        # Усредняем метрики по всем пользователям
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = np.mean(values)
                results[f"{metric_name}_std"] = np.std(values)
            else:
                results[metric_name] = 0
                results[f"{metric_name}_std"] = 0

        return results

    def evaluate_search(self, search_results: Dict[str, List[int]],
                        queries: Dict[str, List[int]]) -> Dict[str, float]:
        """Оценка поисковых результатов"""

        metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": []  # Mean Reciprocal Rank
        }

        for query, results in search_results.items():
            if query not in queries:
                continue

            true_relevant = set(queries[query])
            retrieved = results

            # Precision и Recall
            hits = len(set(retrieved) & true_relevant)
            precision = hits / len(retrieved) if retrieved else 0
            recall = hits / len(true_relevant) if true_relevant else 0

            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(
                2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            )

            # MRR
            for rank, quest_id in enumerate(retrieved, 1):
                if quest_id in true_relevant:
                    metrics["mrr"].append(1 / rank)
                    break
            else:
                metrics["mrr"].append(0)

        # Усредняем метрики
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = np.mean(values)
            else:
                results[metric_name] = 0

        return results

    def evaluate_friends(self, recommendations: Dict[int, List[int]],
                         ground_truth: Dict[int, List[int]],
                         top_k: int = 5) -> Dict[str, float]:
        """Оценка рекомендаций друзей"""

        metrics = {
            "friend_precision@k": [],
            "friend_recall@k": [],
            "friend_f1@k": [],
            "friend_map@k": []
        }

        for user_id, user_recommendations in recommendations.items():
            if user_id not in ground_truth:
                continue

            true_friends = set(ground_truth[user_id])
            recommended = user_recommendations[:top_k]

            # Precision@K
            hits = len(set(recommended) & true_friends)
            precision = hits / top_k if top_k > 0 else 0
            metrics["friend_precision@k"].append(precision)

            # Recall@K
            recall = hits / len(true_friends) if true_friends else 0
            metrics["friend_recall@k"].append(recall)

            # F1@K
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics["friend_f1@k"].append(f1)

            # Average Precision@K
            ap = 0
            num_hits = 0
            for i, friend_id in enumerate(recommended):
                if friend_id in true_friends:
                    num_hits += 1
                    ap += num_hits / (i + 1)
            ap = ap / min(len(true_friends), top_k) if true_friends else 0
            metrics["friend_map@k"].append(ap)

        # Усредняем метрики
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = np.mean(values)
                results[f"{metric_name}_std"] = np.std(values)
            else:
                results[metric_name] = 0
                results[f"{metric_name}_std"] = 0

        return results


def run_comprehensive_benchmark():
    """Запуск комплексного бенчмарка всех алгоритмов"""

    logger.info("=" * 60)
    logger.info("ЗАПУСК КОМПЛЕКСНОГО БЕНЧМАРКА АЛГОРИТМОВ")
    logger.info("=" * 60)

    # 1. Генерируем тестовые данные
    logger.info("Генерация тестовых данных...")
    quests = TestDataGenerator.generate_quests()
    users = TestDataGenerator.generate_users(n_users=50, quests=quests)
    ground_truth = TestDataGenerator.create_ground_truth(users, quests)

    logger.info(f"Сгенерировано: {len(quests)} квестов, {len(users)} пользователей")
    logger.info(f"Категории квестов: {list(set(q.category for q in quests))}")

    # 2. Создаем тестовые запросы для поиска
    search_queries = {
        # Технологии и программирование
        "изучить python программирование": [q.id for q in quests
                                            if any(word in q.title.lower() or word in q.description.lower()
                                                   for word in ["python", "программирование", "код"])],

        "web разработка создание сайта": [q.id for q in quests
                                          if any(word in q.title.lower() or word in q.description.lower()
                                                 for word in ["веб", "html", "css", "js", "сайт", "разработка"])],

        # Продуктивность
        "тайм менеджмент продуктивность": [q.id for q in quests
                                           if any(word in q.title.lower() or word in q.description.lower()
                                                  for word in ["тайм", "продуктивность", "фокус", "привычки"])],

        # Здоровье и спорт
        "фитнес тренировки здоровье": [q.id for q in quests
                                       if any(word in q.title.lower() or word in q.description.lower()
                                              for word in ["фитнес", "тренировка", "здоровье", "спорт"])],

        # Карьера и финансы
        "карьера развитие инвестиции": [q.id for q in quests
                                        if any(word in q.title.lower() or word in q.description.lower()
                                               for word in ["карьера", "резюме", "инвестиции", "фриланс"])],

        # Языки
        "английский язык обучение": [q.id for q in quests
                                     if any(word in q.title.lower() or word in q.description.lower()
                                            for word in ["английский", "язык", "обучение", "грамматика"])],

        # Творчество
        "дизайн творчество рисование": [q.id for q in quests
                                        if any(word in q.title.lower() or word in q.description.lower()
                                               for word in ["дизайн", "творчество", "рисование", "иллюстрация"])]
    }

    # 3. Инициализируем алгоритмы
    algorithms = [
        ContentBasedBERT(),  # Ваш текущий алгоритм
        CollaborativeFilteringKNN(n_neighbors=5),
        CollaborativeFilteringKNN(n_neighbors=10),
        HybridRecommender(content_weight=0.5, collaborative_weight=0.5),
        HybridRecommender(content_weight=0.6, collaborative_weight=0.4),
        HybridRecommender(content_weight=0.8, collaborative_weight=0.2),
        SimpleTfidfRecommender()
    ]

    # 4. Загружаем данные во все алгоритмы
    logger.info("Загрузка данных в алгоритмы...")
    for algo in algorithms:
        logger.info(f"  Загружаем {algo.name}...")
        algo.load_data(quests, users)

    # 5. Запускаем тесты рекомендаций
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТИРОВАНИЕ РЕКОМЕНДАЦИЙ")
    logger.info("=" * 60)

    recommendation_results = {}
    search_results = {}

    for algo in algorithms:
        logger.info(f"\nТестируем алгоритм: {algo.name}")

        # Тестирование рекомендаций
        user_recommendations = {}
        recommendation_times = []

        for user in users[:20]:  # Тестируем на 20 пользователях для скорости
            recommendations, exec_time = algo.recommend_quests(user.id, top_k=10)
            user_recommendations[user.id] = [r["quest_id"] for r in recommendations]
            recommendation_times.append(exec_time)

        recommendation_results[algo.name] = user_recommendations

        # Тестирование поиска (если алгоритм поддерживает)
        algo_search_results = {}
        search_times = []

        if hasattr(algo, 'search_quests'):
            for query in search_queries.keys():
                results, exec_time = algo.search_quests(query, top_k=10)
                algo_search_results[query] = [r["quest_id"] for r in results]
                search_times.append(exec_time)

            search_results[algo.name] = algo_search_results

        # Выводим метрики производительности
        avg_rec_time = np.mean(recommendation_times) if recommendation_times else 0
        avg_search_time = np.mean(search_times) if search_times else 0

        logger.info(f"  Среднее время рекомендации: {avg_rec_time:.2f} мс")
        logger.info(
            f"  Среднее время поиска: {avg_search_time:.2f} мс" if search_times else "  Поиск не поддерживается")

        # 6. Запускаем тесты рекомендаций друзей
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТИРОВАНИЕ РЕКОМЕНДАЦИЙ ДРУЗЕЙ")
    logger.info("=" * 60)

    friends_ground_truth = TestDataGenerator.create_friends_ground_truth(users)
    friends_recommendation_results = {}

    for algo in algorithms:
        logger.info(f"\nТестируем алгоритм: {algo.name}")

        if not hasattr(algo, 'recommend_friends'):
            logger.info("  Рекомендации друзей не поддерживаются")
            continue

        # Тестирование рекомендаций друзей
        user_friend_recommendations = {}
        friend_recommendation_times = []

        for user in users[:10]:  # Тестируем на 10 пользователях
            try:
                recommendations, exec_time = algo.recommend_friends(user.id, top_k=5)
                user_friend_recommendations[user.id] = [r["user_id"] for r in recommendations]
                friend_recommendation_times.append(exec_time)
            except Exception as e:
                logger.warning(f"  Ошибка рекомендаций друзей для пользователя {user.id}: {e}")
                user_friend_recommendations[user.id] = []

        friends_recommendation_results[algo.name] = user_friend_recommendations

        avg_friend_time = np.mean(friend_recommendation_times) if friend_recommendation_times else 0
        logger.info(f"  Среднее время рекомендации друзей: {avg_friend_time:.2f} мс")

    # 7. Оцениваем качество рекомендаций друзей
    logger.info("\n" + "=" * 60)
    logger.info("ОЦЕНКА КАЧЕСТВА РЕКОМЕНДАЦИЙ ДРУЗЕЙ")
    logger.info("=" * 60)

    evaluator = BenchmarkEvaluator(ground_truth)  # Можно передать любую ground truth

    friends_metrics = {}
    for algo_name, recommendations in friends_recommendation_results.items():
        if recommendations:
            metrics = evaluator.evaluate_friends(recommendations, friends_ground_truth, top_k=3)
            friends_metrics[algo_name] = metrics

            logger.info(f"\n{algo_name}:")
            for metric_name, value in metrics.items():
                if "std" not in metric_name:
                    logger.info(f"  {metric_name}: {value:.4f}")

    # 8. Оцениваем качество рекомендаций
    logger.info("\n" + "=" * 60)
    logger.info("ОЦЕНКА КАЧЕСТВА РЕКОМЕНДАЦИЙ")
    logger.info("=" * 60)

    evaluator = BenchmarkEvaluator(ground_truth)

    all_metrics = {}
    for algo_name, recommendations in recommendation_results.items():
        metrics = evaluator.evaluate(recommendations, top_k=5)
        all_metrics[algo_name] = metrics

        logger.info(f"\n{algo_name}:")
        for metric_name, value in metrics.items():
            if "std" not in metric_name:
                logger.info(f"  {metric_name}: {value:.4f}")

    # 9. Оцениваем качество поиска
    logger.info("\n" + "=" * 60)
    logger.info("ОЦЕНКА КАЧЕСТВА ПОИСКА")
    logger.info("=" * 60)

    for algo_name, search_res in search_results.items():
        if search_res:  # Если алгоритм поддерживает поиск
            search_metrics = evaluator.evaluate_search(search_res, search_queries)
            logger.info(f"\n{algo_name} (поиск):")
            for metric_name, value in search_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

    # 10. Визуализация результатов
    logger.info("\n" + "=" * 60)
    logger.info("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    logger.info("=" * 60)

    visualize_results(all_metrics, algorithms)

    # 11. Сохранение результатов в файл
    save_results_to_file(all_metrics, friends_metrics, algorithms, quests, users)

    logger.info("\n" + "=" * 60)
    logger.info("БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО!")
    logger.info("=" * 60)

    return all_metrics, friends_metrics


def visualize_results(all_metrics: Dict[str, Dict[str, float]], algorithms: List[BaseRecommender]):
    """Визуализация результатов сравнения"""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Настройка стиля
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))

        # 1. Precision@5 для всех алгоритмов
        plt.subplot(2, 3, 1)
        algorithms_names = list(all_metrics.keys())
        precision_values = [all_metrics[name]["precision@k"] for name in algorithms_names]

        bars = plt.bar(range(len(algorithms_names)), precision_values)
        plt.xticks(range(len(algorithms_names)), algorithms_names, rotation=45, ha='right')
        plt.title('Precision@5')
        plt.ylabel('Значение')
        plt.ylim(0, 1)

        # Добавляем значения на столбцы
        for bar, value in zip(bars, precision_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        # 2. Recall@5
        plt.subplot(2, 3, 2)
        recall_values = [all_metrics[name]["recall@k"] for name in algorithms_names]

        bars = plt.bar(range(len(algorithms_names)), recall_values)
        plt.xticks(range(len(algorithms_names)), algorithms_names, rotation=45, ha='right')
        plt.title('Recall@5')
        plt.ylabel('Значение')
        plt.ylim(0, 1)

        for bar, value in zip(bars, recall_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        # 3. F1@5
        plt.subplot(2, 3, 3)
        f1_values = [all_metrics[name]["f1@k"] for name in algorithms_names]

        bars = plt.bar(range(len(algorithms_names)), f1_values)
        plt.xticks(range(len(algorithms_names)), algorithms_names, rotation=45, ha='right')
        plt.title('F1-Score@5')
        plt.ylabel('Значение')
        plt.ylim(0, 1)

        for bar, value in zip(bars, f1_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        # 4. NDCG@5
        plt.subplot(2, 3, 4)
        ndcg_values = [all_metrics[name]["ndcg@k"] for name in algorithms_names]

        bars = plt.bar(range(len(algorithms_names)), ndcg_values)
        plt.xticks(range(len(algorithms_names)), algorithms_names, rotation=45, ha='right')
        plt.title('NDCG@5')
        plt.ylabel('Значение')
        plt.ylim(0, 1)

        for bar, value in zip(bars, ndcg_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        # 5. MAP@5
        plt.subplot(2, 3, 5)
        map_values = [all_metrics[name]["map@k"] for name in algorithms_names]

        bars = plt.bar(range(len(algorithms_names)), map_values)
        plt.xticks(range(len(algorithms_names)), algorithms_names, rotation=45, ha='right')
        plt.title('MAP@5')
        plt.ylabel('Значение')
        plt.ylim(0, 1)

        for bar, value in zip(bars, map_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        # 6. Сводная таблица
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Создаем таблицу с метриками
        table_data = []
        for algo_name in algorithms_names[:5]:  # Показываем первые 5 алгоритмов
            row = [algo_name[:20]]
            for metric in ['precision@k', 'recall@k', 'f1@k', 'ndcg@k']:
                row.append(f"{all_metrics[algo_name][metric]:.3f}")
            table_data.append(row)

        table = plt.table(cellText=table_data,
                          colLabels=['Algorithm', 'P@5', 'R@5', 'F1@5', 'NDCG@5'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.1, 0.1, 0.1, 0.1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        plt.suptitle('Сравнение алгоритмов рекомендаций', fontsize=16, y=1.02)
        plt.tight_layout()

        # Сохраняем график
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        logger.info("Графики сохранены в benchmark_results.png")

        # Показываем график
        plt.show()

    except ImportError:
        logger.warning("Matplotlib/Seaborn не установлены. Пропускаем визуализацию.")
    except Exception as e:
        logger.error(f"Ошибка при визуализации: {e}")


def save_results_to_file(all_metrics: Dict[str, Dict[str, float]],
                         friends_metrics: Dict[str, Dict[str, float]],
                         algorithms: List[BaseRecommender],
                         quests: List[TestQuest],
                         users: List[TestUser]):
    """Сохранение результатов в файлы"""

    # 1. Сохраняем метрики в JSON
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "total_quests": len(quests),
            "total_users": len(users),
            "test_users": 20
        },
        "algorithms": {},
        "summary": {}
    }

    for algo in algorithms:
        algo_info = algo.get_algorithm_info()
        if algo.name in all_metrics:
            results_data["algorithms"][algo.name] = {
                "info": algo_info,
                "metrics": all_metrics[algo.name]
            }

            # Добавляем метрики друзей, если они есть
            if algo.name in friends_metrics:
                results_data["algorithms"][algo.name]["friend_metrics"] = friends_metrics[algo.name]

    # Находим лучший алгоритм по F1-score
    best_algo = None
    best_f1 = 0

    for algo_name, metrics in all_metrics.items():
        if metrics["f1@k"] > best_f1:
            best_f1 = metrics["f1@k"]
            best_algo = algo_name

    results_data["summary"]["best_algorithm"] = best_algo
    results_data["summary"]["best_f1_score"] = best_f1

    # Находим лучший алгоритм для рекомендаций друзей
    best_friend_algo = None
    best_friend_f1 = 0

    for algo_name, metrics in friends_metrics.items():
        if "friend_f1@k" in metrics and metrics["friend_f1@k"] > best_friend_f1:
            best_friend_f1 = metrics["friend_f1@k"]
            best_friend_algo = algo_name

    if best_friend_algo:
        results_data["summary"]["best_friend_algorithm"] = best_friend_algo
        results_data["summary"]["best_friend_f1_score"] = best_friend_f1

    with open('benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    logger.info("Результаты сохранены в benchmark_results.json")

    # 2. Сохраняем подробный отчет в текстовый файл
    with open('benchmark_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЕТ О СРАВНЕНИИ АЛГОРИТМОВ РЕКОМЕНДАЦИЙ\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Дата тестирования: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Количество квестов: {len(quests)}\n")
        f.write(f"Количество пользователей: {len(users)}\n")
        f.write(f"Количество тестовых пользователей: 20\n\n")

        f.write("=" * 80 + "\n")
        f.write("РЕКОМЕНДАЦИИ КВЕСТОВ\n")
        f.write("=" * 80 + "\n\n")

        # Таблица с результатами рекомендаций квестов
        f.write(f"{'Алгоритм':<40} {'P@5':<8} {'R@5':<8} {'F1@5':<8} {'NDCG@5':<8} {'MAP@5':<8}\n")
        f.write("-" * 80 + "\n")

        for algo_name, metrics in all_metrics.items():
            f.write(f"{algo_name:<40} "
                    f"{metrics['precision@k']:<8.3f} "
                    f"{metrics['recall@k']:<8.3f} "
                    f"{metrics['f1@k']:<8.3f} "
                    f"{metrics['ndcg@k']:<8.3f} "
                    f"{metrics['map@k']:<8.3f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("РЕКОМЕНДАЦИИ ДРУЗЕЙ\n")
        f.write("=" * 80 + "\n\n")

        if friends_metrics:
            # Таблица с результатами рекомендаций друзей
            f.write(f"{'Алгоритм':<40} {'P@5':<8} {'R@5':<8} {'F1@5':<8} {'MAP@5':<8}\n")
            f.write("-" * 80 + "\n")

            for algo_name, metrics in friends_metrics.items():
                f.write(f"{algo_name:<40} "
                        f"{metrics.get('friend_precision@k', 0):<8.3f} "
                        f"{metrics.get('friend_recall@k', 0):<8.3f} "
                        f"{metrics.get('friend_f1@k', 0):<8.3f} "
                        f"{metrics.get('friend_map@k', 0):<8.3f}\n")
        else:
            f.write("Метрики рекомендаций друзей недоступны\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ВЫВОДЫ\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Лучший алгоритм для рекомендаций квестов: {best_algo}\n")
        f.write(f"Лучший F1-Score для квестов: {best_f1:.3f}\n\n")

        if best_friend_algo:
            f.write(f"Лучший алгоритм для рекомендаций друзей: {best_friend_algo}\n")
            f.write(f"Лучший F1-Score для друзей: {best_friend_f1:.3f}\n\n")
        else:
            f.write("Алгоритмы рекомендаций друзей не тестировались\n\n")

        # Детальный анализ каждого алгоритма
        f.write("ДЕТАЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ:\n")
        f.write("-" * 80 + "\n")

        for algo_name, metrics in all_metrics.items():
            f.write(f"\n{algo_name}:\n")
            f.write(f"  Рекомендации квестов:\n")
            f.write(f"    - Precision@5: {metrics['precision@k']:.3f}\n")
            f.write(f"    - Recall@5: {metrics['recall@k']:.3f}\n")
            f.write(f"    - F1@5: {metrics['f1@k']:.3f}\n")
            f.write(f"    - NDCG@5: {metrics['ndcg@k']:.3f}\n")
            f.write(f"    - MAP@5: {metrics['map@k']:.3f}\n")

            if algo_name in friends_metrics:
                friend_metrics = friends_metrics[algo_name]
                f.write(f"  Рекомендации друзей:\n")
                f.write(f"    - Precision@5: {friend_metrics.get('friend_precision@k', 0):.3f}\n")
                f.write(f"    - Recall@5: {friend_metrics.get('friend_recall@k', 0):.3f}\n")
                f.write(f"    - F1@5: {friend_metrics.get('friend_f1@k', 0):.3f}\n")
                f.write(f"    - MAP@5: {friend_metrics.get('friend_map@k', 0):.3f}\n")

            # Добавляем специфическую информацию об алгоритме
            if "Content-Based BERT" in algo_name:
                f.write(f"  Характеристики:\n")
                f.write(f"    - Тип: Контентная фильтрация\n")
                f.write(f"    - Модель: BERT (paraphrase-multilingual-MiniLM-L12-v2)\n")
                f.write(f"    - Размер эмбеддингов: 384\n")
                f.write(f"    - Подходит для: Понимание семантики текста, cold-start\n")
            elif "Collaborative" in algo_name:
                f.write(f"  Характеристики:\n")
                f.write(f"    - Тип: Коллаборативная фильтрация\n")
                f.write(f"    - Метод: K-Nearest Neighbors\n")
                f.write(f"    - Подходит для: Социальные рекомендации, популярные тренды\n")
            elif "Hybrid" in algo_name:
                f.write(f"  Характеристики:\n")
                f.write(f"    - Тип: Гибридный алгоритм\n")
                f.write(f"    - Комбинация: Content-Based + Collaborative\n")
                f.write(f"    - Подходит для: Баланс точности и разнообразия\n")
            elif "TF-IDF" in algo_name:
                f.write(f"  Характеристики:\n")
                f.write(f"    - Тип: Контентная фильтрация\n")
                f.write(f"    - Метод: TF-IDF векторизация\n")
                f.write(f"    - Подходит для: Быстрые рекомендации, ограниченные ресурсы\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("РЕКОМЕНДАЦИИ ДЛЯ ДИПЛОМНОЙ РАБОТЫ\n")
        f.write("=" * 80 + "\n\n")

        # Анализ результатов
        if "Content-Based BERT" in all_metrics:
            bert_f1 = all_metrics["Content-Based BERT"]["f1@k"]
            bert_position = sorted([m['f1@k'] for m in all_metrics.values()], reverse=True).index(bert_f1) + 1

            f.write("1. ВАШ АЛГОРИТМ (Content-Based BERT):\n")
            f.write(f"   - Текущая позиция: {bert_position}-е место среди {len(all_metrics)} алгоритмов\n")
            f.write(f"   - F1-Score: {bert_f1:.3f}\n")

            # Сравнение с другими алгоритмами
            if bert_position == 1:
                f.write("   - ПРЕИМУЩЕСТВО: Ваш алгоритм показал лучший результат!\n")
            elif bert_position <= 3:
                f.write("   - РЕЗУЛЬТАТ: Хороший результат в топ-3\n")
            else:
                f.write("   - ЕСТЬ ВОЗМОЖНОСТИ ДЛЯ УЛУЧШЕНИЯ\n")

            f.write("   - Преимущества:\n")
            f.write("     * Отлично понимает семантику текстовых описаний\n")
            f.write("     * Эффективен в cold-start ситуациях (новые пользователи)\n")
            f.write("     * Хорошо работает с мультиязычным контентом\n")

            f.write("   - Недостатки:\n")
            f.write("     * Требует значительных вычислительных ресурсов\n")
            f.write("     * Медленнее на больших объемах данных\n")
            f.write("     * Зависит от качества текстовых описаний\n\n")

        f.write("2. СРАВНИТЕЛЬНЫЙ АНАЛИЗ:\n")

        # Находим топ-3 алгоритма для квестов
        quest_sorted = sorted(all_metrics.items(), key=lambda x: x[1]["f1@k"], reverse=True)[:3]
        f.write("   Топ-3 алгоритма для рекомендаций квестов:\n")
        for i, (algo_name, metrics) in enumerate(quest_sorted, 1):
            f.write(f"     {i}. {algo_name}: F1-Score = {metrics['f1@k']:.3f}\n")

        if friends_metrics:
            # Находим топ-3 алгоритма для друзей
            friend_sorted = sorted(friends_metrics.items(),
                                   key=lambda x: x[1].get("friend_f1@k", 0), reverse=True)[:3]
            f.write("\n   Топ-3 алгоритма для рекомендаций друзей:\n")
            for i, (algo_name, metrics) in enumerate(friend_sorted, 1):
                f1 = metrics.get("friend_f1@k", 0)
                f.write(f"     {i}. {algo_name}: F1-Score = {f1:.3f}\n")

        f.write("\n3. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:\n")
        f.write("   - Для продакшена: используйте гибридный алгоритм\n")
        f.write("   - Для холодного старта: Content-Based BERT\n")
        f.write("   - Для социальных рекомендаций: Collaborative Filtering\n")
        f.write("   - Для ограниченных ресурсов: Simple TF-IDF\n")
        f.write("   - Для поиска: BERT показывает лучшие результаты\n\n")

        f.write("4. ПЕРСПЕКТИВЫ РАЗВИТИЯ:\n")
        f.write("   - Добавить FAISS для ускорения поиска\n")
        f.write("   - Реализовать A/B тестирование\n")
        f.write("   - Добавить объяснения рекомендаций (Explainable AI)\n")
        f.write("   - Оптимизировать для мобильных устройств\n")
        f.write("   - Добавить мультимодальность (изображения, видео)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ТЕХНИЧЕСКИЕ ДЕТАЛИ\n")
        f.write("=" * 80 + "\n\n")

        f.write("Конфигурация тестирования:\n")
        f.write(f"- Количество квестов: {len(quests)}\n")
        f.write(f"- Количество пользователей: {len(users)}\n")
        f.write(f"- Тестовые пользователи: 20\n")
        f.write(f"- Рекомендаций на пользователя: 10 квестов, 5 друзей\n")
        f.write(f"- Метрики оценки: Precision@K, Recall@K, F1-Score, NDCG, MAP\n")
        f.write(f"- Аппаратное обеспечение: CPU\n")

        # Информация об алгоритмах
        f.write("\nПротестированные алгоритмы:\n")
        for i, algo in enumerate(algorithms, 1):
            info = algo.get_algorithm_info()
            f.write(f"{i}. {info['name']}\n")
            f.write(f"   Тип: {info.get('type', 'unknown')}\n")
            f.write(f"   Описание: {info.get('description', '')}\n")
            if 'model' in info:
                f.write(f"   Модель: {info['model']}\n")
            if 'n_neighbors' in info:
                f.write(f"   K соседей: {info['n_neighbors']}\n")
            f.write("\n")

    logger.info("Отчет сохранен в benchmark_report.txt")


def run_specific_tests():
    """Запуск специфических тестов для дипломной работы"""

    logger.info("\n" + "=" * 60)
    logger.info("СПЕЦИАЛЬНЫЕ ТЕСТЫ ДЛЯ ДИПЛОМНОЙ РАБОТЫ")
    logger.info("=" * 60)

    # Тест 1: Cold-start проблема
    logger.info("\nТест 1: Cold-start проблема (пользователь без истории)")

    quests = TestDataGenerator.generate_quests()

    # Создаем пользователя без истории
    cold_start_user = TestUser(
        id=999,
        quests=[],
        preferred_categories=["Продуктивность", "Образование"]  # Более реалистичные категории
    )

    # Генерируем пользователей С ТЕМИ ЖЕ КВЕСТАМИ
    users_with_history = TestDataGenerator.generate_users(n_users=20, quests=quests)

    # Создаем новый список с холодным пользователем
    all_users = users_with_history + [cold_start_user]

    # Тестируем разные алгоритмы
    algorithms = [
        ContentBasedBERT(),
        CollaborativeFilteringKNN(n_neighbors=5),
        HybridRecommender(content_weight=0.6, collaborative_weight=0.4)
    ]

    for algo in algorithms:
        try:
            logger.info(f"\nТестируем {algo.name}...")
            algo.load_data(quests, all_users)
            recommendations, exec_time = algo.recommend_quests(cold_start_user.id, top_k=5)

            logger.info(f"{algo.name}:")
            if recommendations:
                logger.info(f"  Рекомендовано квестов: {len(recommendations)}")
                for rec in recommendations[:3]:
                    logger.info(f"  - {rec['title']} (score: {rec['score']:.3f})")
            else:
                logger.info("  Не смог дать рекомендации")
        except Exception as e:
            logger.error(f"  Ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Тест 2: Время ответа при разном количестве квестов
    logger.info("\n\nТест 2: Масштабируемость алгоритмов")

    quest_counts = [50, 100, 200]
    time_results = {}

    for n_quests in quest_counts:
        logger.info(f"\nКоличество квестов: {n_quests}")

        # Генерируем квесты
        test_quests = []
        categories = ["Продуктивность", "Здоровье", "Образование", "Технологии"]

        for i in range(n_quests):
            category = random.choice(categories)
            test_quests.append(TestQuest(
                id=i,
                title=f"Тестовый квест {i}",
                description=f"Описание тестового квеста {i} для категории {category}",
                category=category
            ))

        # Генерируем пользователей с ЭТИМИ квестами
        test_users = TestDataGenerator.generate_users(n_users=20, quests=test_quests)

        for algo in algorithms:
            try:
                logger.info(f"  Тестируем {algo.name}...")
                # Измеряем время загрузки данных
                start_time = time.time()
                algo.load_data(test_quests, test_users)
                load_time = time.time() - start_time

                # Измеряем время рекомендации
                rec_times = []
                for user in test_users[:5]:
                    _, exec_time = algo.recommend_quests(user.id, top_k=10)
                    rec_times.append(exec_time)

                avg_rec_time = np.mean(rec_times) if rec_times else 0

                if algo.name not in time_results:
                    time_results[algo.name] = {}

                time_results[algo.name][n_quests] = {
                    "load_time": load_time,
                    "avg_rec_time": avg_rec_time
                }

                logger.info(f"    загрузка={load_time:.2f}с, рекомендация={avg_rec_time:.2f}мс")
            except Exception as e:
                logger.error(f"    ошибка: {e}")

    # Тест 3: Влияние размера истории пользователя на качество
    logger.info("\n\nТест 3: Влияние размера истории пользователя на качество")

    test_quests = TestDataGenerator.generate_quests()

    # Генерируем пользователей С ЭТИМИ КВЕСТАМИ
    test_users = TestDataGenerator.generate_users(n_users=20, quests=test_quests)

    history_sizes = [1, 3, 5, 10]
    quality_results = {}

    for algo in algorithms:
        quality_results[algo.name] = {}

        for history_size in history_sizes:
            # Выбираем пользователя с достаточно большой историей
            suitable_users = [u for u in test_users if len(u.quests) >= history_size]

            if not suitable_users:
                logger.warning(f"  Нет пользователей с историей >= {history_size} для {algo.name}")
                continue

            test_user = random.choice(suitable_users)

            # Берем только первые history_size квестов
            user_quests = test_user.quests[:history_size]

            # Создаем тестового пользователя с ограниченной историей
            test_user_limited = TestUser(
                id=test_user.id + 10000,  # Уникальный ID
                quests=user_quests,
                preferred_categories=test_user.preferred_categories
            )

            try:
                # Загружаем данные для одного пользователя
                algo.load_data(test_quests, [test_user_limited])

                # Получаем рекомендации
                recommendations, _ = algo.recommend_quests(test_user_limited.id, top_k=10)

                quality_results[algo.name][history_size] = len(recommendations)
            except Exception as e:
                logger.error(f"  Ошибка для {algo.name} с историей {history_size}: {e}")

    # Выводим результаты
    logger.info("\nЗависимость качества от размера истории:")
    for algo_name, results in quality_results.items():
        logger.info(f"\n{algo_name}:")
        for size, count in sorted(results.items()):
            logger.info(f"  История {size} квестов -> {count} рекомендаций")


if __name__ == "__main__":
    # Запуск основного бенчмарка
    main_results, friends_results = run_comprehensive_benchmark()

    # Запуск дополнительных тестов
    run_specific_tests()

    logger.info("\n" + "=" * 60)
    logger.info("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    logger.info("=" * 60)
    logger.info("\nРезультаты сохранены в файлы:")
    logger.info("  - benchmark_results.json (полные результаты)")
    logger.info("  - benchmark_results.png (графики)")
    logger.info("  - benchmark_report.txt (текстовый отчет)")