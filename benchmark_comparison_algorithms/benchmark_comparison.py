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


class RealisticTestDataGenerator:
    """Генератор реалистичных тестовых данных с детерминированным ground truth"""

    @staticmethod
    def generate_quests() -> List[TestQuest]:
        """Генерация тестовых квестов для развития и саморазвития"""

        # Реалистичные квесты для развития
        quests = [
            # Технологии и программирование (ID: 0-4)
            TestQuest(0, "Python за 30 дней: от нуля до первого проекта",
                      "Интенсив по основам программирования на Python. Ежедневные задачи, мини-проекты, основы ООП и работа с API.",
                      "Технологии"),
            TestQuest(1, "Веб-разработка: HTML/CSS/JS марафон",
                      "Создание полноценного сайта-портфолио за 3 недели. Адаптивная верстка, основы JavaScript и деплой на GitHub Pages.",
                      "Технологии"),
            TestQuest(2, "Data Science для начинающих",
                      "Введение в анализ данных: pandas, numpy, визуализация. Реальные датасеты, статистика и ML basics для нетекстовых задач.",
                      "Технологии"),
            TestQuest(3, "SQL для анализа данных",
                      "От простых запросов до сложных аналитических отчетов. JOIN, оконные функции, оптимизация и работа с большими объемами данных.",
                      "Технологии"),
            TestQuest(4, "DevOps для разработчиков: Docker & Kubernetes",
                      "Основы контейнеризации и оркестрации. Практика с реальными проектами, деплой микросервисов и CI/CD пайплайны.",
                      "Технологии"),

            # Здоровье и спорт (ID: 5-9)
            TestQuest(5, "Утренний энерджайзер: 21 день ранних подъемов",
                      "Сформируйте привычку раннего пробуждения через постепенное изменение режима.",
                      "Здоровье"),
            TestQuest(6, "Функциональный тренинг: тело за 30 дней",
                      "Комплекс упражнений для развития силы, выносливости и гибкости без оборудования.",
                      "Спорт"),
            TestQuest(7, "Йога для офисных работников",
                      "Ежедневные 15-минутные комплексы для снятия напряжения в спине и шее.",
                      "Здоровье"),
            TestQuest(8, "Марафон здорового питания",
                      "4 недели сбалансированного питания с рецептами и планом покупок.",
                      "Здоровье"),
            TestQuest(9, "Бег для начинающих: от 0 до 5км",
                      "Постепенная программа для тех, кто никогда не бегал. Техника, дыхание, мотивация.",
                      "Спорт"),

            # Продуктивность (ID: 10-13)
            TestQuest(10, "Марафон глубокой работы: 40 часов фокуса",
                      "Интенсивный курс по развитию концентрации. Техники Pomodoro, digital detox.",
                      "Продуктивность"),
            TestQuest(11, "Привычки на миллион: 66 дней трансформации",
                      "Научный подход к формированию устойчивых привычек. От трекинга до автоматизации.",
                      "Продуктивность"),
            TestQuest(12, "Тайм-менеджмент для remote-работы",
                      "Эффективная организация удаленного рабочего дня. Приоритизация, борьба с прокрастинацией.",
                      "Продуктивность"),
            TestQuest(13, "Цифровой минимализм: наведение порядка",
                      "Оптимизация цифрового пространства для ясности ума. Чистка соцсетей, настройка уведомлений.",
                      "Продуктивность"),

            # Творчество (ID: 14-17)
            TestQuest(14, "Дизайн-мышление: создание продукта",
                      "Полный цикл от идеи до прототипа. Исследование пользователей, скетчинг, UX/UI основы.",
                      "Творчество"),
            TestQuest(15, "Цифровая иллюстрация в Procreate",
                      "Освоение iPad + Procreate с нуля. Брашпики, слои, анимация и создание стикерпаков.",
                      "Творчество"),
            TestQuest(16, "Фотография на iPhone: про уровень",
                      "Профессиональная съемка на смартфон. Композиция, свет, мобильный Lightroom.",
                      "Творчество"),
            TestQuest(17, "Кулинарный экспресс: 15 быстрых ужинов",
                      "Рецепты для занятых людей. Приготовление за 30 минут, meal prep, основы ножевой техники.",
                      "Творчество"),

            # Языки (ID: 18-19)
            TestQuest(18, "Английский для IT: 90 дней immersion",
                      "Специализированный курс технического английского. Документация, код-ревью, техники чтения.",
                      "Языки"),
            TestQuest(19, "Испанский с нуля: 60 дней до A1",
                      "Интенсив для начинающих. Основы грамматики, бытовые диалоги, испанская культура.",
                      "Языки"),

            # Карьера и финансы (ID: 20-22)
            TestQuest(20, "Карьерный рост: стратегия за 4 недели",
                      "Планирование карьерного развития, составление резюме, подготовка к собеседованиям.",
                      "Карьера"),
            TestQuest(21, "Инвестиции для начинающих: первые 100к",
                      "Безопасное введение в мир инвестиций. Акции, облигации, ETF, риск-менеджмент.",
                      "Финансы"),
            TestQuest(22, "Фриланс-старт: уход в свободное плавание",
                      "Пошаговый план перехода на фриланс. Поиск клиентов, ценообразование, юридические аспекты.",
                      "Карьера"),

            # Социальные навыки (ID: 23-24)
            TestQuest(23, "Эмоциональный интеллект: управление отношениями",
                      "Развитие EQ для карьерного и личного роста. Распознавание эмоций, эмпатия.",
                      "Социальные навыки"),
            TestQuest(24, "Публичные выступления: от страха к мастерству",
                      "Преодоление страха сцены через постепенную экспозицию. Техника речи, сторителлинг.",
                      "Социальные навыки"),

            # Ментальное здоровье (ID: 25-26)
            TestQuest(25, "Медитация для занятых людей",
                      "10-минутные практики для интеграции в рабочий день. Mindfulness, дыхательные техники.",
                      "Ментальное здоровье"),
            TestQuest(26, "Критическое мышление: фильтрация информации",
                      "Навыки анализа новостей, статей и исследований. Логические ошибки, когнитивные искажения.",
                      "Образование"),
        ]

        return quests

    @staticmethod
    def generate_realistic_users(quests: List[TestQuest]) -> List[TestUser]:
        """Генерация реалистичных пользователей с осмысленными профилями"""

        # Типичные профили пользователей платформы саморазвития
        user_profiles = [
            # 1. Программист-спортсмен (технологии + здоровье)
            {
                "id": 1,
                "name": "Алексей",
                "description": "Backend разработчик, занимается функциональным тренингом",
                "core_interests": ["Технологии", "Здоровье", "Спорт"],
                "quests_logic": [
                    # Уже прошел базовые технологии
                    lambda q: q.id in [0, 1, 2],  # Python, Веб, Data Science
                    # И некоторые спортивные
                    lambda q: q.id in [6, 9],  # Функциональный тренинг, Бег
                ]
            },

            # 2. Креативный дизайнер (творчество + продуктивность)
            {
                "id": 2,
                "name": "Мария",
                "description": "UI/UX дизайнер, хочет систематизировать работу",
                "core_interests": ["Творчество", "Продуктивность", "Социальные навыки"],
                "quests_logic": [
                    lambda q: q.id in [14, 15],  # Дизайн-мышление, Иллюстрация
                    lambda q: q.id in [10, 13],  # Глубокая работа, Цифровой минимализм
                ]
            },

            # 3. Начинающий инвестор (финансы + карьера)
            {
                "id": 3,
                "name": "Дмитрий",
                "description": "Менеджер проекта, интересуется инвестициями",
                "core_interests": ["Финансы", "Карьера", "Образование"],
                "quests_logic": [
                    lambda q: q.id in [20, 22],  # Карьерный рост, Фриланс
                    lambda q: q.id == 21,  # Инвестиции
                ]
            },

            # 4. Полиглот-путешественник (языки + здоровье)
            {
                "id": 4,
                "name": "Анна",
                "description": "Переводчик, любит активный отдых",
                "core_interests": ["Языки", "Здоровье", "Творчество"],
                "quests_logic": [
                    lambda q: q.id in [18, 19],  # Английский, Испанский
                    lambda q: q.id in [5, 7],  # Ранние подъемы, Йога
                ]
            },

            # 5. Предприниматель-гуру (продуктивность + социальные навыки)
            {
                "id": 5,
                "name": "Иван",
                "description": "Основатель стартапа, развивает soft skills",
                "core_interests": ["Продуктивность", "Социальные навыки", "Карьера"],
                "quests_logic": [
                    lambda q: q.id in [10, 11, 12],  # Все про продуктивность
                    lambda q: q.id in [23, 24],  # Эмоциональный интеллект, Выступления
                ]
            },

            # 6. Фитнес-тренер (спорт + здоровье + творчество)
            {
                "id": 6,
                "name": "Ольга",
                "description": "Персональный тренер, создает контент",
                "core_interests": ["Спорт", "Здоровье", "Творчество"],
                "quests_logic": [
                    lambda q: q.id in [6, 7, 8, 9],  # Все спортивное и здоровье
                    lambda q: q.id == 16,  # Фотография для соцсетей
                ]
            },

            # 7. Data Scientist (технологии + продуктивность)
            {
                "id": 7,
                "name": "Сергей",
                "description": "Аналитик данных, оптимизирует workflow",
                "core_interests": ["Технологии", "Продуктивность", "Образование"],
                "quests_logic": [
                    lambda q: q.id in [2, 3, 4],  # Data Science, SQL, DevOps
                    lambda q: q.id in [10, 12],  # Глубокая работа, Тайм-менеджмент
                ]
            },

            # 8. Студент-активист (образование + социальные навыки)
            {
                "id": 8,
                "name": "Екатерина",
                "description": "Студентка, развивает лидерские качества",
                "core_interests": ["Образование", "Социальные навыки", "Творчество"],
                "quests_logic": [
                    lambda q: q.id in [26],  # Критическое мышление
                    lambda q: q.id in [23, 24],  # Социальные навыки
                    lambda q: q.id == 15,  # Творчество
                ]
            },

            # 9. Digital nomad (продуктивность + языки + технологии)
            {
                "id": 9,
                "name": "Михаил",
                "description": "Удаленный разработчик, путешествует",
                "core_interests": ["Продуктивность", "Языки", "Технологии"],
                "quests_logic": [
                    lambda q: q.id in [0, 1],  # Программирование
                    lambda q: q.id in [12, 13],  # Remote работа + минимализм
                    lambda q: q.id == 18,  # Английский для IT
                ]
            },

            # 10. Ментор по осознанности (ментальное здоровье + социальные навыки)
            {
                "id": 10,
                "name": "Татьяна",
                "description": "Коуч, преподает mindfulness",
                "core_interests": ["Ментальное здоровье", "Социальные навыки", "Здоровье"],
                "quests_logic": [
                    lambda q: q.id in [25, 26],  # Медитация, Критическое мышление
                    lambda q: q.id in [23, 24],  # Эмоциональный интеллект
                    lambda q: q.id == 7,  # Йога
                ]
            },
        ]

        users = []
        for profile in user_profiles:
            # Собираем квесты по логике профиля
            completed_quests = []
            for logic_func in profile["quests_logic"]:
                for quest in quests:
                    if logic_func(quest) and quest.id not in completed_quests:
                        completed_quests.append(quest.id)

            # Создаем пользователя
            user = TestUser(
                id=profile["id"],
                quests=completed_quests,
                preferred_categories=profile["core_interests"]
            )
            users.append(user)

        # Добавляем описания для логов
        for user in users:
            profile = next(p for p in user_profiles if p["id"] == user.id)
            user.description = profile["description"]

        return users

    @staticmethod
    def create_deterministic_ground_truth(users: List[TestUser], quests: List[TestQuest]) -> Dict[int, List[int]]:
        """Создание детерминированного ground truth на основе профилей пользователей"""

        ground_truth = {}

        # Логика рекомендаций для каждого типа пользователя
        recommendation_rules = {
            # Программист-спортсмен
            1: {
                "high_priority": ["Технологии", "Спорт", "Здоровье"],
                "medium_priority": ["Продуктивность", "Образование"],
                "exclude": ["Творчество", "Языки"],
                "specific_quests": [3, 4, 8],  # SQL, DevOps, Питание
            },

            # Креативный дизайнер
            2: {
                "high_priority": ["Творчество", "Продуктивность"],
                "medium_priority": ["Социальные навыки", "Ментальное здоровье"],
                "exclude": ["Технологии", "Спорт"],
                "specific_quests": [16, 17, 25],  # Фотография, Кулинария, Медитация
            },

            # Начинающий инвестор
            3: {
                "high_priority": ["Финансы", "Карьера"],
                "medium_priority": ["Продуктивность", "Социальные навыки"],
                "exclude": ["Творчество", "Спорт"],
                "specific_quests": [12, 13, 23],  # Тайм-менеджмент, Минимализм, Эмоц. интеллект
            },

            # Полиглот-путешественник
            4: {
                "high_priority": ["Языки", "Здоровье"],
                "medium_priority": ["Творчество", "Ментальное здоровье"],
                "exclude": ["Технологии", "Финансы"],
                "specific_quests": [8, 16, 25],  # Питание, Фотография, Медитация
            },

            # Предприниматель-гуру
            5: {
                "high_priority": ["Продуктивность", "Социальные навыки"],
                "medium_priority": ["Карьера", "Ментальное здоровье"],
                "exclude": ["Спорт", "Творчество"],
                "specific_quests": [21, 25, 26],  # Инвестиции, Медитация, Критич. мышление
            },

            # Фитнес-тренер
            6: {
                "high_priority": ["Спорт", "Здоровье"],
                "medium_priority": ["Творчество", "Социальные навыки"],
                "exclude": ["Технологии", "Финансы"],
                "specific_quests": [17, 23, 24],  # Кулинария, Эмоц. интеллект, Выступления
            },

            # Data Scientist
            7: {
                "high_priority": ["Технологии", "Продуктивность"],
                "medium_priority": ["Образование", "Ментальное здоровье"],
                "exclude": ["Творчество", "Спорт"],
                "specific_quests": [0, 1, 25],  # Python, Веб, Медитация
            },

            # Студент-активист
            8: {
                "high_priority": ["Образование", "Социальные навыки"],
                "medium_priority": ["Творчество", "Ментальное здоровье"],
                "exclude": ["Финансы", "Технологии"],
                "specific_quests": [14, 17, 25],  # Дизайн-мышление, Кулинария, Медитация
            },

            # Digital nomad
            9: {
                "high_priority": ["Продуктивность", "Языки"],
                "medium_priority": ["Технологии", "Ментальное здоровье"],
                "exclude": ["Спорт", "Финансы"],
                "specific_quests": [19, 25, 26],  # Испанский, Медитация, Критич. мышление
            },

            # Ментор по осознанности
            10: {
                "high_priority": ["Ментальное здоровье", "Социальные навыки"],
                "medium_priority": ["Здоровье", "Образование"],
                "exclude": ["Технологии", "Финансы"],
                "specific_quests": [5, 8, 16],  # Ранние подъемы, Питание, Фотография
            },
        }

        for user in users:
            relevant_quests = []
            user_completed = set(user.quests)
            rules = recommendation_rules.get(user.id, {})

            if not rules:
                ground_truth[user.id] = []
                continue

            # 1. Высокоприоритетные категории (обязательно должны быть в рекомендациях)
            high_prio_quests = []
            for quest in quests:
                if (quest.category in rules["high_priority"] and
                        quest.id not in user_completed and
                        quest.category not in rules.get("exclude", [])):
                    high_prio_quests.append(quest.id)

            # Берем 3-5 квестов из высокоприоритетных
            relevant_quests.extend(high_prio_quests[:5])

            # 2. Среднеприоритетные категории
            medium_prio_quests = []
            for quest in quests:
                if (quest.category in rules["medium_priority"] and
                        quest.id not in user_completed and
                        quest.id not in relevant_quests and
                        quest.category not in rules.get("exclude", [])):
                    medium_prio_quests.append(quest.id)

            relevant_quests.extend(medium_prio_quests[:3])

            # 3. Специфичные квесты (ручная настройка)
            for quest_id in rules.get("specific_quests", []):
                if (quest_id not in user_completed and
                        quest_id not in relevant_quests and
                        quest_id < len(quests)):
                    relevant_quests.append(quest_id)

            # 4. Для разнообразия - 1-2 квеста из других категорий (кроме исключенных)
            other_quests = []
            for quest in quests:
                if (quest.id not in user_completed and
                        quest.id not in relevant_quests and
                        quest.category not in rules.get("exclude", []) and
                        quest.category not in rules.get("high_priority", []) and
                        quest.category not in rules.get("medium_priority", [])):
                    other_quests.append(quest.id)

            relevant_quests.extend(other_quests[:2])

            ground_truth[user.id] = relevant_quests[:10]  # Ограничиваем 10 квестами

        return ground_truth

    @staticmethod
    def create_deterministic_friends_ground_truth(users: List[TestUser]) -> Dict[int, List[int]]:
        """Создание детерминированного ground truth для друзей"""

        ground_truth = {}

        # Логика формирования друзей по группам интересов
        interest_groups = {
            "tech_health": [1, 7, 9],  # Технологии + здоровье/продуктивность
            "creative_productivity": [2, 5, 8],  # Творчество + продуктивность
            "finance_career": [3, 5],  # Финансы + карьера
            "languages_wellness": [4, 6, 10],  # Языки + wellness
            "sports_creativity": [6, 4],  # Спорт + творчество
            "mindfulness_social": [10, 8, 2],  # Медитация + социальные навыки
        }

        # Обратное отображение: пользователь -> группы
        user_to_groups = {}
        for group_name, user_ids in interest_groups.items():
            for user_id in user_ids:
                if user_id not in user_to_groups:
                    user_to_groups[user_id] = []
                user_to_groups[user_id].append(group_name)

        for user in users:
            friends = []
            user_groups = user_to_groups.get(user.id, [])

            # 1. Друзья из тех же групп интересов
            for group_name in user_groups:
                for potential_friend_id in interest_groups[group_name]:
                    if (potential_friend_id != user.id and
                            potential_friend_id not in friends):
                        friends.append(potential_friend_id)

            # 2. Дополнительные связи по комплементарным интересам
            complementary_pairs = {
                1: [7, 9],  # Программист-спортсмен с другими технарями
                2: [6, 10],  # Дизайнер с творческими и mindfulness
                3: [5],  # Инвестор с предпринимателем
                4: [6, 10],  # Полиглот с wellness-энтузиастами
                5: [3, 7],  # Предприниматель с финансистом и аналитиком
                6: [2, 4],  # Тренер с творческими
                7: [1, 5],  # Data Scientist с программистом и предпринимателем
                8: [2, 10],  # Студент с творческими и mindfulness
                9: [1, 4],  # Digital nomad с программистом и полиглотом
                10: [2, 8],  # Ментор с творческими и студентом
            }

            for friend_id in complementary_pairs.get(user.id, []):
                if friend_id not in friends:
                    friends.append(friend_id)

            # Ограничиваем 5 друзьями
            ground_truth[user.id] = friends[:5]

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
                        queries: Dict[str, List[int]],
                        top_k: int = 5) -> Dict[str, float]:
        """Улучшенная оценка поисковых результатов с учетом ранжирования"""

        metrics = {
            "precision@1": [],
            "precision@3": [],
            "precision@5": [],
            "recall@5": [],
            "f1@5": [],
            "mrr": [],  # Mean Reciprocal Rank
            "ndcg@5": [],  # Normalized Discounted Cumulative Gain
            "map@5": []  # Mean Average Precision
        }

        total_queries_processed = 0

        for query, results in search_results.items():
            if query not in queries:
                continue

            true_relevant = set(queries[query])
            retrieved = results[:top_k]  # Берем только top_k результатов

            # Пропускаем запросы без ground truth
            if not true_relevant:
                continue

            # Пропускаем запросы, где алгоритм ничего не нашел
            if not retrieved:
                continue

            total_queries_processed += 1

            # Precision@K для разных K
            for k in [1, 3, 5]:
                if len(retrieved) >= k:
                    hits_at_k = len(set(retrieved[:k]) & true_relevant)
                    precision_at_k = hits_at_k / k
                    metrics[f"precision@{k}"].append(precision_at_k)
                else:
                    # Если результатов меньше, чем k, считаем по доступным
                    hits_at_k = len(set(retrieved) & true_relevant)
                    precision_at_k = hits_at_k / len(retrieved) if retrieved else 0
                    metrics[f"precision@{k}"].append(precision_at_k)

            # Recall@5
            hits = len(set(retrieved) & true_relevant)
            recall = hits / len(true_relevant) if true_relevant else 0
            metrics["recall@5"].append(recall)

            # F1@5
            precision_at_5 = hits / len(retrieved) if retrieved else 0
            f1 = 2 * precision_at_5 * recall / (precision_at_5 + recall) if (precision_at_5 + recall) > 0 else 0
            metrics["f1@5"].append(f1)

            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for rank, quest_id in enumerate(retrieved, 1):
                if quest_id in true_relevant:
                    mrr = 1 / rank
                    break
            metrics["mrr"].append(mrr)

            # NDCG@5
            dcg = 0
            for i, quest_id in enumerate(retrieved, 1):
                if quest_id in true_relevant:
                    dcg += 1 / np.log2(i + 1)  # log2(i+1) потому что i начинается с 1

            # Ideal DCG (сортировка релевантных результатов на первые позиции)
            ideal_retrieved = min(len(true_relevant), len(retrieved))
            idcg = sum(1 / np.log2(i + 1) for i in range(1, ideal_retrieved + 1))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics["ndcg@5"].append(ndcg)

            # Average Precision@5
            ap = 0
            num_hits = 0
            relevant_count = min(len(true_relevant), len(retrieved))

            if relevant_count > 0:
                for i, quest_id in enumerate(retrieved, 1):
                    if quest_id in true_relevant:
                        num_hits += 1
                        ap += num_hits / i
                ap = ap / relevant_count
            metrics["map@5"].append(ap)

        # Логируем статистику
        logger.info(f"Обработано запросов: {total_queries_processed} из {len(search_results)}")

        # Усредняем метрики
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = np.mean(values)
                results[f"{metric_name}_std"] = np.std(values)
            else:
                results[metric_name] = 0
                results[f"{metric_name}_std"] = 0

        # Добавляем дополнительную информацию
        results["total_queries_processed"] = total_queries_processed

        return results

    def evaluate_search_with_explanations(self, search_results: Dict[str, List[Dict[str, Any]]],
                                          queries: Dict[str, List[int]],
                                          quests: List[TestQuest]) -> Dict[str, Any]:
        """Оценка поиска с объяснениями и анализом качества"""

        detailed_analysis = {
            "queries": {},
            "summary": {},
            "algorithm_comparison": {}
        }

        # Собираем метрики для каждого запроса
        for query, results in search_results.items():
            if query not in queries:
                continue

            true_relevant = set(queries[query])
            retrieved_ids = [r["quest_id"] for r in results[:5]]

            analysis = {
                "expected_count": len(true_relevant),
                "found_count": len(retrieved_ids),
                "hits": [],
                "misses": [],
                "false_positives": [],
                "precision@1": 0,
                "precision@3": 0,
                "precision@5": 0,
                "recall": 0
            }

            # Анализ попаданий и промахов
            for i, result in enumerate(results[:5]):
                quest_id = result["quest_id"]
                score = result.get("score", 0)

                if quest_id in true_relevant:
                    analysis["hits"].append({
                        "quest_id": quest_id,
                        "title": quests[quest_id].title if quest_id < len(quests) else "Unknown",
                        "rank": i + 1,
                        "score": score
                    })
                else:
                    analysis["false_positives"].append({
                        "quest_id": quest_id,
                        "title": quests[quest_id].title if quest_id < len(quests) else "Unknown",
                        "rank": i + 1,
                        "score": score,
                        "category": quests[quest_id].category if quest_id < len(quests) else "Unknown"
                    })

            # Не найденные релевантные квесты
            for quest_id in true_relevant:
                if quest_id not in retrieved_ids:
                    analysis["misses"].append({
                        "quest_id": quest_id,
                        "title": quests[quest_id].title if quest_id < len(quests) else "Unknown",
                        "category": quests[quest_id].category if quest_id < len(quests) else "Unknown"
                    })

            # Рассчитываем метрики
            hits_count = len(analysis["hits"])

            for k in [1, 3, 5]:
                hits_at_k = len([h for h in analysis["hits"] if h["rank"] <= k])
                analysis[f"precision@{k}"] = hits_at_k / min(k, len(retrieved_ids))

            analysis["recall"] = hits_count / len(true_relevant) if true_relevant else 0
            analysis["f1"] = 2 * analysis["precision@5"] * analysis["recall"] / (
                        analysis["precision@5"] + analysis["recall"]) if (analysis["precision@5"] + analysis[
                "recall"]) > 0 else 0

            detailed_analysis["queries"][query] = analysis

        # Сводная статистика
        all_precisions = []
        all_recalls = []

        for query_analysis in detailed_analysis["queries"].values():
            all_precisions.append(query_analysis["precision@5"])
            all_recalls.append(query_analysis["recall"])

        if all_precisions:
            detailed_analysis["summary"] = {
                "avg_precision@5": np.mean(all_precisions),
                "avg_recall": np.mean(all_recalls),
                "avg_f1": 2 * np.mean(all_precisions) * np.mean(all_recalls) / (
                            np.mean(all_precisions) + np.mean(all_recalls)) if (np.mean(all_precisions) + np.mean(
                    all_recalls)) > 0 else 0,
                "total_queries": len(detailed_analysis["queries"]),
                "avg_expected_per_query": np.mean([a["expected_count"] for a in detailed_analysis["queries"].values()]),
                "avg_found_per_query": np.mean([a["found_count"] for a in detailed_analysis["queries"].values()])
            }

        return detailed_analysis

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

#
# def run_comprehensive_benchmark():
#     """Запуск комплексного бенчмарка всех алгоритмов"""
#
#     logger.info("=" * 60)
#     logger.info("ЗАПУСК КОМПЛЕКСНОГО БЕНЧМАРКА АЛГОРИТМОВ")
#     logger.info("=" * 60)
#
#     # 1. Генерируем тестовые данные
#     logger.info("Генерация тестовых данных...")
#     quests = TestDataGenerator.generate_quests()
#     users = TestDataGenerator.generate_users(n_users=50, quests=quests)
#     ground_truth = TestDataGenerator.create_ground_truth(users, quests)
#
#     logger.info(f"Сгенерировано: {len(quests)} квестов, {len(users)} пользователей")
#     logger.info(f"Категории квестов: {list(set(q.category for q in quests))}")
#
#     # 2. Создаем тестовые запросы для поиска
#     search_queries = {
#         # Технологии и программирование
#         "изучить python программирование": [q.id for q in quests
#                                             if any(word in q.title.lower() or word in q.description.lower()
#                                                    for word in ["python", "программирование", "код"])],
#
#         "web разработка создание сайта": [q.id for q in quests
#                                           if any(word in q.title.lower() or word in q.description.lower()
#                                                  for word in ["веб", "html", "css", "js", "сайт", "разработка"])],
#
#         # Продуктивность
#         "тайм менеджмент продуктивность": [q.id for q in quests
#                                            if any(word in q.title.lower() or word in q.description.lower()
#                                                   for word in ["тайм", "продуктивность", "фокус", "привычки"])],
#
#         # Здоровье и спорт
#         "фитнес тренировки здоровье": [q.id for q in quests
#                                        if any(word in q.title.lower() or word in q.description.lower()
#                                               for word in ["фитнес", "тренировка", "здоровье", "спорт"])],
#
#         # Карьера и финансы
#         "карьера развитие инвестиции": [q.id for q in quests
#                                         if any(word in q.title.lower() or word in q.description.lower()
#                                                for word in ["карьера", "резюме", "инвестиции", "фриланс"])],
#
#         # Языки
#         "английский язык обучение": [q.id for q in quests
#                                      if any(word in q.title.lower() or word in q.description.lower()
#                                             for word in ["английский", "язык", "обучение", "грамматика"])],
#
#         # Творчество
#         "дизайн творчество рисование": [q.id for q in quests
#                                         if any(word in q.title.lower() or word in q.description.lower()
#                                                for word in ["дизайн", "творчество", "рисование", "иллюстрация"])]
#     }
#
#     # 3. Инициализируем алгоритмы
#     algorithms = [
#         ContentBasedBERT(),  # Ваш текущий алгоритм
#         CollaborativeFilteringKNN(n_neighbors=5),
#         CollaborativeFilteringKNN(n_neighbors=10),
#         HybridRecommender(content_weight=0.5, collaborative_weight=0.5),
#         HybridRecommender(content_weight=0.6, collaborative_weight=0.4),
#         HybridRecommender(content_weight=0.8, collaborative_weight=0.2),
#         SimpleTfidfRecommender()
#     ]
#
#     # 4. Загружаем данные во все алгоритмы
#     logger.info("Загрузка данных в алгоритмы...")
#     for algo in algorithms:
#         logger.info(f"  Загружаем {algo.name}...")
#         algo.load_data(quests, users)
#
#     # 5. Запускаем тесты рекомендаций
#     logger.info("\n" + "=" * 60)
#     logger.info("ТЕСТИРОВАНИЕ РЕКОМЕНДАЦИЙ")
#     logger.info("=" * 60)
#
#     recommendation_results = {}
#     search_results = {}
#
#     for algo in algorithms:
#         logger.info(f"\nТестируем алгоритм: {algo.name}")
#
#         # Тестирование рекомендаций
#         user_recommendations = {}
#         recommendation_times = []
#
#         for user in users[:20]:  # Тестируем на 20 пользователях для скорости
#             recommendations, exec_time = algo.recommend_quests(user.id, top_k=10)
#             user_recommendations[user.id] = [r["quest_id"] for r in recommendations]
#             recommendation_times.append(exec_time)
#
#         recommendation_results[algo.name] = user_recommendations
#
#         # Тестирование поиска (если алгоритм поддерживает)
#         algo_search_results = {}
#         search_times = []
#
#         if hasattr(algo, 'search_quests'):
#             for query in search_queries.keys():
#                 results, exec_time = algo.search_quests(query, top_k=10)
#                 algo_search_results[query] = [r["quest_id"] for r in results]
#                 search_times.append(exec_time)
#
#             search_results[algo.name] = algo_search_results
#
#         # Выводим метрики производительности
#         avg_rec_time = np.mean(recommendation_times) if recommendation_times else 0
#         avg_search_time = np.mean(search_times) if search_times else 0
#
#         logger.info(f"  Среднее время рекомендации: {avg_rec_time:.2f} мс")
#         logger.info(
#             f"  Среднее время поиска: {avg_search_time:.2f} мс" if search_times else "  Поиск не поддерживается")
#
#         # 6. Запускаем тесты рекомендаций друзей
#     logger.info("\n" + "=" * 60)
#     logger.info("ТЕСТИРОВАНИЕ РЕКОМЕНДАЦИЙ ДРУЗЕЙ")
#     logger.info("=" * 60)
#
#     friends_ground_truth = TestDataGenerator.create_friends_ground_truth(users)
#     friends_recommendation_results = {}
#
#     for algo in algorithms:
#         logger.info(f"\nТестируем алгоритм: {algo.name}")
#
#         if not hasattr(algo, 'recommend_friends'):
#             logger.info("  Рекомендации друзей не поддерживаются")
#             continue
#
#         # Тестирование рекомендаций друзей
#         user_friend_recommendations = {}
#         friend_recommendation_times = []
#
#         for user in users[:10]:  # Тестируем на 10 пользователях
#             try:
#                 recommendations, exec_time = algo.recommend_friends(user.id, top_k=5)
#                 user_friend_recommendations[user.id] = [r["user_id"] for r in recommendations]
#                 friend_recommendation_times.append(exec_time)
#             except Exception as e:
#                 logger.warning(f"  Ошибка рекомендаций друзей для пользователя {user.id}: {e}")
#                 user_friend_recommendations[user.id] = []
#
#         friends_recommendation_results[algo.name] = user_friend_recommendations
#
#         avg_friend_time = np.mean(friend_recommendation_times) if friend_recommendation_times else 0
#         logger.info(f"  Среднее время рекомендации друзей: {avg_friend_time:.2f} мс")
#
#     # 7. Оцениваем качество рекомендаций друзей
#     logger.info("\n" + "=" * 60)
#     logger.info("ОЦЕНКА КАЧЕСТВА РЕКОМЕНДАЦИЙ ДРУЗЕЙ")
#     logger.info("=" * 60)
#
#     evaluator = BenchmarkEvaluator(ground_truth)  # Можно передать любую ground truth
#
#     friends_metrics = {}
#     for algo_name, recommendations in friends_recommendation_results.items():
#         if recommendations:
#             metrics = evaluator.evaluate_friends(recommendations, friends_ground_truth, top_k=3)
#             friends_metrics[algo_name] = metrics
#
#             logger.info(f"\n{algo_name}:")
#             for metric_name, value in metrics.items():
#                 if "std" not in metric_name:
#                     logger.info(f"  {metric_name}: {value:.4f}")
#
#     # 8. Оцениваем качество рекомендаций
#     logger.info("\n" + "=" * 60)
#     logger.info("ОЦЕНКА КАЧЕСТВА РЕКОМЕНДАЦИЙ")
#     logger.info("=" * 60)
#
#     evaluator = BenchmarkEvaluator(ground_truth)
#
#     all_metrics = {}
#     for algo_name, recommendations in recommendation_results.items():
#         metrics = evaluator.evaluate(recommendations, top_k=5)
#         all_metrics[algo_name] = metrics
#
#         logger.info(f"\n{algo_name}:")
#         for metric_name, value in metrics.items():
#             if "std" not in metric_name:
#                 logger.info(f"  {metric_name}: {value:.4f}")
#
#     # 9. Оцениваем качество поиска
#     logger.info("\n" + "=" * 60)
#     logger.info("ОЦЕНКА КАЧЕСТВА ПОИСКА")
#     logger.info("=" * 60)
#
#     for algo_name, search_res in search_results.items():
#         if search_res:  # Если алгоритм поддерживает поиск
#             search_metrics = evaluator.evaluate_search(search_res, search_queries)
#             logger.info(f"\n{algo_name} (поиск):")
#             for metric_name, value in search_metrics.items():
#                 logger.info(f"  {metric_name}: {value:.4f}")
#
#     # 10. Визуализация результатов
#     logger.info("\n" + "=" * 60)
#     logger.info("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
#     logger.info("=" * 60)
#
#     visualize_results(all_metrics, algorithms)
#
#     # 11. Сохранение результатов в файл
#     save_results_to_file(all_metrics, friends_metrics, algorithms, quests, users)
#
#     logger.info("\n" + "=" * 60)
#     logger.info("БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО!")
#     logger.info("=" * 60)
#
#     return all_metrics, friends_metrics

def create_better_search_queries(quests: List[TestQuest]) -> Dict[str, List[int]]:
    """Создание более качественных тестовых запросов для поиска"""

    # Группируем квесты по темам
    quest_groups = {
        "python": [q for q in quests if "python" in q.title.lower()],
        "веб разработка": [q for q in quests if any(word in q.title.lower()
                                                    for word in ["веб", "html", "css", "javascript", "сайт"])],
        "data science": [q for q in quests if "data science" in q.title.lower()],
        "sql": [q for q in quests if "sql" in q.title.lower()],
        "здоровье": [q for q in quests if "здоров" in q.title.lower()],
        "спорт": [q for q in quests if any(word in q.title.lower()
                                           for word in ["спорт", "трениров", "фитнес", "бег", "йога"])],
        "продуктивность": [q for q in quests if any(word in q.title.lower()
                                                    for word in ["продуктив", "фокус", "привычк", "тайм-менеджмент"])],
        "дизайн": [q for q in quests if any(word in q.title.lower()
                                            for word in ["дизайн", "иллюстрац", "рисован", "творчеств"])],
        "английский": [q for q in quests if "английск" in q.title.lower()],
        "карьера": [q for q in quests if any(word in q.title.lower()
                                             for word in ["карьер", "резюме", "собеседован", "фриланс"])],
        "инвестиции": [q for q in quests if "инвестиц" in q.title.lower()],
        "медитация": [q for q in quests if any(word in q.title.lower()
                                               for word in ["медитац", "mindfulness", "осознанност"])],
    }

    # Убираем пустые группы
    quest_groups = {k: v for k, v in quest_groups.items() if v}

    # Создаем запросы разной сложности
    search_queries = {}

    # 1. Простые точные запросы (высокая точность ожидается)
    simple_queries = {
        "python программирование": [q.id for q in quest_groups.get("python", [])],
        "изучить python": [q.id for q in quest_groups.get("python", [])],
        "курс python": [q.id for q in quest_groups.get("python", [])],
        "веб разработка": [q.id for q in quest_groups.get("веб разработка", [])],
        "создать сайт": [q.id for q in quest_groups.get("веб разработка", [])],
        "html css javascript": [q.id for q in quest_groups.get("веб разработка", [])],
        "анализ данных": [q.id for q in quest_groups.get("data science", [])],
        "data science": [q.id for q in quest_groups.get("data science", [])],
        "изучить sql": [q.id for q in quest_groups.get("sql", [])],
        "запросы sql": [q.id for q in quest_groups.get("sql", [])],
        "утренний подъем": [q.id for q in quests if "утренний" in q.title.lower()],
        "функциональный тренинг": [q.id for q in quests if "функциональный" in q.title.lower()],
        "глубокая работа": [q.id for q in quests if "глубокой" in q.title.lower()],
        "привычки": [q.id for q in quests if "привычки" in q.title.lower()],
        "дизайн мышление": [q.id for q in quest_groups.get("дизайн", [])],
        "цифровая иллюстрация": [q.id for q in quests if "иллюстрация" in q.title.lower()],
        "английский для it": [q.id for q in quest_groups.get("английский", [])],
        "технический английский": [q.id for q in quest_groups.get("английский", [])],
        "карьерный рост": [q.id for q in quest_groups.get("карьера", [])],
        "инвестиции для начинающих": [q.id for q in quest_groups.get("инвестиции", [])],
        "медитация mindfulness": [q.id for q in quest_groups.get("медитация", [])],
    }

    # Убираем запросы без ожидаемых результатов
    simple_queries = {k: v for k, v in simple_queries.items() if v}

    # 2. Сложные запросы (несколько тем) - но не все подряд!
    complex_queries = {
        "программирование и разработка": list(set(
            [q.id for q in quest_groups.get("python", [])] +
            [q.id for q in quest_groups.get("веб разработка", [])]
        )),
        "здоровье и продуктивность": list(set(
            [q.id for q in quest_groups.get("здоровье", [])] +
            [q.id for q in quest_groups.get("продуктивность", [])]
        )),
        "творчество и дизайн": list(set(
            [q.id for q in quest_groups.get("дизайн", [])]
        )),
        "карьера и финансы": list(set(
            [q.id for q in quest_groups.get("карьера", [])] +
            [q.id for q in quest_groups.get("инвестиции", [])]
        )),
    }

    # Убираем пустые запросы
    complex_queries = {k: v for k, v in complex_queries.items() if v}

    # 3. Синонимичные запросы (проверка понимания семантики)
    synonym_queries = {
        # Для программирования
        "научиться писать код": [q.id for q in quest_groups.get("python", [])],
        "освоить программирование": [q.id for q in quest_groups.get("python", [])],

        # Для здоровья
        "утро бодрость энергия": [q.id for q in quests if "утренний" in q.title.lower()],
        "фитнес упражнения тренировка": [q.id for q in quest_groups.get("спорт", [])],
        "йога для расслабления": [q.id for q in quests if "йога" in q.title.lower()],

        # Для продуктивности
        "концентрация внимание фокус": [q.id for q in quest_groups.get("продуктивность", [])],
        "управление временем эффективность": [q.id for q in quest_groups.get("продуктивность", [])],
        "цифровая гигиена": [q.id for q in quests if "цифровой минимализм" in q.title],

        # Для творчества
        "рисование на ipad": [q.id for q in quests if "иллюстрация" in q.title.lower()],
        "фотосъемка на телефон": [q.id for q in quests if "фотография" in q.title.lower()],
        "готовка ужинов быстро": [q.id for q in quests if "кулинарный" in q.title.lower()],
    }

    # Убираем пустые запросы
    synonym_queries = {k: v for k, v in synonym_queries.items() if v}

    # Объединяем все запросы
    search_queries.update(simple_queries)
    search_queries.update(complex_queries)
    search_queries.update(synonym_queries)

    # Убираем дубликаты в списках
    for query, quest_ids in search_queries.items():
        search_queries[query] = list(set(quest_ids))

    return search_queries

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


# def run_specific_tests():
#     """Запуск специфических тестов для дипломной работы"""
#
#     logger.info("\n" + "=" * 60)
#     logger.info("СПЕЦИАЛЬНЫЕ ТЕСТЫ ДЛЯ ДИПЛОМНОЙ РАБОТЫ")
#     logger.info("=" * 60)
#
#     # Тест 1: Cold-start проблема
#     logger.info("\nТест 1: Cold-start проблема (пользователь без истории)")
#
#     quests = TestDataGenerator.generate_quests()
#
#     # Создаем пользователя без истории
#     cold_start_user = TestUser(
#         id=999,
#         quests=[],
#         preferred_categories=["Продуктивность", "Образование"]  # Более реалистичные категории
#     )
#
#     # Генерируем пользователей С ТЕМИ ЖЕ КВЕСТАМИ
#     users_with_history = TestDataGenerator.generate_users(n_users=20, quests=quests)
#
#     # Создаем новый список с холодным пользователем
#     all_users = users_with_history + [cold_start_user]
#
#     # Тестируем разные алгоритмы
#     algorithms = [
#         ContentBasedBERT(),
#         CollaborativeFilteringKNN(n_neighbors=5),
#         HybridRecommender(content_weight=0.6, collaborative_weight=0.4)
#     ]
#
#     for algo in algorithms:
#         try:
#             logger.info(f"\nТестируем {algo.name}...")
#             algo.load_data(quests, all_users)
#             recommendations, exec_time = algo.recommend_quests(cold_start_user.id, top_k=5)
#
#             logger.info(f"{algo.name}:")
#             if recommendations:
#                 logger.info(f"  Рекомендовано квестов: {len(recommendations)}")
#                 for rec in recommendations[:3]:
#                     logger.info(f"  - {rec['title']} (score: {rec['score']:.3f})")
#             else:
#                 logger.info("  Не смог дать рекомендации")
#         except Exception as e:
#             logger.error(f"  Ошибка: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#
#     # Тест 2: Время ответа при разном количестве квестов
#     logger.info("\n\nТест 2: Масштабируемость алгоритмов")
#
#     quest_counts = [50, 100, 200]
#     time_results = {}
#
#     for n_quests in quest_counts:
#         logger.info(f"\nКоличество квестов: {n_quests}")
#
#         # Генерируем квесты
#         test_quests = []
#         categories = ["Продуктивность", "Здоровье", "Образование", "Технологии"]
#
#         for i in range(n_quests):
#             category = random.choice(categories)
#             test_quests.append(TestQuest(
#                 id=i,
#                 title=f"Тестовый квест {i}",
#                 description=f"Описание тестового квеста {i} для категории {category}",
#                 category=category
#             ))
#
#         # Генерируем пользователей с ЭТИМИ квестами
#         test_users = TestDataGenerator.generate_users(n_users=20, quests=test_quests)
#
#         for algo in algorithms:
#             try:
#                 logger.info(f"  Тестируем {algo.name}...")
#                 # Измеряем время загрузки данных
#                 start_time = time.time()
#                 algo.load_data(test_quests, test_users)
#                 load_time = time.time() - start_time
#
#                 # Измеряем время рекомендации
#                 rec_times = []
#                 for user in test_users[:5]:
#                     _, exec_time = algo.recommend_quests(user.id, top_k=10)
#                     rec_times.append(exec_time)
#
#                 avg_rec_time = np.mean(rec_times) if rec_times else 0
#
#                 if algo.name not in time_results:
#                     time_results[algo.name] = {}
#
#                 time_results[algo.name][n_quests] = {
#                     "load_time": load_time,
#                     "avg_rec_time": avg_rec_time
#                 }
#
#                 logger.info(f"    загрузка={load_time:.2f}с, рекомендация={avg_rec_time:.2f}мс")
#             except Exception as e:
#                 logger.error(f"    ошибка: {e}")
#
#     # Тест 3: Влияние размера истории пользователя на качество
#     logger.info("\n\nТест 3: Влияние размера истории пользователя на качество")
#
#     test_quests = TestDataGenerator.generate_quests()
#
#     # Генерируем пользователей С ЭТИМИ КВЕСТАМИ
#     test_users = TestDataGenerator.generate_users(n_users=20, quests=test_quests)
#
#     history_sizes = [1, 3, 5, 10]
#     quality_results = {}
#
#     for algo in algorithms:
#         quality_results[algo.name] = {}
#
#         for history_size in history_sizes:
#             # Выбираем пользователя с достаточно большой историей
#             suitable_users = [u for u in test_users if len(u.quests) >= history_size]
#
#             if not suitable_users:
#                 logger.warning(f"  Нет пользователей с историей >= {history_size} для {algo.name}")
#                 continue
#
#             test_user = random.choice(suitable_users)
#
#             # Берем только первые history_size квестов
#             user_quests = test_user.quests[:history_size]
#
#             # Создаем тестового пользователя с ограниченной историей
#             test_user_limited = TestUser(
#                 id=test_user.id + 10000,  # Уникальный ID
#                 quests=user_quests,
#                 preferred_categories=test_user.preferred_categories
#             )
#
#             try:
#                 # Загружаем данные для одного пользователя
#                 algo.load_data(test_quests, [test_user_limited])
#
#                 # Получаем рекомендации
#                 recommendations, _ = algo.recommend_quests(test_user_limited.id, top_k=10)
#
#                 quality_results[algo.name][history_size] = len(recommendations)
#             except Exception as e:
#                 logger.error(f"  Ошибка для {algo.name} с историей {history_size}: {e}")
#
#     # Выводим результаты
#     logger.info("\nЗависимость качества от размера истории:")
#     for algo_name, results in quality_results.items():
#         logger.info(f"\n{algo_name}:")
#         for size, count in sorted(results.items()):
#             logger.info(f"  История {size} квестов -> {count} рекомендаций")
#
#
# if __name__ == "__main__":
#     # Запуск основного бенчмарка
#     main_results, friends_results = run_comprehensive_benchmark()
#
#     # Запуск дополнительных тестов
#     run_specific_tests()
#
#     logger.info("\n" + "=" * 60)
#     logger.info("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
#     logger.info("=" * 60)
#     logger.info("\nРезультаты сохранены в файлы:")
#     logger.info("  - benchmark_results.json (полные результаты)")
#     logger.info("  - benchmark_results.png (графики)")
#     logger.info("  - benchmark_report.txt (текстовый отчет)")


def run_realistic_benchmark():
    """Запуск реалистичного бенчмарка с типичными пользователями"""

    logger.info("=" * 60)
    logger.info("РЕАЛИСТИЧНЫЙ БЕНЧМАРК С ТИПИЧНЫМИ ПОЛЬЗОВАТЕЛЯМИ")
    logger.info("=" * 60)

    # 1. Генерируем реалистичные данные
    logger.info("Генерация реалистичных данных...")
    quests = RealisticTestDataGenerator.generate_quests()
    users = RealisticTestDataGenerator.generate_realistic_users(quests)

    # Логируем профили пользователей
    logger.info("\nПрофили пользователей:")
    for user in users:
        completed_titles = [quests[qid].title[:30] + "..." for qid in user.quests[:3]]
        logger.info(f"ID {user.id}: {getattr(user, 'description', 'Нет описания')}")
        logger.info(f"  Интересы: {user.preferred_categories}")
        logger.info(f"  Завершено квестов: {len(user.quests)} (напр.: {', '.join(completed_titles)})")

    # 2. Создаем детерминированный ground truth
    ground_truth = RealisticTestDataGenerator.create_deterministic_ground_truth(users, quests)
    friends_ground_truth = RealisticTestDataGenerator.create_deterministic_friends_ground_truth(users)

    # 3. Создаем тестовые запросы для поиска
    logger.info("Создание тестовых запросов для поиска...")
    search_queries = create_better_search_queries(quests)

    # Логируем статистику запросов
    query_stats = []
    for query, quest_ids in search_queries.items():
        query_stats.append({
            "query": query,
            "expected_results": len(quest_ids),
            "categories": list(set(quests[qid].category for qid in quest_ids))
        })

    logger.info(f"\nСтатистика поисковых запросов:")
    logger.info(f"  Всего запросов: {len(search_queries)}")

    # Группируем по количеству ожидаемых результатов
    counts = {}
    for stats in query_stats:
        count = stats["expected_results"]
        counts[count] = counts.get(count, 0) + 1

    for count, num_queries in sorted(counts.items()):
        logger.info(f"  Запросов с {count} ожидаемыми результатами: {num_queries}")

    # Показываем примеры запросов
    logger.info(f"\nПримеры тестовых запросов:")
    for query, quest_ids in list(search_queries.items())[:5]:
        quest_titles = [quests[qid].title[:30] + "..." for qid in quest_ids[:2]]
        logger.info(f"  '{query}' → {len(quest_ids)} квестов (напр.: {', '.join(quest_titles)})")

    # Логируем ожидаемые рекомендации
    logger.info("\nОжидаемые рекомендации (ground truth):")
    for user in users[:3]:  # Покажем для первых 3 пользователей
        logger.info(f"\nПользователь {user.id}:")
        expected_quests = ground_truth.get(user.id, [])
        if expected_quests:
            for qid in expected_quests[:5]:
                quest = quests[qid]
                logger.info(f"  - {quest.title} ({quest.category})")
        else:
            logger.info("  Нет ожидаемых рекомендаций")

    logger.info(f"\nСгенерировано: {len(quests)} квестов, {len(users)} пользователей")
    logger.info(f"Тестовых запросов для поиска: {len(search_queries)}")

    # 4. Инициализируем алгоритмы
    algorithms = [
        ContentBasedBERT(),
        CollaborativeFilteringKNN(n_neighbors=3),
        HybridRecommender(content_weight=0.7, collaborative_weight=0.3),
        SimpleTfidfRecommender()
    ]

    # 5. Запускаем тесты
    results = run_custom_benchmark(algorithms, quests, users, ground_truth, friends_ground_truth, search_queries)

    return results


def run_custom_benchmark(algorithms, quests, users, ground_truth, friends_ground_truth, search_queries=None):
    """Запуск кастомного бенчмарка с заданными данными"""

    if search_queries is None:
        search_queries = {}

    # Загружаем данные во все алгоритмы
    logger.info("\nЗагрузка данных в алгоритмы...")
    for algo in algorithms:
        logger.info(f"  Загружаем {algo.name}...")
        algo.load_data(quests, users)

    # Тестирование рекомендаций квестов
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТИРОВАНИЕ РЕКОМЕНДАЦИЙ КВЕСТОВ")
    logger.info("=" * 60)

    recommendation_results = {}
    recommendation_times_all = {}

    for algo in algorithms:
        logger.info(f"\nТестируем алгоритм: {algo.name}")

        user_recommendations = {}
        recommendation_times = []

        for user in users:
            recommendations, exec_time = algo.recommend_quests(user.id, top_k=10)
            user_recommendations[user.id] = [r["quest_id"] for r in recommendations]
            recommendation_times.append(exec_time)

            # Детальный лог для первых 2 пользователей
            if user.id <= 2 and recommendations:
                logger.info(f"\n  Рекомендации для пользователя {user.id}:")
                for rec in recommendations[:3]:
                    quest = quests[rec["quest_id"]]
                    logger.info(f"    - {quest.title} (score: {rec['score']:.3f}, категория: {quest.category})")

        recommendation_results[algo.name] = user_recommendations
        recommendation_times_all[algo.name] = recommendation_times

        avg_rec_time = np.mean(recommendation_times) if recommendation_times else 0
        logger.info(f"\n  Среднее время рекомендации: {avg_rec_time:.2f} мс")

    # Тестирование поиска квестов
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТИРОВАНИЕ ПОИСКА КВЕСТОВ")
    logger.info("=" * 60)

    search_results = {}
    search_times_all = {}

    for algo in algorithms:
        if not hasattr(algo, 'search_quests'):
            logger.info(f"\n{algo.name}: поиск не поддерживается")
            continue

        logger.info(f"\nТестируем поиск для алгоритма: {algo.name}")

        algo_search_results = {}
        search_times = []

        for query, expected_results in list(search_queries.items())[:10]:  # Тестируем 10 запросов
            try:
                results, exec_time = algo.search_quests(query, top_k=5)
                # results - это список словарей, нужно извлечь ID
                algo_search_results[query] = [r["quest_id"] for r in results]
                search_times.append(exec_time)

                # Детальный лог для первых 3 запросов
                if list(search_queries.keys()).index(query) < 3 and results:
                    logger.info(f"\n  Поиск по запросу: '{query}'")
                    logger.info(f"  Ожидаемые результаты: {expected_results}")
                    logger.info(f"  Найденные результаты:")
                    for rec in results[:3]:
                        quest = quests[rec["quest_id"]]
                        is_expected = "✓" if rec["quest_id"] in expected_results else " "
                        logger.info(f"    {is_expected} {quest.title} (score: {rec['score']:.3f})")
            except Exception as e:
                logger.warning(f"  Ошибка поиска по запросу '{query}': {e}")

        search_results[algo.name] = algo_search_results
        search_times_all[algo.name] = search_times

        if search_times:
            avg_search_time = np.mean(search_times)
            logger.info(f"\n  Среднее время поиска: {avg_search_time:.2f} мс")

    # Тестирование рекомендаций друзей
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТИРОВАНИЕ РЕКОМЕНДАЦИЙ ДРУЗЕЙ")
    logger.info("=" * 60)

    friends_recommendation_results = {}
    friend_times_all = {}

    for algo in algorithms:
        if not hasattr(algo, 'recommend_friends'):
            continue

        logger.info(f"\nТестируем алгоритм: {algo.name}")

        user_friend_recommendations = {}
        friend_recommendation_times = []

        for user in users:
            try:
                recommendations, exec_time = algo.recommend_friends(user.id, top_k=5)
                user_friend_recommendations[user.id] = [r["user_id"] for r in recommendations]
                friend_recommendation_times.append(exec_time)

                # Детальный лог для первых 2 пользователей
                if user.id <= 2 and recommendations:
                    logger.info(f"\n  Рекомендации друзей для пользователя {user.id}:")
                    for rec in recommendations[:3]:
                        friend_user = next(u for u in users if u.id == rec["user_id"])
                        logger.info(
                            f"    - Пользователь {rec['user_id']} (score: {rec.get('similarity_score', 0):.3f})")
                        logger.info(
                            f"      Общие интересы: {set(user.preferred_categories) & set(friend_user.preferred_categories)}")
            except Exception as e:
                logger.warning(f"  Ошибка рекомендаций друзей: {e}")

        friends_recommendation_results[algo.name] = user_friend_recommendations
        friend_times_all[algo.name] = friend_recommendation_times

    # Оценка качества
    logger.info("\n" + "=" * 60)
    logger.info("ОЦЕНКА КАЧЕСТВА")
    logger.info("=" * 60)

    evaluator = BenchmarkEvaluator(ground_truth)

    # Оценка рекомендаций квестов
    all_metrics = {}
    for algo_name, recommendations in recommendation_results.items():
        metrics = evaluator.evaluate(recommendations, top_k=5)
        all_metrics[algo_name] = metrics

        logger.info(f"\n{algo_name} (квесты):")
        logger.info(f"  Precision@5: {metrics['precision@k']:.3f}")
        logger.info(f"  Recall@5:    {metrics['recall@k']:.3f}")
        logger.info(f"  F1@5:        {metrics['f1@k']:.3f}")
        logger.info(f"  NDCG@5:      {metrics['ndcg@k']:.3f}")

    # Оценка поиска
    search_metrics = {}
    for algo_name, search_res in search_results.items():
        metrics = evaluator.evaluate_search(search_res, search_queries)
        search_metrics[algo_name] = metrics

        logger.info(f"\n{algo_name} (поиск):")
        logger.info(f"  Precision: {metrics.get('precision', 0):.3f}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.3f}")
        logger.info(f"  F1:        {metrics.get('f1', 0):.3f}")
        logger.info(f"  MRR:       {metrics.get('mrr', 0):.3f}")

    # Оценка рекомендаций друзей
    friends_metrics = {}
    for algo_name, recommendations in friends_recommendation_results.items():
        metrics = evaluator.evaluate_friends(recommendations, friends_ground_truth, top_k=3)
        friends_metrics[algo_name] = metrics

        logger.info(f"\n{algo_name} (друзья):")
        logger.info(f"  Precision@3: {metrics.get('friend_precision@k', 0):.3f}")
        logger.info(f"  Recall@3:    {metrics.get('friend_recall@k', 0):.3f}")
        logger.info(f"  F1@3:        {metrics.get('friend_f1@k', 0):.3f}")

    # Сохранение результатов
    save_comprehensive_results(
        quests=quests,
        users=users,
        algorithms=algorithms,
        quest_metrics=all_metrics,
        search_metrics=search_metrics,
        friend_metrics=friends_metrics,
        recommendation_results=recommendation_results,
        search_results=search_results,
        friend_results=friends_recommendation_results,
        rec_times=recommendation_times_all,
        search_times=search_times_all,
        friend_times=friend_times_all,
        ground_truth=ground_truth,
        search_queries=search_queries,
        friends_ground_truth=friends_ground_truth
    )

    return {
        "quest_metrics": all_metrics,
        "search_metrics": search_metrics,
        "friend_metrics": friends_metrics,
        "recommendations": recommendation_results,
        "search_results": search_results,
        "friend_recommendations": friends_recommendation_results
    }


def save_comprehensive_results(quests, users, algorithms, quest_metrics, search_metrics, friend_metrics,
                               recommendation_results, search_results, friend_results,
                               rec_times, search_times, friend_times,
                               ground_truth, search_queries, friends_ground_truth):
    """Комплексное сохранение результатов всех тестов"""

    import datetime
    import csv

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Сохраняем основные метрики в JSON
    results_data = {
        "timestamp": timestamp,
        "config": {
            "total_quests": len(quests),
            "total_users": len(users),
            "tested_users": len(users),
            "search_queries": len(search_queries)
        },
        "algorithms": {},
        "summary": {
            "best_quest_algorithm": None,
            "best_search_algorithm": None,
            "best_friend_algorithm": None
        }
    }

    # Заполняем данные по алгоритмам
    for algo in algorithms:
        algo_name = algo.name
        algo_info = algo.get_algorithm_info()

        results_data["algorithms"][algo_name] = {
            "info": algo_info,
            "quest_metrics": quest_metrics.get(algo_name, {}),
            "search_metrics": search_metrics.get(algo_name, {}),
            "friend_metrics": friend_metrics.get(algo_name, {}),
            "performance": {
                "avg_recommendation_time_ms": np.mean(rec_times.get(algo_name, [0])),
                "avg_search_time_ms": np.mean(search_times.get(algo_name, [0])),
                "avg_friend_recommendation_time_ms": np.mean(friend_times.get(algo_name, [0]))
            }
        }

    # Находим лучшие алгоритмы
    best_quest_algo = max(quest_metrics.items(), key=lambda x: x[1].get('f1@k', 0), default=(None, {}))[0]
    best_search_algo = max(search_metrics.items(), key=lambda x: x[1].get('f1', 0), default=(None, {}))[0]
    best_friend_algo = max(friend_metrics.items(),
                           key=lambda x: x[1].get('friend_f1@k', 0),
                           default=(None, {}))[0]

    results_data["summary"]["best_quest_algorithm"] = best_quest_algo
    results_data["summary"]["best_search_algorithm"] = best_search_algo
    results_data["summary"]["best_friend_algorithm"] = best_friend_algo

    # Сохраняем JSON
    json_filename = f"benchmark_results_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Основные результаты сохранены в {json_filename}")

    # 2. Сохраняем детальные рекомендации в CSV
    csv_filename = f"detailed_recommendations_{timestamp}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'User_ID', 'Recommended_Quest_IDs', 'Expected_Quest_IDs', 'Hits_Count'])

        for algo_name, user_recs in recommendation_results.items():
            for user_id, rec_ids in user_recs.items():
                expected = ground_truth.get(user_id, [])
                hits = len(set(rec_ids[:5]) & set(expected))
                writer.writerow([
                    algo_name,
                    user_id,
                    ','.join(map(str, rec_ids[:10])),
                    ','.join(map(str, expected)),
                    hits
                ])
    logger.info(f"Детальные рекомендации сохранены в {csv_filename}")

    # 3. Сохраняем результаты поиска в CSV
    if search_results:
        search_csv = f"search_results_{timestamp}.csv"
        with open(search_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm', 'Query', 'Found_Quest_IDs', 'Expected_Quest_IDs', 'Precision', 'Recall'])

            for algo_name, query_results in search_results.items():
                for query, found_ids in query_results.items():
                    expected = search_queries.get(query, [])
                    hits = len(set(found_ids) & set(expected))
                    precision = hits / len(found_ids) if found_ids else 0
                    recall = hits / len(expected) if expected else 0

                    writer.writerow([
                        algo_name,
                        query,
                        ','.join(map(str, found_ids)),
                        ','.join(map(str, expected)),
                        f"{precision:.3f}",
                        f"{recall:.3f}"
                    ])
        logger.info(f"Результаты поиска сохранены в {search_csv}")

    # 4. Сохраняем текстовый отчет
    report_filename = f"benchmark_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("КОМПЛЕКСНЫЙ ОТЧЕТ ПО ТЕСТИРОВАНИЮ АЛГОРИТМОВ\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Дата тестирования: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Количество квестов: {len(quests)}\n")
        f.write(f"Количество пользователей: {len(users)}\n")
        f.write(f"Количество тестовых запросов: {len(search_queries)}\n\n")

        f.write("ЛУЧШИЕ АЛГОРИТМЫ:\n")
        f.write(f"- Рекомендации квестов: {best_quest_algo}\n")
        f.write(f"- Поиск квестов: {best_search_algo}\n")
        f.write(f"- Рекомендации друзей: {best_friend_algo}\n\n")

        f.write("ДЕТАЛЬНЫЕ МЕТРИКИ:\n")
        for algo in algorithms:
            algo_name = algo.name
            f.write(f"\n{algo_name}:\n")

            if algo_name in quest_metrics:
                f.write("  Рекомендации квестов:\n")
                for metric, value in quest_metrics[algo_name].items():
                    if 'std' not in metric:
                        f.write(f"    - {metric}: {value:.3f}\n")

            if algo_name in search_metrics:
                f.write("  Поиск квестов:\n")
                for metric, value in search_metrics[algo_name].items():
                    f.write(f"    - {metric}: {value:.3f}\n")

            if algo_name in friend_metrics:
                f.write("  Рекомендации друзей:\n")
                for metric, value in friend_metrics[algo_name].items():
                    if 'std' not in metric:
                        f.write(f"    - {metric}: {value:.3f}\n")

    logger.info(f"Текстовый отчет сохранен в {report_filename}")


def analyze_specific_user(algorithms, user_id, quests, users, ground_truth):
    """Детальный анализ рекомендаций для конкретного пользователя"""

    logger.info(f"\n" + "=" * 60)
    logger.info(f"ДЕТАЛЬНЫЙ АНАЛИЗ ДЛЯ ПОЛЬЗОВАТЕЛЯ {user_id}")
    logger.info("=" * 60)

    user = next(u for u in users if u.id == user_id)
    logger.info(f"Профиль: {getattr(user, 'description', 'Нет описания')}")
    logger.info(f"Интересы: {user.preferred_categories}")

    completed = [quests[qid].title for qid in user.quests]
    logger.info(f"Завершенные квесты ({len(completed)}):")
    for title in completed[:5]:
        logger.info(f"  - {title}")
    if len(completed) > 5:
        logger.info(f"  ... и еще {len(completed) - 5}")

    logger.info(f"\nОжидаемые рекомендации (ground truth):")
    expected = ground_truth.get(user_id, [])
    for qid in expected:
        quest = quests[qid]
        logger.info(f"  - {quest.title} ({quest.category})")

    logger.info("\nРекомендации алгоритмов:")

    for algo in algorithms:
        recommendations, _ = algo.recommend_quests(user_id, top_k=10)

        logger.info(f"\n{algo.name}:")

        # Находим пересечение с ground truth
        rec_ids = [r["quest_id"] for r in recommendations]
        hits = set(rec_ids) & set(expected)

        logger.info(f"  Совпадений с ground truth: {len(hits)}/{len(expected)}")

        for i, rec in enumerate(recommendations[:5], 1):
            quest = quests[rec["quest_id"]]
            is_hit = "✓" if rec["quest_id"] in hits else " "
            logger.info(f"  {i}. {is_hit} {quest.title[:40]}... (score: {rec['score']:.3f}, {quest.category})")

        if len(hits) > 0:
            logger.info(f"  Точные совпадения:")
            for qid in hits:
                quest = quests[qid]
                pos = rec_ids.index(qid) + 1 if qid in rec_ids else "не в топ-10"
                logger.info(f"    - {quest.title} (позиция: {pos})")


# Обновляем main для запуска нового бенчмарка
if __name__ == "__main__":
    # Запуск реалистичного бенчмарка
    logger.info("\n" + "=" * 60)
    logger.info("ЗАПУСК РЕАЛИСТИЧНОГО ТЕСТИРОВАНИЯ")
    logger.info("=" * 60)

    results = run_realistic_benchmark()

    # Детальный анализ для ключевых пользователей
    quests = RealisticTestDataGenerator.generate_quests()
    users = RealisticTestDataGenerator.generate_realistic_users(quests)
    ground_truth = RealisticTestDataGenerator.create_deterministic_ground_truth(users, quests)

    algorithms = [
        ContentBasedBERT(),
        CollaborativeFilteringKNN(n_neighbors=3),
        # HybridRecommender(content_weight=0.7, collaborative_weight=0.3),
        SimpleTfidfRecommender()
    ]

    for algo in algorithms:
        algo.load_data(quests, users)

    # Анализ для программиста-спортсмена и креативного дизайнера
    analyze_specific_user(algorithms, 1, quests, users, ground_truth)  # Программист-спортсмен
    analyze_specific_user(algorithms, 2, quests, users, ground_truth)  # Креативный дизайнер

    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    logger.info("=" * 60)