# quest_semantic_search.py

from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Dict, Tuple

# ----------------------------
# 1. Тестовые данные
# ----------------------------
def load_test_quests() -> List[Dict]:
    return [
        {
            "id": 1,
            "title": "30 дней дисциплины",
            "description": "Ежедневные задания на развитие самодисциплины и силы воли",
            "category": "willpower"
        },
        {
            "id": 2,
            "title": "Утренние ритуалы",
            "description": "Формирование полезных утренних привычек и осознанного начала дня",
            "category": "habits"
        },
        {
            "id": 3,
            "title": "Физическая перезагрузка",
            "description": "Тренировки, бег, зарядка и физическая активность каждый день",
            "category": "health"
        },
        {
            "id": 4,
            "title": "Глубокая концентрация",
            "description": "Развитие фокуса внимания и способности работать без отвлечений",
            "category": "focus"
        },
        {
            "id": 5,
            "title": "Социальная прокачка",
            "description": "Задания на уверенность, харизму и общение с людьми",
            "category": "social"
        },
        {
            "id": 6,
            "title": "Сила воли и контроль",
            "description": "Упражнения для контроля импульсов и укрепления силы воли",
            "category": "willpower"
        },
    ]


# ----------------------------
# 2. Класс семантического поисковика
# ----------------------------
class SemanticQuestSearcher:
    def __init__(self, quests: List[Dict], model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.quests = quests
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self._prepare_embeddings()
        print(f"🔍 Sentence-BERT индекс построен для {len(quests)} квестов (device={self.device})")

    def _prepare_embeddings(self):
        texts = [q['title'] + ". " + q['description'] for q in self.quests]
        # вычисляем эмбеддинги (с конвертацией в torch tensor)
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = torch.topk(cosine_scores, k=min(top_k, len(self.quests)))
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            quest = self.quests[idx]
            results.append((quest, float(score)))
        return results


# ----------------------------
# 3. Тестовый запуск
# ----------------------------
if __name__ == "__main__":
    quests = load_test_quests()
    searcher = SemanticQuestSearcher(quests)

    test_queries = [
        "дисциплина",
        "бегать",
        "утренние привычки",
        "концентрация внимания",
        "харизма и общение",
        "контроль своих импульсов",
        "хочу научиться находить друзей",
        "познакомиться с девушкой",
        "стать более спортивным и активным",
        "стать умнее",
        "хочу стать лучше в программировании",
        "устал на работе - у меня выгорание"
    ]

    for query in test_queries:
        print(f"\n🔎 Запрос: '{query}'")
        results = searcher.search(query, top_k=3)
        for quest, score in results:
            print(f"  • {quest['title']} (score={round(score, 3)})")
