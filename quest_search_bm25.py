# quest_search.py

import re
import numpy as np
from rank_bm25 import BM25Okapi


# ----------------------------
# 1. Токенизация
# ----------------------------
def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    return text.split()


# ----------------------------
# 2. Поисковик BM25
# ----------------------------
class QuestSearchEngine:
    def __init__(self, quests: list[dict]):
        """
        quests: список словарей с ключами:
        id, title, description, category, difficulty
        """
        self.quests = quests

        # Собираем текст каждого квеста в один документ
        self.corpus = [
            tokenize(
                q["title"] + " " +
                q["description"] + " " +
                q["category"]
            )
            for q in quests
        ]

        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 5):
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                quest = self.quests[idx]
                results.append({
                    "id": quest["id"],
                    "title": quest["title"],
                    "score": round(float(scores[idx]), 3)
                })

        return results


# ----------------------------
# 3. Тестовые данные
# ----------------------------
def load_test_quests():
    return [
        {
            "id": 1,
            "title": "30 дней дисциплины",
            "description": "Ежедневные задания для развития самодисциплины и силы воли",
            "category": "willpower",
            "difficulty": 3
        },
        {
            "id": 2,
            "title": "Утренние ритуалы",
            "description": "Формирование полезных привычек и правильного утра",
            "category": "mental_health",
            "difficulty": 2
        },
        {
            "id": 3,
            "title": "Физическая перезагрузка",
            "description": "Тренировки, бег и физическая активность каждый день",
            "category": "health",
            "difficulty": 4
        },
        {
            "id": 4,
            "title": "Глубокая концентрация",
            "description": "Упражнения на фокус внимания и интеллектуальную работу",
            "category": "intelligence",
            "difficulty": 3
        },
        {
            "id": 5,
            "title": "Социальная прокачка",
            "description": "Задания на уверенность, харизму и общение с людьми",
            "category": "charisma",
            "difficulty": 2
        },
    ]


# ----------------------------
# 4. Простой интерактивный тест
# ----------------------------
if __name__ == "__main__":
    quests = load_test_quests()
    engine = QuestSearchEngine(quests)

    print("🔍 Поиск квестов (BM25)")
    print("Введите запрос (exit для выхода)\n")

    while True:
        query = input("> ")
        if query.lower() in ("exit", "quit"):
            break

        results = engine.search(query)

        if not results:
            print("Ничего не найдено\n")
            continue

        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']} (score={r['score']})")
        print()
