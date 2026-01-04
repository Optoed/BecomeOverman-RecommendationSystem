from typing import List


def _get_recommendation_explanation(recommended_quest, user_quest_ids, quests_data):
    """Генерирует текстовое объяснение, почему квест рекомендован"""

    recommended_cat = recommended_quest.get('category')

    # Считаем, сколько у пользователя квестов в этой категории
    user_cats = []
    for quest_id in user_quest_ids:
        quest = quests_data.get(quest_id, {})
        if cat := quest.get('category'):
            user_cats.append(cat)

    from collections import Counter
    cat_counter = Counter(user_cats)

    # Если у пользователя есть квесты в этой категории
    if recommended_cat in cat_counter:
        count = cat_counter[recommended_cat]
        total = len(user_quest_ids)
        percent = int((count / total) * 100) if total > 0 else 0

        return f"Вам нравятся квесты категории '{recommended_cat}' ({percent}% ваших квестов)"

    # Если категория новая
    elif user_quest_ids:
        most_common_cat, _ = cat_counter.most_common(1)[0]
        return f"Похоже на ваши квесты категории '{most_common_cat}', но в новой категории '{recommended_cat}'"

    return "Попробуйте что-то новое!"


def _get_common_quests(user_id1: int, user_id2: int, users_data: dict) -> List[int]:
    """Получить общие квесты двух пользователей"""
    quests1 = set(users_data.get(user_id1, {}).get("quest_ids", []))
    quests2 = set(users_data.get(user_id2, {}).get("quest_ids", []))
    return list(quests1.intersection(quests2))