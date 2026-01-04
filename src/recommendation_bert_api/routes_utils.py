from typing import List, Dict, Any


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


def _get_user_recommendation_explanation(
        cur_user_id: int,
        cur_user_quests: List[int],
        other_user_id: int,
        other_user_quests: List[int],
        similarity_score: float,
        quests_data: Dict[int, Dict]
) -> Dict[str, Any]:
    """Генерирует объяснение рекомендации пользователя"""

    # 1. Находим общие квесты
    common_quests = set(cur_user_quests) & set(other_user_quests)

    # 2. Анализируем категории квестов
    cur_categories = {}
    other_categories = {}
    common_categories = set()

    for quest_id in cur_user_quests:
        quest = quests_data.get(quest_id)
        if quest and quest.get('category'):
            category = quest['category']
            cur_categories[category] = cur_categories.get(category, 0) + 1

    for quest_id in other_user_quests:
        quest = quests_data.get(quest_id)
        if quest and quest.get('category'):
            category = quest['category']
            other_categories[category] = other_categories.get(category, 0) + 1
            if category in cur_categories:
                common_categories.add(category)

    # 3. Определяем самые популярные категории
    top_cur_categories = sorted(cur_categories.items(), key=lambda x: x[1], reverse=True)[:3]
    top_other_categories = sorted(other_categories.items(), key=lambda x: x[1], reverse=True)[:3]

    # 4. Генерируем объяснение на основе анализа
    explanation_parts = []

    # По схожести профилей
    if similarity_score > 0.7:
        similarity_text = "очень высокая"
    elif similarity_score > 0.5:
        similarity_text = "высокая"
    elif similarity_score > 0.3:
        similarity_text = "средняя"
    else:
        similarity_text = "умеренная"

    explanation_parts.append(f"Схожесть интересов: {similarity_text} ({similarity_score:.2%})")

    # По общим квестам
    if common_quests:
        quest_count = len(common_quests)
        quest_titles = []
        for quest_id in list(common_quests)[:3]:  # Берем до 3 квестов для примера
            quest = quests_data.get(quest_id, {})
            if quest.get('title'):
                quest_titles.append(quest['title'][:30] + "..." if len(quest['title']) > 30 else quest['title'])

        if quest_count == 1:
            explanation_parts.append(f"У вас 1 общий квест: {quest_titles[0]}")
        else:
            quests_text = f"{quest_count} общих квестов"
            if quest_titles:
                quests_text += f", включая: {', '.join(quest_titles)}"
            explanation_parts.append(quests_text)

    # По общим категориям
    if common_categories:
        categories_text = f"Общие интересы: {', '.join(list(common_categories)[:3])}"
        explanation_parts.append(categories_text)

    # По дополняющим категориям (если нет общих)
    elif top_cur_categories and top_other_categories:
        # Находим самые частые категории каждого пользователя
        cur_top = top_cur_categories[0][0] if top_cur_categories else ""
        other_top = top_other_categories[0][0] if top_other_categories else ""

        if cur_top and other_top and cur_top != other_top:
            explanation_parts.append(f"Разные, но дополняющие интересы: {cur_top} и {other_top}")

    # Статистика по квестам
    if cur_user_quests and other_user_quests:
        quest_ratio = len(other_user_quests) / len(cur_user_quests) if cur_user_quests else 0

        if quest_ratio > 1.5:
            explanation_parts.append(
                f"У пользователя на {int((quest_ratio - 1) * 100)}% больше квестов - может поделиться опытом")
        elif quest_ratio < 0.67:
            explanation_parts.append(f"У пользователя меньше квестов - может быть заинтересован в ваших рекомендациях")

    # Формируем итоговое объяснение
    explanation = {
        "summary": " | ".join(explanation_parts[:3]),  # Краткое описание
        "details": {
            "common_quests_count": len(common_quests),
            "common_quests_ids": list(common_quests)[:10],  # Ограничиваем для размера ответа
            "common_categories": list(common_categories),
            "user_categories_top": [cat for cat, _ in top_cur_categories],
            "other_user_categories_top": [cat for cat, _ in top_other_categories],
            "user_quests_count": len(cur_user_quests),
            "other_user_quests_count": len(other_user_quests),
            "similarity_level": similarity_text
        }
    }

    return explanation