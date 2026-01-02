# test_bert_api.py
import requests
import json
import time

# Базовый URL API
BASE_URL = "http://localhost:8000"

# Ваши квесты в формате JSON
quests_data = [
    {
        "id": 1,
        "title": "Утренний дружеский марафон",
        "description": "Совместный недельный челлендж для развития силы воли и здоровья",
        "category": "health"
    },
    {
        "id": 2,
        "title": "Основы продуктивности",
        "description": "Недельный план для формирования полезных привычек",
        "category": "willpower"
    },
    {
        "id": 3,
        "title": "Творческий дуэт",
        "description": "Совместное создание творческого проекта за неделю",
        "category": "charisma"
    },
    {
        "id": 4,
        "title": "Путь к гармонии",
        "description": "14-дневный путь к внутреннему балансу и осознанности",
        "category": "mental_health"
    },
    {
        "id": 5,
        "title": "Кулинарный дуэт",
        "description": "Неделя совместного кулинарного мастерства",
        "category": "charisma"
    },
    {
        "id": 6,
        "title": "Фитнес-марафон",
        "description": "21 день для формирования спортивной привычки",
        "category": "health"
    },
    {
        "id": 7,
        "title": "Утренний старт и вечерний баланс",
        "description": "Квест для формирования утренней привычки и вечернего физического здоровья без нагрузки на колени. Поможет преодолеть лень и начать день активно.",
        "category": "willpower"
    },
    {
        "id": 8,
        "title": "Путь Богатыря",
        "description": "Стань сильнее, выносливее и мощнее за неделю интенсивных тренировок. Выполни все задачи и почувствуй прилив энергии!",
        "category": "health"
    },
    {
        "id": 9,
        "title": "Король и Шут: Гитарный фестиваль",
        "description": "Освой игру на гитаре легендарных песен 'Король и Шут' и покори фестиваль 20 мая. Пройди путь от базовых аккордов до полноценного сета из трех хитов.",
        "category": "creativity"
    },
    {
        "id": 10,
        "title": "The Friendship Forge",
        "description": "Embark on a journey to build meaningful connections and overcome social isolation. This quest will guide you through self-discovery, finding your tribe, and taking courageous steps to create lasting friendships.",
        "category": "social"
    }
]


def test_api():
    print("🧪 Тестируем BERT API\n")

    # 1. Проверяем health
    print("1️⃣ Проверяем здоровье API...")
    try:
        health_response = requests.get(f"{BASE_URL}/api/health")
        if health_response.status_code == 200:
            print(f"   ✅ API работает: {health_response.json()}")
        else:
            print(f"   ❌ Ошибка: {health_response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Не могу подключиться к API: {e}")
        print("   Убедитесь, что сервер запущен на localhost:8000")
        return

    # 2. Добавляем квесты
    print("\n2️⃣ Добавляем квесты в индекс...")
    add_response = requests.post(
        f"{BASE_URL}/api/quests/add",
        json={"quests": quests_data}
    )

    if add_response.status_code == 200:
        print(f"   ✅ Добавлено квестов: {add_response.json()}")
    else:
        print(f"   ❌ Ошибка добавления: {add_response.status_code}")
        print(f"   Ответ: {add_response.text}")
        return

    # 3. Проверяем статистику
    print("\n3️⃣ Проверяем статистику...")
    stats_response = requests.get(f"{BASE_URL}/api/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"   ✅ Квестов в индексе: {stats['quests_count']}")
        print(f"   Размерность эмбеддингов: {stats['embedding_dimension']}")
    else:
        print(f"   ❌ Ошибка статистики: {stats_response.status_code}")

    # 4. Тестируем поиск
    print("\n4️⃣ Тестируем поиск...")
    test_queries = [
        "бегать и тренироваться",
        "утренние привычки",
        "творчество и музыка",
        "общение с друзьями",
        "кулинария готовка",
        "продуктивность работа",
        "гитара песни",
        "спорт здоровье"
    ]

    for query in test_queries:
        print(f"\n   🔎 Запрос: '{query}'")

        search_data = {
            "query": query,
            "top_k": 3,
            "category": None  # Можно указать "health", "willpower" и т.д.
        }

        search_response = requests.post(
            f"{BASE_URL}/api/search",
            json=search_data
        )

        if search_response.status_code == 200:
            results = search_response.json()
            print(f"   ⏱️ Время поиска: {results['search_time_ms']} мс")

            for i, result in enumerate(results['results'], 1):
                print(f"   {i}. {result['title']}")
                print(f"      Категория: {result.get('category', 'N/A')}")
                print(f"      Схожесть: {result['similarity_score']:.3f}")
        else:
            print(f"   ❌ Ошибка поиска: {search_response.status_code}")
            print(f"   Ответ: {search_response.text}")

    # 5. Тестируем поиск похожих квестов
    print("\n5️⃣ Тестируем поиск похожих квестов...")

    # Ищем похожие на квест про здоровье (id=1)
    similar_data = {
        "quest_id": 1,  # Утренний дружеский марафон
        "top_k": 3
    }

    similar_response = requests.post(
        f"{BASE_URL}/api/similar",
        json=similar_data
    )

    if similar_response.status_code == 200:
        results = similar_response.json()
        print(f"   🔍 Похожие на 'Утренний дружеский марафон':")
        for i, quest in enumerate(results.get('similar_quests', []), 1):
            print(f"   {i}. {quest['title']} (схожесть: {quest['similarity_score']:.3f})")
    else:
        print(f"   ❌ Ошибка поиска похожих: {similar_response.status_code}")

    # 6. Тестируем поиск с фильтром по категории
    print("\n6️⃣ Тестируем поиск с фильтром категории...")

    search_with_filter = {
        "query": "тренировки спорт",
        "top_k": 5,
        "category": "health"  # Только квесты категории health
    }

    filter_response = requests.post(
        f"{BASE_URL}/api/search",
        json=search_with_filter
    )

    if filter_response.status_code == 200:
        results = filter_response.json()
        print(f"   🔍 Запрос: 'тренировки спорт' (только категория health)")
        print(f"   Найдено результатов: {len(results['results'])}")
        for i, result in enumerate(results['results'], 1):
            print(f"   {i}. {result['title']} (категория: {result.get('category', 'N/A')})")
    else:
        print(f"   ❌ Ошибка поиска с фильтром: {filter_response.status_code}")

    print("\n" + "=" * 50)
    print("✅ Тестирование завершено!")


if __name__ == "__main__":
    test_api()