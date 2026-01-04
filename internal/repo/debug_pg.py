import psycopg2

PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'becomeoverman',
    'user': 'postgres',
    'password': 'postgres'
}

conn = psycopg2.connect(**PG_CONFIG)
cursor = conn.cursor()

print("=== КВЕСТЫ В POSTGRESQL ===")
cursor.execute("SELECT id, title FROM quests ORDER BY id")
quests = cursor.fetchall()
print(f"Всего квестов: {len(quests)}")
for id, title in quests:
    print(f"  ID={id}: {title}")

print("\n=== ПОЛЬЗОВАТЕЛИ И ИХ КВЕСТЫ ===")
cursor.execute("""
    SELECT uq.user_id, uq.quest_id, q.title
    FROM user_quests uq
    LEFT JOIN quests q ON uq.quest_id = q.id
    WHERE uq.user_id = 22
    ORDER BY uq.user_id, uq.quest_id
""")
user_quests = cursor.fetchall()
print(f"Всего записей для пользователя 22: {len(user_quests)}")
for user_id, quest_id, title in user_quests:
    print(f"  User {user_id} -> Quest {quest_id}: {title}")

print("\n=== ПРОВЕРКА КВЕСТА ID=2 ===")
cursor.execute("SELECT id, title, description FROM quests WHERE id = 2")
quest_2 = cursor.fetchone()
if quest_2:
    print(f"Квест найден: ID={quest_2[0]}, Title={quest_2[1]}")
    print(f"Описание: {quest_2[2][:100]}...")
else:
    print("Квест с ID=2 не найден в PostgreSQL!")

cursor.close()
conn.close()