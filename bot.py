# Импорт стандартных библиотек
import os
import pickle
import numpy as np
import asyncio

# Загружаю переменные окружения
from dotenv import load_dotenv
load_dotenv()

# Импорт библиотек для бота
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

# Импорт клиента OpenAI
from openai import OpenAI

# Читаю токены и ключи из .env
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Создаю объект бота
bot = Bot(token=BOT_TOKEN)

# Dispatcher управляет обработкой сообщений
dp = Dispatcher()

# Создаю клиента OpenAI
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

# Загружаю индекс из файла
with open("index.pkl", "rb") as f:
    data = pickle.load(f)

# Достаю документы, эмбеддинги и названия терминов
documents = data["documents"]
embeddings = np.array(data["embeddings"])
metainfo = data["metainfo"]

# Функция вычисления косинусного сходства между двумя векторами
# Она показывает, насколько тексты похожи по смыслу
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Функция поиска наиболее релевантных документов
def search_docs(query, top_k=5):

    # Создаю эмбеддинг для запроса пользователя
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Считаю сходство запроса с каждым документом
    scores = [cosine_similarity(q_emb, emb) for emb in embeddings]

    # Получаю индексы top_k самых релевантных документов
    top_indices = np.argsort(scores)[-top_k:][::-1]

    # Возвращаю тексты документов и их названия
    return [(documents[i], metainfo[i]) for i in top_indices]

# Обработка команды /start
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "Привет! Я бот Фонетист.\n"
        "Задай вопрос по терминам из экзаменационных билетов."
    )

# Обработка всех остальных сообщений
@dp.message()
async def answer(message: types.Message):

    # Получаю текст запроса пользователя
    query = message.text.strip()

    # Ищу релевантные документы
    docs = search_docs(query, top_k=5)

    # Формирую текст контекста для GPT
    context_blocks = []
    for text, title in docs:
        context_blocks.append(f"Источник: {title}\n{text}")

    # Объединяю найденные документы в один контекст
    context = "\n\n".join(context_blocks)

    try:
        # Отправляю запрос в GPT с контекстом
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты помощник по фонетике. "
                        "Используй ТОЛЬКО предоставленный контекст. "
                        "Если термин описан — дай определение. "
                        "Если термина нет в базе — прямо скажи об этом. "
                        "Отвечай кратко, по сути, списком. "
                        "Не используй Markdown. "
                        "Примеры используй только если их явно просят."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {query}\n\n"
                        f"Контекст:\n{context}"
                    )
                }
            ]
        )

        # Получаю текст ответа
        answer_text = response.choices[0].message.content

        # Отправляю ответ пользователю (ограничение телеги — 4000 символов)
        await message.answer(answer_text[:4000])

    except Exception as e:
        # Обрабатываю возможные ошибки
        await message.answer(f"Ошибка при обработке запроса: {e}")

# Точка входа в программу
if __name__ == "__main__":
    # Запускаю бота в асинхронном режиме
    asyncio.run(dp.start_polling(bot))
