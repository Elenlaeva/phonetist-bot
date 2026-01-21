import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv
from bpe_index import build_index, search_article
from llm_module import generate_answer

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# Строим BPE и индекс базы знаний
bpe, index = build_index()

@dp.message_handler()
async def answer_question(message: types.Message):
    query = message.text.strip()
    if not query:
        await message.answer("Пожалуйста, введите ваш вопрос.")
        return

    # 1. Поиск релевантной статьи
    article = search_article(query, bpe, index)

    # 2. Генерация ответа через LLM
    answer = generate_answer(query, article)
    await message.answer(answer)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)

