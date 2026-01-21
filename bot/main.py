from aiogram import Bot, Dispatcher, executor, types
from config import BOT_TOKEN
from rag import load_documents, find_relevant_text

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

documents = load_documents()


@dp.message_handler()
async def handle_message(message: types.Message):
    user_question = message.text

    answer = find_relevant_text(user_question, documents)

    await message.answer(answer)


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)


from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from bpe_index import build_index, search_article
import os
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# === Строим BPE и индекс ===
bpe, index = build_index()

@dp.message_handler()
async def answer_question(message: types.Message):
    query = message.text
    article = search_article(query, bpe, index)
    await message.answer(article)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)

