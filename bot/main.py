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
