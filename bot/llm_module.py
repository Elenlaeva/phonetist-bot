import os
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def generate_answer(question, context):
    """
    Генерация развернутого академического ответа с использованием LLM
    question: строка с вопросом пользователя
    context: текст статьи из базы знаний
    """
    prompt = f"""
Ты консультант для студентов по фонетике.
Используй только предоставленный контекст из курса по сегментным и супрасегментным единицам.
Контекст:
{context}

Вопрос:
{question}

Ответ дай подробно, в разговорном тоне, с примерами и пояснениями.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']
