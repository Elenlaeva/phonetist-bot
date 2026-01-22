import os              # для работы с файлами
import pickle          # для сериализации данных (сохранение пайтон-объектов в файл)
import numpy as np     # работа с массивами и векторами

# Загружаю переменные окружения из файла .env
from dotenv import load_dotenv
load_dotenv()

# Импорт клиента OpenAI
from openai import OpenAI

# Создаю клиента OpenAI, используя ключ и базовый URL из .env
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# Папка, где лежат маркдауны с теорией
TICKETS_DIR = "tickets"

# Список для хранения текстов документов
documents = []

# Список для хранения названий терминов (имена файлов)
metainfo = []

# Прохожусь по всем файлам в папке tickets
for filename in os.listdir(TICKETS_DIR):

    # На всякий случай отсею все немаркдауны
    if not filename.endswith(".md"):
        continue

    # Формирую полный путь к файлу
    path = os.path.join(TICKETS_DIR, filename)

    # Открываю файл в кодировке UTF-8
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Получаю название термина из имени файла
    term_name = filename.replace(".md", "")

    # Добавляю название термина в начало текста
    # Это по идее должно усилить семантическую связь между запросом и документом
    full_text = f"Термин: {term_name}\n\n{text}"

    # Сохраняю текст документа
    documents.append(full_text)

    # Сохраняю название термина отдельно (для отладки и пояснений)
    metainfo.append(term_name)

# Вывожу количество документов (для контроля)
print(f"Документов: {len(documents)}")

# Список для хранения эмбеддингов
embeddings = []

# Для каждого документа создаю векторное представление aka эмбеддинг
for doc in documents:
    emb = client.embeddings.create(
        model="text-embedding-3-small",  # компактная и быстрая модель
        input=doc
    )

    # Беру сам вектор из ответа API
    embeddings.append(emb.data[0].embedding)

# Преобразую список в numpy-массив
embeddings = np.array(embeddings)

# Сохраняю документы, эмбеддинги и метаданные в файл index.pkl
with open("index.pkl", "wb") as f:
    pickle.dump(
        {
            "documents": documents,
            "embeddings": embeddings,
            "metainfo": metainfo
        },
        f
    )

# Сообщаю себе в будущем через терминал об успешном завершении чтоб не нервничать 
print("Индекс успешно сохранён!")
