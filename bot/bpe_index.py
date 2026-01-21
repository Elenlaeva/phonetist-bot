import os
from collections import Counter

class BytePairEncoding:
    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def get_pairs(self, word_tokens):
        pairs = Counter()
        for token_list in word_tokens:
            for i in range(len(token_list) - 1):
                pairs[(token_list[i], token_list[i + 1])] += 1
        return pairs

    def merge_pair(self, pair, word_tokens):
        first, second = pair
        new_tokens = []
        for token_list in word_tokens:
            merged = []
            i = 0
            while i < len(token_list):
                if i < len(token_list) - 1 and token_list[i] == first and token_list[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(token_list[i])
                    i += 1
            new_tokens.append(merged)
        return new_tokens

    def fit(self, text: str):
        text = text.lower()
        words = text.strip().split()
        word_tokens = [[c for c in w] + ['</w>'] for w in words]

        while True:
            pairs = self.get_pairs(word_tokens)
            if not pairs:
                break
            most_common = pairs.most_common(1)[0][0]
            word_tokens = self.merge_pair(most_common, word_tokens)
            self.merges.append(most_common)
            self.vocab[''.join(most_common)] = len(self.vocab)
            if len(self.vocab) >= self.vocab_size:
                break

        self.word_tokens = word_tokens

    def encode(self, word: str):
        tokens = [c for c in word.lower()] + ['</w>']
        for first, second in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

# === Индексация базы знаний ===
def build_index(knowledge_dir='knowledge_base', bpe_vocab_size=50):
    bpe = BytePairEncoding(vocab_size=bpe_vocab_size)
    corpus_text = ""

    # 1. Собираем весь текст из базы
    for fname in os.listdir(knowledge_dir):
        if fname.endswith('.md'):
            with open(os.path.join(knowledge_dir, fname), 'r', encoding='utf-8') as f:
                corpus_text += f.read() + " "

    # 2. Обучаем BPE
    bpe.fit(corpus_text)

    # 3. Создаем индекс: словарь "токен -> список файлов, где встречается"
    index = {}
    for fname in os.listdir(knowledge_dir):
        if fname.endswith('.md'):
            with open(os.path.join(knowledge_dir, fname), 'r', encoding='utf-8') as f:
                text = f.read()
            words = text.lower().split()
            for word in words:
                tokens = bpe.encode(word)
                for token in tokens:
                    if token not in index:
                        index[token] = set()
                    index[token].add(fname)

    return bpe, index

# === Пример поиска статьи по вопросу ===
def search_article(query, bpe, index, knowledge_dir='knowledge_base'):
    query_tokens = []
    for word in query.lower().split():
        query_tokens += bpe.encode(word)

    file_scores = Counter()
    for token in query_tokens:
        if token in index:
            for fname in index[token]:
                file_scores[fname] += 1

    # Выбираем файл с максимальным совпадением
    if file_scores:
        best_file = file_scores.most_common(1)[0][0]
        with open(os.path.join(knowledge_dir, best_file), 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Извините, статья не найдена."
