import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Загрузка ресурсов при первом запуске
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def clean_text(sent):
    # Удаление пунктуации и приведение к нижнему регистру
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    words = word_tokenize(sent)
    words = [word for word in words if word not in stop_words]

    return words # Возвращаем список слов для удобства индексации