FROM python:3.10-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала копируем зависимости для кэширования слоев
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Загружаем данные NLTK (пунктуация и стоп-слова)
RUN python -m nltk.downloader punkt stopwords

# Копируем остальные файлы приложения
COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]