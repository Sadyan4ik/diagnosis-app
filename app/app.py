from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.templating import Jinja2Templates
from services import DiagnosisService
import torch
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Список диагнозов (тот же порядок, что и при обучении)
DIAG_LIST = [
    'Псориаз', 'Варикозное расширение вен', 'Брюшной тиф', 'Ветряная оспа',
    'Импетиго', 'Лихорадка денге', 'Грибковая инфекция', 'Обычная простуда',
    'Пневмония', 'Диморфный геморрой', 'Артрит', 'Угревая сыпь',
    'Бронхиальная астма', 'Гипертония', 'Мигрень', 'Шейный спондилез',
    'Желтуха', 'Малярия', 'Инфекция мочевыводящих путей', 'Аллергия',
    'Гастроэзофагеальная рефлюксная болезнь', 'Лекарственная реакция',
    'Язвенная болезнь', 'Сахарный диабет'
]

# Инициализируем сервис один раз при запуске
service = DiagnosisService(
    model_path='model_Symptoms2Disease_gru.pth',
    vocab_path='vocab.pkl',
    diagnoses=DIAG_LIST,
    max_words=32,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Prometheus определяет приложение FastAPI
@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, symptoms: str = Form(...)):
    # Вся логика теперь внутри метода сервиса
    results = service.predict_top3(symptoms)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "user_input": symptoms
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)