"""API-слой (Presentation) FastAPI-приложения.

Модули:
  - routes: главный роутер, связывающий HTTP-эндпоинты с бизнес-логикой.
            Отвечает за отдачу HTML-страницы (index), а также JSON-данных
            для дашборда (/dashboard), прогнозов (/predict, /predict_all),
            метрик (/metrics), версий (/versions) и бэктеста (/backtest).
"""

# Явно экспортируем роутер для удобства (например, для main.py)
from app.api.routes import router

__all__ = ["router"]