"""Crypto Forecast App — прогнозирование курсов криптовалют с помощью ML.

Архитектура: трёхуровневый монолит (C4, уровень контейнеров C2/C3).

Слои:
  1. Presentation (app/api):
     - FastAPI (routes.py), Jinja2 (templates/index.html), 
       Lightweight Charts (static/js/app.js), CSS (static/css/style.css).
  
  2. Business (app/services):
     - orchestrator.py: координатор (выбор модели, сборка ответа, дашборд)
     - feature_engineer.py: сборка 51 признака + winsorization
     - predictor.py: фасад ModelPredictor (version cache)
     - att_gru_predictor.py / arimax_predictor.py: инференс моделей
     - aci.py: Adaptive Conformal Inference (онлайн-калибровка интервалов)

  3. Infrastructure (app/services):
     - database_repository.py: асинхронная БД (SQLite / PostgreSQL)
     - data_collector.py: Binance OHLCV
     - external_data_collector.py: Blockchain.com, Alternative.me, FRED
     - cache_manager.py: двухуровневый кэш (память + диск)
     - scheduler.py: фоновые задачи (klines 4h, external 03:00, retrain Sun 02:00)

Ядро: app/core/config.py (Settings, pydantic-settings).
Точка входа: main.py (uvicorn main:app, lifespan с scheduler).
"""