"""
Конфигурация приложения.

Читает переменные окружения из .env и предоставляет типизированный
объект Settings, который импортируют остальные модули.
"""
from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения, читаемые из .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Пути ---
    data_dir: Path = Path("data")
    models_dir: Path = Path("data/models/deploy")
    cache_dir: Path = Path("data/cache")
    db_path: Path = Path("data/crypto.db")
    db_url: str = "sqlite:///data/crypto.db"

    # --- API ---
    binance_api_url: str = "https://api.binance.com"
    coingecko_api_url: str = "https://api.coingecko.com/api/v3"
    fred_api_key: str = ""
    blockchain_api_url: str = "https://api.blockchain.info"
    alternative_api_url: str = "https://api.alternative.me"
        
    # --- Кэш ---
    cache_ttl_seconds: int = 60

    # --- Модели: общее ---
    active_models: List[str] = Field(default_factory=lambda: ["attention_gru"])
    default_model: str = "attention_gru"
    model_selector: dict = Field(default_factory=dict)

    # --- Attention-GRU ---
    gru_model_dir: Path = Path("data/models/attention_gru")
    gru_model_file: str = "attention_gru.keras"
    gru_scaler_x_file: str = "scaler_X.pkl"
    gru_scaler_y_file: str = "scaler_y.pkl"
    
    gru_config_file: str = "model_config.json"

    # --- ARIMAX-LSTM ---
    arimax_model_dir: Path = Path("data/models/arimax_lstm")
    arimax_models_subdir: str = "models_per_h"
    arimax_state_file: str = "arima_result.pkl"
    arimax_config_file: str = "config.json"
    

    # --- Модели: параметры ---
    aci_alpha_target: float = 0.05
    aci_gamma: float = 0.01
    aci_lookback: int = 100
    aci_alpha_min: float = 0.01
    aci_alpha_max: float = 0.5
    lookback: int = 30

    @field_validator("active_models", mode="before")
    @classmethod
    def _split_models(cls, v):
        """Разрешает 'gru,arimax' → ['gru', 'arimax']."""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("model_selector", mode="before")
    @classmethod
    def _parse_selector(cls, v):
        """Разрешает строку JSON → dict."""
        if isinstance(v, str) and v.strip():
            import json
            return json.loads(v)
        return v or {}

    def ensure_dirs(self) -> None:
        """Создаёт директории, если их нет."""
        for d in (self.data_dir, self.models_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()