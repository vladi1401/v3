# pyrobot/news_filter.py

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# TODO: Implementar lógica de scraping o API (ej. DailyFX, ForexFactory)
# Esta función debería ser llamada una vez al día para cachear los eventos.
def fetch_high_impact_news_times() -> dict:
    """
    Simulador (STUB) para descargar noticias de alto impacto.
    
    Debería devolver un diccionario como:
    {
        "EUR": [
            ("2025-10-30T08:00:00Z", "2025-10-30T09:00:00Z"), # Rango de embargo
            ("2025-10-30T12:30:00Z", "2025-10-30T13:30:00Z")
        ],
        "USD": [
            ("2025-10-29T18:00:00Z", "2025-10-29T19:00:00Z") # FOMC
        ]
    }
    """
    logger.debug("News filter STUB: No se están cargando noticias reales.")
    return {} # Devolver vacío = sin embargos

# Cachear noticias (en un bot real, esto se recargaría cada 24h)
CACHED_NEWS_TIMES = fetch_high_impact_news_times()


def is_news_embargo_active(ticker: str, current_time_utc: datetime) -> bool:
    """
    Verifica si hay un embargo de noticias activo para el ticker dado.
    """
    # Lógica simple de mapeo de ticker a divisas
    currencies_in_ticker = []
    if "EUR" in ticker: currencies_in_ticker.append("EUR")
    if "USD" in ticker: currencies_in_ticker.append("USD")
    if "GBP" in ticker: currencies_in_ticker.append("GBP")
    # ... añadir más ...
    
    if not currencies_in_ticker:
        return False # No se reconoce el ticker, operar con precaución

    for currency in currencies_in_ticker:
        if currency in CACHED_NEWS_TIMES:
            for embargo_start, embargo_end in CACHED_NEWS_TIMES[currency]:
                # Asumir que los tiempos están en UTC (formato ISO)
                start_dt = datetime.fromisoformat(embargo_start)
                end_dt = datetime.fromisoformat(embargo_end)
                
                if start_dt <= current_time_utc <= end_dt:
                    logger.warning(f"¡EMBARGO DE NOTICIAS ACTIVO! Evento de {currency} en curso. Trading pausado.")
                    return True
                    
    return False # No hay embargo activo