# pyrobot/alerts.py

import logging

logger = logging.getLogger(__name__)

def send_telegram_alert(message: str, level: str = "info"):
    """
    Simulador (STUB) para enviar alertas a Telegram.
    Actualmente solo loggea el mensaje.
    
    Puedes reemplazar esto con una llamada real a la API de Telegram:
    
    import requests
    TOKEN = "TU_TOKEN_DE_BOT"
    CHAT_ID = "TU_CHAT_ID"
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        logger.error(f"Fallo al enviar alerta de Telegram: {e}")
    """
    
    log_message = f"[ALERTA_STUB] {message}"
    
    if level == "critical":
        logger.critical(log_message)
    elif level == "error":
        logger.error(log_message)
    elif level == "warning":
        logger.warning(log_message)
    else:
        logger.info(log_message)