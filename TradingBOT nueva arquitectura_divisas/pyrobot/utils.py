# pyrobot/utils.py

import logging
import time
from logging.handlers import RotatingFileHandler
from functools import wraps
# Asegúrate de que este import sea relativo si utils.py está en el mismo directorio que exceptions.py
from .exceptions import ConnectionError

def setup_logging():
    """Configura el logging para consola y archivo rotativo."""
    
    # Silenciar otros loggers (ej. 'urllib3')
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger = logging.getLogger() # Logger raíz
    logger.setLevel(logging.INFO)
    
    # Evitar añadir handlers duplicados si se llama de nuevo
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formato
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler para consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Handler para archivo (rota a los 5MB, mantiene 3 backups)
    try:
        fh = RotatingFileHandler('bot.log', maxBytes=5*1024*1024, backupCount=3)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except PermissionError:
        print("Error de permisos: No se puede escribir en 'bot.log'. El logging en archivo se omitirá.")

    logger.info("Logging configurado.")


def ensure_connection(func):
    """
    Decorador que asegura que MT5 esté inicializado y logueado
    antes de llamar a una función del broker.
    Implementa 'exponential backoff' para reintentos de conexión.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected():
            self.logger.warning("Conexión MT5 perdida. Intentando reconectar...")
            try:
                self._connect_with_backoff()
            except ConnectionError as e:
                self.logger.critical(f"Fallo de reconexión permanente: {e}")
                # Propagar el error para que el bucle principal lo maneje
                raise
        
        # Si la conexión es exitosa (o ya estaba), ejecuta la función
        return func(self, *args, **kwargs)
    return wrapper