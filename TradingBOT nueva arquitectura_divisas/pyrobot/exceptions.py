# pyrobot/exceptions.py

"""
Excepciones personalizadas para el Bot de Trading.
"""

class BotException(Exception):
    """Excepción base para todos los errores del bot."""
    pass

class ConnectionError(BotException):
    """Errores relacionados con la conexión a MT5."""
    pass

class OrderError(BotException):
    """Errores relacionados con el envío o cierre de órdenes."""
    pass

class DataError(BotException):
    """Errores relacionados con la obtención o procesamiento de datos."""
    pass