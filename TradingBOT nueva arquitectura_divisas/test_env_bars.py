# test_env_vars.py
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info("Intentando leer las variables de entorno MT5...")

login = os.environ.get('MT5_LOGIN')
password = os.environ.get('MT5_PASS')
server = os.environ.get('MT5_SERVER')

found_all = True

if login:
    logger.info(f"MT5_LOGIN encontrada: {login}")
else:
    logger.error("MT5_LOGIN NO encontrada.")
    found_all = False

if password:
    logger.info(f"MT5_PASS encontrada: *** (oculta por seguridad)") # No imprimimos la contraseña
else:
    logger.error("MT5_PASS NO encontrada.")
    found_all = False

if server:
    logger.info(f"MT5_SERVER encontrada: {server}")
else:
    logger.error("MT5_SERVER NO encontrada.")
    found_all = False

if found_all:
    logger.info("¡Éxito! Todas las variables fueron leídas por Python.")
else:
    logger.warning("Al menos una variable no fue encontrada por Python.")

# Opcional: Imprimir TODAS las variables que Python ve (puede ser largo)
# print("\n--- Todas las variables de entorno visibles para Python ---")
# for name, value in os.environ.items():
#     print(f"{name}={value}")