# download_data.py

import os
import yaml
import logging
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
TIMEZONE = pytz.utc

def connect_mt5():
    """Conecta a MT5 usando variables de entorno."""
    try:
        login = int(os.environ['1511879566'])
        password = os.environ['E97xsv?Q']
        server = os.environ['FTMO-Demo']
    except KeyError as e:
        logger.critical(f"¡ERROR DE CREDENCIALES! Variable de entorno no encontrada: {e}")
        logger.critical("Por favor, configura MT5_LOGIN, MT5_PASS, y MT5_SERVER.")
        return False
    except (ValueError, TypeError):
        logger.critical("¡ERROR! MT5_LOGIN debe ser un número entero.")
        return False

    if not mt5.initialize():
        logger.error(f"mt5.initialize() falló. Código: {mt5.last_error()}")
        return False
        
    if not mt5.login(login, password, server):
        logger.error(f"mt5.login() falló. Código: {mt5.last_error()}. Revisa credenciales.")
        return False
        
    logger.info(f"Conexión MT5 establecida para login {login}.")
    return True

def download_data_for_year(year: int, ticker: str, interval_mt5, save_path: str):
    """Descarga los datos de un año completo y los guarda en CSV."""
    
    logger.info(f"Descargando datos para {ticker} - {year}...")
    
    # Definir el rango de fechas
    date_from = datetime(year, 1, 1, tzinfo=TIMEZONE)
    date_to = datetime(year + 1, 1, 1, tzinfo=TIMEZONE) # Hasta el 1 de Enero del siguiente año
    
    try:
        # Descargar los datos
        rates = mt5.copy_rates_range(ticker, interval_mt5, date_from, date_to)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No se recibieron datos para {ticker} en {year}. Código: {mt5.last_error()}")
            return

        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        
        # --- FORMATEO CRÍTICO ---
        # 1. Convertir la columna 'time' de timestamp a datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 2. Seleccionar solo las columnas que rl_processor.py necesita
        # (El script original espera 'time', 'open', 'high', 'low', 'close')
        df_save = df[['time', 'open', 'high', 'low', 'close']]
        
        # 3. Guardar en CSV CON encabezado e SIN índice
        # Esto crea un archivo con "time,open,high,low,close" en la primera línea
        df_save.to_csv(save_path, index=False)
        
        logger.info(f"¡Éxito! Datos de {year} guardados en {save_path} ({len(df_save)} velas)")

    except Exception as e:
        logger.error(f"Error al descargar datos para {year}: {e}", exc_info=True)

def main():
    logger.info("--- Iniciando Script de Descarga de Datos ---")
    
    # 1. Cargar Config
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Configuración 'config.yaml' cargada.")
    except Exception as e:
        logger.critical(f"Error al leer 'config.yaml': {e}", exc_info=True)
        return

    config_trade = config['trading']
    config_rl = config['rl_params']

    # 2. Conectar a MT5
    if not connect_mt5():
        return
        
    # 3. Preparar parámetros
    ticker = config_trade['ticker']
    interval_str = config_trade['interval_str']
    interval_mt5 = getattr(mt5, config_trade['interval_mt5'])
    csv_dir = config_rl.get('csv_data_path', 'csv_data')
    
    # Crear carpeta 'csv_data' si no existe
    os.makedirs(csv_dir, exist_ok=True)
    
    # 4. Obtener todos los años necesarios
    years_to_download = sorted(list(set(
        config_rl.get('training_years', []) +
        config_rl.get('validation_years', []) +
        config_rl.get('test_years', [])
    )))
    
    if not years_to_download:
        logger.warning("No hay años definidos en 'training_years', 'validation_years', o 'test_years' en config.yaml.")
        return

    logger.info(f"Años a descargar: {years_to_download}")

    # 5. Iterar y descargar
    for year in years_to_download:
        # Crear el nombre de archivo que rl_processor.py espera
        file_name = f"{ticker}_{interval_str}_{year}.csv"
        save_path = os.path.join(csv_dir, file_name)
        
        download_data_for_year(year, ticker, interval_mt5, save_path)

    # 6. Desconectar
    mt5.shutdown()
    logger.info("--- Descarga de Datos Completada ---")

if __name__ == "__main__":
    main()