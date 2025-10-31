# download_data.py

import os
import sys
import logging
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
    pd = None  # type: ignore
    PANDAS_IMPORT_ERROR = exc
else:
    PANDAS_IMPORT_ERROR = None
from datetime import datetime, timezone
from typing import List

try:
    import MetaTrader5 as mt5
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
    mt5 = None  # type: ignore
    MT5_IMPORT_ERROR = exc
else:
    MT5_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from pyrobot.config_loader import load_config

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Parámetros de Descarga (Leídos desde config.yaml) ---
try:
    config_path = PARENT_DIR / 'config.yaml'
    config = load_config(config_path)

    TICKER = config['trading']['ticker']
    TIMEFRAME_STR = config['trading']['interval_str']
    TIMEFRAME_NAME = config['trading']['interval_mt5']
    OUTPUT_DIR = config['rl_params']['csv_data_path']

    # Combinar todos los años necesarios
    YEARS_TO_DOWNLOAD = sorted(list(set(
        config['rl_params']['training_years'] +
        config['rl_params']['validation_years'] +
        config['rl_params']['test_years']
    )))

    logger.info(f"Configuración cargada. Se descargará: {TICKER} ({TIMEFRAME_STR})")
    logger.info(f"Años: {YEARS_TO_DOWNLOAD}")
    logger.info(f"Directorio de salida: {OUTPUT_DIR}")

except Exception as e:
    logger.critical(f"Error al leer 'config.yaml': {e}")
    raise SystemExit(1)

# ---------------------------------------------------------

def connect_mt5():
    """Conecta a MT5 usando variables de entorno."""
    if mt5 is None:
        logger.critical(
            "MetaTrader5 no está instalado. Instala la librería oficial 'MetaTrader5' para continuar."
        )
        if MT5_IMPORT_ERROR:
            logger.debug("Detalle del error de importación: %s", MT5_IMPORT_ERROR)
        return False

    if pd is None:
        logger.critical(
            "Pandas no está instalado. Instala 'pandas' para poder procesar los datos descargados."
        )
        if PANDAS_IMPORT_ERROR:
            logger.debug("Detalle del error de importación: %s", PANDAS_IMPORT_ERROR)
        return False

    try:
        login = int(os.environ['MT5_LOGIN'])
        password = os.environ['MT5_PASS']
        server = os.environ['MT5_SERVER']
        logger.info(f"Conectando a {server} con login {login}...")
    except KeyError as e:
        logger.critical(f"¡ERROR! Variable de entorno no encontrada: {e}")
        logger.critical("Por favor, configura MT5_LOGIN, MT5_PASS, y MT5_SERVER.")
        return False
    except (ValueError, TypeError):
        logger.critical("¡ERROR! MT5_LOGIN debe ser un número entero.")
        return False

    if not mt5.initialize():
        logger.critical(f"mt5.initialize() falló. Código: {mt5.last_error()}")
        return False
    
    if not mt5.login(login, password, server):
        logger.critical(f"mt5.login() falló. Código: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    logger.info("Conexión MT5 establecida.")
    return True

def download_data(ticker: str, timeframe_mt5, timeframe_str: str, years: List[int], output_dir: str):
    """Descarga los datos y los guarda en el formato CSV requerido."""

    if mt5 is None or pd is None:
        logger.error("No se puede descargar datos porque faltan dependencias críticas (MetaTrader5 o pandas).")
        return

    # Asegurarse de que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Definir la zona horaria UTC para MT5
    utc_tz = timezone.utc

    for year in years:
        logger.info(f"--- Procesando año: {year} ---")
        
        # Definir el rango de fechas para el año completo
        date_from = datetime(year, 1, 1, tzinfo=utc_tz)
        date_to = datetime(year + 1, 1, 1, tzinfo=utc_tz) # Hasta el inicio del próximo año

        try:
            # Obtener los datos
            rates = mt5.copy_rates_range(ticker, timeframe_mt5, date_from, date_to)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No se obtuvieron datos para {ticker} en {year}.")
                continue
                
            logger.info(f"Se obtuvieron {len(rates)} velas para {year}.")

            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            
            # --- FORMATEO CRÍTICO ---
            # El rl_processor.py espera columnas 'date' (YYYY.MM.DD) y 'time_str' (HH:MM)
            
            # 1. Convertir timestamp a datetime
            df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
            
            # 2. Crear las columnas de texto separadas
            df['date'] = df['time_dt'].dt.strftime('%Y.%m.%d')
            df['time_str'] = df['time_dt'].dt.strftime('%H:%M')
            
            # 3. Renombrar volumen
            df = df.rename(columns={'tick_volume': 'volume'})
            
            # 4. Seleccionar y ordenar las columnas FINALES
            df_out = df[['date', 'time_str', 'open', 'high', 'low', 'close', 'volume']]
            
            # 5. Definir el nombre del archivo de salida
            # (Usamos DAT_ASCII_ para coincidir con la corrección anterior)
            file_name = f"DAT_ASCII_{ticker}_{timeframe_str}_{year}.csv"
            file_path = os.path.join(output_dir, file_name)

            # 6. Guardar sin encabezado y sin índice
            df_out.to_csv(file_path, header=False, index=False)
            
            logger.info(f"Año {year} guardado exitosamente en: {file_path}")

        except Exception as e:
            logger.error(f"Error al procesar el año {year}: {e}", exc_info=True)

def main():
    if not connect_mt5():
        return

    try:
        timeframe_mt5 = getattr(mt5, TIMEFRAME_NAME)
    except AttributeError:
        logger.critical(
            f"El timeframe '{TIMEFRAME_NAME}' no existe en la librería MetaTrader5."
        )
        if mt5 is not None:
            mt5.shutdown()
        return

    download_data(
        ticker=TICKER,
        timeframe_mt5=timeframe_mt5,
        timeframe_str=TIMEFRAME_STR,
        years=YEARS_TO_DOWNLOAD,
        output_dir=OUTPUT_DIR
    )

    if mt5 is not None:
        mt5.shutdown()
    logger.info("Descarga completada. Conexión MT5 cerrada.")

if __name__ == "__main__":
    main()