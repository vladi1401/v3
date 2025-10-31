# download_data.py (Usando la biblioteca 'duka' - CORREGIDO v3)

import os
import logging
import pandas as pd
# import MetaTrader5 as mt5 # No necesario
import pytz
from datetime import datetime
from typing import List
import yaml

# --- Importar la biblioteca duka ---
try:
    import duka.main as duka # Importar la función principal de la biblioteca
except ImportError:
    print("*"*60)
    print("ERROR: La biblioteca 'duka' no está instalada.")
    print("Por favor, instálala ejecutando: pip install duka")
    print("*"*60)
    exit()
# ------------------------------------

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Parámetros (Leídos desde config.yaml) ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    TICKER = config['trading']['ticker'].upper() # Asegurar mayúsculas
    TIMEFRAME_STR = config['trading']['interval_str'] # M1, H1 etc.
    OUTPUT_DIR = config['rl_params']['csv_data_path'] # Carpeta csv_data

    # Combinar todos los años necesarios
    YEARS_TO_DOWNLOAD = sorted(list(set(
        config['rl_params']['training_years'] +
        config['rl_params']['validation_years'] +
        config['rl_params']['test_years']
    )))

    logger.info(f"Configuración cargada. Se descargará: {TICKER} ({TIMEFRAME_STR}) usando 'duka'")
    logger.info(f"Años: {YEARS_TO_DOWNLOAD}")
    logger.info(f"Directorio de salida: {OUTPUT_DIR}")

except Exception as e:
    logger.critical(f"Error al leer 'config.yaml': {e}")
    exit()

# ---------------------------------------------------------

# Mapeo de timeframe para 'duka'
DUKA_TIMEFRAME = TIMEFRAME_STR.lower() # Convertir 'M1' a 'm1'

# Directorio temporal donde 'duka' descargará los datos inicialmente
DUKA_CACHE_DIR = "duka_cache"

def format_downloaded_csv(duka_csv_path: str, output_filepath: str, year: int):
    """Lee el CSV descargado por 'duka' y lo guarda en el formato para rl_processor.py."""
    try:
        logger.info(f"Formateando archivo: {duka_csv_path}")

        df = pd.read_csv(
            duka_csv_path,
            parse_dates=['time'],
            index_col='time'
        )

        df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        }, inplace=True, errors='ignore')

        if df is None or df.empty:
            logger.warning(f"El archivo {duka_csv_path} está vacío o no se pudo leer.")
            return False

        logger.info(f"Se leyeron {len(df)} velas del archivo descargado.")

        # --- FORMATEO CRÍTICO para rl_processor.py ---
        df_out = pd.DataFrame()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
             df.index = df.index.tz_localize('UTC')
        else:
             df.index = df.index.tz_convert('UTC')

        df_out['date'] = df.index.strftime('%Y.%m.%d')
        df_out['time_str'] = df.index.strftime('%H:%M')

        required_cols_in = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols_in):
            logger.error(f"Columnas OHLC faltantes en {duka_csv_path}. Columnas: {df.columns.tolist()}")
            return False

        df_out['open'] = df['open']
        df_out['high'] = df['high']
        df_out['low'] = df['low']
        df_out['close'] = df['close']
        df_out['volume'] = df.get('volume', 0.0)

        df_out.to_csv(output_filepath, header=False, index=False, float_format='%.5f')

        logger.info(f"Año {year} formateado y guardado exitosamente en: {output_filepath}")
        return True

    except Exception as e:
        logger.error(f"Error al formatear el archivo {duka_csv_path}: {e}", exc_info=True)
        return False

def main():
    logger.info("--- Iniciando Script de Descarga de Datos (usando 'duka') ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DUKA_CACHE_DIR, exist_ok=True)

    processed_count = 0
    for year in YEARS_TO_DOWNLOAD:
        logger.info(f"--- Procesando año: {year} ---")

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        try:
            logger.info(f"Llamando a duka.main para descargar {TICKER} ({DUKA_TIMEFRAME}) para {year}...")

            # --- INICIO DE LA CORRECCIÓN ---
            # Usar los nombres de argumento correctos: 'start' y 'end'
            duka.main(
                TICKER,
                start=start_date,     # <-- CORREGIDO
                end=end_date,       # <-- CORREGIDO
                timeframe=DUKA_TIMEFRAME,
                folder=DUKA_CACHE_DIR,
                header=True
            )
            # --- FIN DE LA CORRECCIÓN ---

            logger.info(f"Descarga de 'duka' para {year} completada (o ya existía en caché).")

            # --- Encontrar el archivo CSV descargado por duka ---
            start_str = start_date.strftime('%Y_%m_%d')
            end_str = end_date.strftime('%Y_%m_%d')
            expected_duka_filename = f"{TICKER}_{DUKA_TIMEFRAME}_{start_str}_{end_str}.csv"
            duka_csv_path = os.path.join(DUKA_CACHE_DIR, TICKER, DUKA_TIMEFRAME, expected_duka_filename)

            if not os.path.exists(duka_csv_path):
                logger.error(f"No se encontró el archivo CSV esperado descargado por duka: {duka_csv_path}")
                found = False
                search_dir = os.path.join(DUKA_CACHE_DIR, TICKER, DUKA_TIMEFRAME)
                if os.path.isdir(search_dir):
                    for fname in os.listdir(search_dir):
                        if TICKER in fname and DUKA_TIMEFRAME in fname and str(year) in fname and fname.endswith('.csv'):
                             duka_csv_path = os.path.join(search_dir, fname)
                             logger.info(f"Encontrado archivo alternativo: {duka_csv_path}")
                             found = True
                             break
                if not found:
                    logger.error(f"No se encontró ningún archivo CSV de duka para {year} en {search_dir}")
                    continue

            # --- Formatear el archivo encontrado ---
            output_filename = f"DAT_ASCII_{TICKER}_{TIMEFRAME_STR}_{year}.csv"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            if format_downloaded_csv(duka_csv_path, output_filepath, year):
                processed_count += 1

        except Exception as e:
            logger.error(f"Error al descargar o procesar el año {year} con 'duka': {e}", exc_info=True)


    if processed_count == len(YEARS_TO_DOWNLOAD):
        logger.info(f"--- Descarga ('duka') y Formateo Completados ({processed_count} años) ---")
    elif processed_count > 0:
         logger.warning(f"--- Descarga ('duka') y Formateo Parcial ({processed_count}/{len(YEARS_TO_DOWNLOAD)} años completados) ---")
    else:
        logger.error("--- Falló la Descarga ('duka') y/o Formateo para todos los años. Revisa los logs. ---")


if __name__ == "__main__":
    main()