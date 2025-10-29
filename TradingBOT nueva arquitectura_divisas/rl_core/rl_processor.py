# rl_core/rl_processor.py

import pandas as pd
import numpy as np
# import MetaTrader5 as mt5 # No necesario para leer CSV
# from datetime import datetime # No necesario para leer CSV
# import pytz # No necesario para leer CSV
import logging
import json
import os
from typing import Dict, Any, List

# from pyrobot.broker import PyRobot # No necesario para leer CSV
from pyrobot.exceptions import ConnectionError, DataError # DataError sí se usa
from .feature_calculator import calculate_features

class RLProcessor:
    """
    Maneja la carga, procesamiento y normalización de datos históricos
    para el entrenamiento y validación del agente de RL DESDE ARCHIVOS CSV.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.config_trade = config['trading']
        self.config_features = config['features']
        self.config_rl = config['rl_params']

        self.ticker = self.config_trade['ticker']
        # self.interval_mt5 = getattr(mt5, self.config_trade['interval_mt5']) # No necesario
        # self.timezone = pytz.utc # No necesario

        self.norm_stats = {} # Almacenará la media y std
        self.stats_path = self.config_rl['norm_stats_path']
        self.csv_data_path = self.config_rl.get('csv_data_path', 'csv_data') # Carga la ruta

        self.logger.info(f"RLProcessor inicializado. Cargando datos desde: {self.csv_data_path}")
        if not os.path.isdir(self.csv_data_path):
             self.logger.warning(f"La carpeta especificada en csv_data_path ('{self.csv_data_path}') no existe o no es un directorio.")
             # Podríamos lanzar un error aquí, pero get_data_for_years ya lo hará si no encuentra archivos.

    def get_data_for_years(self, years: List[int], normalize: bool = True) -> pd.DataFrame:
        """
        Carga datos desde archivos CSV para los años especificados, calcula features
        y (opcionalmente) normaliza los datos.
        """
        all_dfs = []
        interval_str = self.config_trade.get('interval_str', 'M1') # Ej: "M1"

        for year in years:
            try:
                # Construir el nombre del archivo. Ej: "EURUSD_M1_2020.csv"
                file_name = f"{self.ticker}_{interval_str}_{year}.csv"
                file_path = os.path.join(self.csv_data_path, file_name)

                self.logger.info(f"Cargando datos locales desde: {file_path}...")

                if not os.path.exists(file_path):
                    self.logger.warning(f"No se encontró el archivo: {file_path}. Saltando año {year}.")
                    continue

                # Especificar dtype puede ayudar con CSVs grandes o con formatos mixtos
                df_year = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

                # Validar columnas necesarias (OHLC)
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in df_year.columns for col in required_cols):
                     missing = [col for col in required_cols if col not in df_year.columns]
                     self.logger.error(f"Columnas requeridas {missing} no encontradas en {file_path}. Saltando.")
                     continue

                # Seleccionar y reordenar por si acaso
                df_year = df_year[required_cols]

                all_dfs.append(df_year)

            except Exception as e:
                self.logger.error(f"Error cargando o procesando el archivo para el año {year} ({file_path}): {e}", exc_info=True)

        if not all_dfs:
            raise DataError(f"No se pudieron cargar datos CSV válidos para ningún año en la lista: {years}. Verifica la ruta '{self.csv_data_path}' y los nombres/contenido de los archivos (Ej: {self.ticker}_{interval_str}_2020.csv con columnas 'time', 'open', 'high', 'low', 'close')")

        df_full = pd.concat(all_dfs).sort_index()
        df_full = df_full[~df_full.index.duplicated(keep='first')] # Eliminar duplicados si los hubiera

        self.logger.info(f"Datos CSV cargados. Total velas: {len(df_full)}. Calculando características...")

        # 1. Calcular características
        df_processed = calculate_features(df_full, self.config_features)

        # 2. Limpiar NaNs (importante antes de normalizar y entrenar)
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        dropped_rows = initial_rows - len(df_processed)
        self.logger.info(f"Características calculadas. {dropped_rows} filas eliminadas por NaNs. Velas restantes: {len(df_processed)}")

        if df_processed.empty:
            raise DataError("El DataFrame quedó vacío después de calcular features y dropna. Revisa los datos de entrada o los periodos de los indicadores.")

        # 3. Normalizar (si se solicita, ej. para datos de entrenamiento)
        if normalize:
            self.logger.info("Calculando y guardando estadísticas de normalización...")
            self.norm_stats = {}

            # Definir columnas a normalizar
            cols_to_norm = self.config_features.get('cols_to_normalize', [])
            momentum_cols = [f'momentum_{w}' for w in self.config_features.get('momentum_windows', [])]
            all_cols_to_norm = cols_to_norm + momentum_cols

            for col in all_cols_to_norm:
                if col in df_processed.columns:
                    mean = df_processed[col].mean()
                    std = df_processed[col].std()
                    # Prevenir división por cero o std muy pequeño
                    if pd.isna(std) or std < 1e-9:
                        std = 1e-9
                        self.logger.warning(f"Std dev para '{col}' es casi cero o NaN. Usando 1e-9 para evitar división por cero.")

                    self.norm_stats[f"{col}_mean"] = mean
                    self.norm_stats[f"{col}_std"] = std

                    # Aplicar normalización Z-score
                    df_processed[f"{col}_norm"] = (df_processed[col] - mean) / std
                else:
                    self.logger.warning(f"Columna '{col}' definida para normalizar no se encontró en el DataFrame.")

            # Normalización especial para RSI (entre -1 y 1)
            if 'rsi' in df_processed.columns:
                df_processed['rsi_norm'] = (df_processed['rsi'] - 50.0) / 50.0
                # Guardar info de que RSI se normaliza así
                self.norm_stats["rsi_norm_method"] = "minus_50_div_50"

            # Guardar estadísticas
            try:
                with open(self.stats_path, 'w') as f:
                    json.dump(self.norm_stats, f, indent=4)
                self.logger.info(f"Estadísticas de normalización guardadas en {self.stats_path}")
            except Exception as e:
                self.logger.error(f"No se pudo guardar {self.stats_path}: {e}")

        return df_processed

    def shutdown(self):
        """Función vacía (ya no hay conexión MT5 que cerrar)."""
        self.logger.info("RLProcessor (modo CSV) finalizado.")