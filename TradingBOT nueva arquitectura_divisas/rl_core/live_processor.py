# rl_core/live_processor.py

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
import json
import os
import pytz
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from pyrobot.exceptions import DataError
from .feature_calculator import calculate_features

class LiveProcessor:
    """
    Maneja la obtención, procesamiento y normalización de datos EN VIVO desde MT5
    para la ejecución del agente de RL.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.config_trade = config['trading']
        self.config_features = config['features']
        self.config_rl = config['rl_params']

        self.ticker = self.config_trade['ticker']
        self.interval_mt5 = getattr(mt5, self.config_trade['interval_mt5'])
        self.mt5_timeframe_minutes = self.config_trade['mt5_timeframe']
        self.timezone = pytz.utc

        # --- Cargar Estadísticas de Normalización ---
        self.norm_stats_path = self.config_rl['norm_stats_path']
        self.norm_stats = {}
        try:
            with open(self.norm_stats_path, 'r') as f:
                self.norm_stats = json.load(f)
            if not self.norm_stats:
                raise DataError("El archivo de estadísticas de normalización está vacío.")
            self.logger.info(f"Estadísticas de normalización cargadas desde {self.norm_stats_path}")
        except Exception as e:
            self.logger.critical(f"¡Error fatal! No se pudo cargar '{self.norm_stats_path}'. El bot no puede normalizar datos en vivo. {e}")
            raise

        # --- Definir columnas de features ---
        self.feature_names = self._get_feature_names(self.config_features)
        self.norm_feature_names = self._get_norm_feature_names(self.config_features)
        self.expected_state_size = len(self.norm_feature_names)
        self.logger.info(f"LiveProcessor usará {self.expected_state_size} features normalizadas.")

        # --- Calcular Lookback Máximo ---
        self.max_lookback = self._calculate_max_lookback()
        self.logger.info(f"Lookback máximo requerido para indicadores: {self.max_lookback} velas")
        
        self.df_buffer = pd.DataFrame()
        self.last_candle_time = None

    def _calculate_max_lookback(self) -> int:
        """Calcula el número de velas pasadas necesarias para calentar los indicadores."""
        lookbacks = [
            self.config_features.get('ema_window', 0),
            self.config_features.get('rsi_window', 0),
            self.config_features.get('atr_window', 0),
        ]
        momentum_windows = self.config_features.get('momentum_windows', [])
        if momentum_windows:
            lookbacks.append(max(momentum_windows))
        
        # Añadir un buffer (ej. 50 velas) para asegurar que los TAs tengan datos
        return max(lookbacks) + 50 

    def _get_feature_names(self, features_config: Dict[str, Any]) -> List[str]:
        """Obtiene la lista de nombres de features (sin normalizar)."""
        cols = self.config_features.get('cols_to_normalize', [])
        momentum_cols = [f'momentum_{w}' for w in self.config_features.get('momentum_windows', [])]
        cols.extend(momentum_cols)
        if 'rsi_window' in features_config:
            cols.append('rsi')
        return sorted(list(set(cols)))

    def _get_norm_feature_names(self, features_config: Dict[str, Any]) -> List[str]:
        """Obtiene la lista ordenada de nombres de features normalizadas."""
        cols_to_norm = features_config.get('cols_to_normalize', [])
        momentum_cols = [f'momentum_{w}' for w in features_config.get('momentum_windows', [])]
        
        norm_cols = [f"{col}_norm" for col in (cols_to_norm + momentum_cols)]
        
        if 'rsi_window' in features_config:
            norm_cols.append('rsi_norm')
            
        return sorted(list(set(norm_cols)))

    def _fetch_candles(self, count: int) -> pd.DataFrame:
        """Obtiene velas de MT5 y las formatea en un DataFrame."""
        try:
            rates = mt5.copy_rates_from_pos(self.ticker, self.interval_mt5, 0, count)
            if rates is None or len(rates) == 0:
                raise DataError(f"No se recibieron datos de MT5 para {self.ticker}. Código: {mt5.last_error()}")
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            # Asegurar columnas requeridas por feature_calculator
            df = df[['open', 'high', 'low', 'close', 'volume']]
            return df

        except Exception as e:
            self.logger.error(f"Error al obtener velas de MT5: {e}")
            raise DataError(f"Error en _fetch_candles: {e}")

    def _process_and_normalize_buffer(self) -> pd.DataFrame:
        """
        Toma el buffer de OHLC, calcula features y las normaliza.
        """
        # 1. Calcular características
        df_features = calculate_features(self.df_buffer, self.config_features)
        
        # 2. Normalizar
        df_normalized = df_features.copy()
        
        all_cols_to_norm = self.feature_names # Nombres sin normalizar
        
        for col in all_cols_to_norm:
            if col == 'rsi': continue # RSI tiene manejo especial
            if col in df_normalized.columns:
                mean = self.norm_stats.get(f"{col}_mean")
                std = self.norm_stats.get(f"{col}_std")
                
                if mean is None or std is None:
                    self.logger.warning(f"No se encontraron mean/std para '{col}' en norm_stats.json. Saltando normalización de esta columna.")
                    df_normalized[f"{col}_norm"] = 0.0 # Poner a cero por seguridad
                else:
                    df_normalized[f"{col}_norm"] = (df_normalized[col] - mean) / std
            else:
                self.logger.warning(f"Columna '{col}' para normalizar no encontrada post-cálculo de features.")

        # Normalización especial para RSI
        if 'rsi' in df_normalized.columns and self.norm_stats.get("rsi_norm_method") == "minus_50_div_50":
            df_normalized['rsi_norm'] = (df_normalized['rsi'] - 50.0) / 50.0
        elif 'rsi' in df_normalized.columns:
            self.logger.warning("RSI presente pero 'rsi_norm_method' no es 'minus_50_div_50'. 'rsi_norm' podría faltar.")

        # Limpiar NaNs que resultan del cálculo de indicadores
        df_normalized = df_normalized.dropna()
        
        return df_normalized

    def prime_buffer(self):
        """
        Carga el historial inicial de velas para calentar los indicadores.
        """
        self.logger.info(f"Priming data buffer con {self.max_lookback} velas...")
        try:
            self.df_buffer = self._fetch_candles(self.max_lookback)
            
            # Procesar el buffer inicial
            df_processed = self._process_and_normalize_buffer()
            
            if df_processed.empty:
                raise DataError(f"El buffer quedó vacío después de calcular features y dropna. Aumentar lookback ({self.max_lookback}) o revisar datos/indicadores.")
            
            self.last_candle_time = self.df_buffer.index[-1]
            self.logger.info(f"Buffer 'primeado'. Última vela: {self.last_candle_time}. Velas procesadas: {len(df_processed)}")
        
        except Exception as e:
            self.logger.critical(f"Fallo al 'primear' el buffer de datos: {e}", exc_info=True)
            raise

    def update_and_get_state(self, current_position: int) -> Optional[np.ndarray]:
        """
        Obtiene la última vela, actualiza el buffer, y devuelve el nuevo
        vector de estado para el agente.
        Devuelve None si no hay una vela nueva.
        """
        try:
            # Obtener solo las últimas 2 velas para chequear si hay una nueva
            df_new = self._fetch_candles(2)
            
            latest_candle_time = df_new.index[-1]
            
            if self.last_candle_time is None:
                 self.logger.warning("El buffer no está 'primeado'. Llamando a prime_buffer() primero.")
                 self.prime_buffer()
                 return None # No hay estado en este ciclo, esperar al siguiente

            # Comprobar si hay una vela nueva
            if latest_candle_time <= self.last_candle_time:
                # self.logger.debug(f"No hay vela nueva. Última conocida: {self.last_candle_time}, Última recibida: {latest_candle_time}")
                return None # No hay vela nueva
            
            # --- Hay una vela nueva ---
            self.logger.info(f"Nueva vela detectada: {latest_candle_time} (Anterior: {self.last_candle_time})")
            
            # Obtener la(s) vela(s) que faltan.
            # Podría ser más de una si el bot estuvo desconectado.
            new_candles = df_new[df_new.index > self.last_candle_time]
            
            # Añadir al buffer y mantener el tamaño (eliminar las más antiguas)
            self.df_buffer = pd.concat([self.df_buffer, new_candles])
            self.df_buffer = self.df_buffer.iloc[-self.max_lookback:] # Mantener el buffer manejable
            self.last_candle_time = latest_candle_time

            # --- Procesar el buffer actualizado ---
            df_processed = self._process_and_normalize_buffer()
            
            if df_processed.empty:
                self.logger.error("El buffer procesado está vacío. No se puede generar estado.")
                return None

            # --- Construir el vector de estado ---
            # Obtener la última fila con datos completos
            last_valid_state_row = df_processed.iloc[-1]
            
            # Asegurar que tenemos todas las columnas
            feature_values = last_valid_state_row[self.norm_feature_names].values.astype(np.float32)
            
            if len(feature_values) != self.expected_state_size:
                 self.logger.error(f"¡Discrepancia en features! Esperado: {self.expected_state_size}, Obtenido: {len(feature_values)}")
                 return None

            # Añadir la posición actual
            current_pos_float = np.float32(current_position)
            state_constructed = np.append(feature_values, current_pos_float)
            
            return state_constructed.astype(np.float32)

        except DataError as e:
            self.logger.error(f"Error de datos en update_and_get_state: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error inesperado en update_and_get_state: {e}", exc_info=True)
            return None