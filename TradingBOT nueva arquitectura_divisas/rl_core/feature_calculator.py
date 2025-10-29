# rl_core/feature_calculator.py

import pandas as pd
import talib as ta
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_features(df: pd.DataFrame, features_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calcula las características técnicas (sin normalizar) en un DataFrame.
    """
    data = df.copy()
    close = data['close']
    high = data['high']
    low = data['low']

    try:
        # --- Calcular Indicadores (similar a RLProcessor) ---
        ema_window = features_config['ema_window']
        data['ema'] = ta.EMA(close, timeperiod=ema_window)
        data['ema_spread'] = data['close'] - data['ema']
        
        data['rsi'] = ta.RSI(close, timeperiod=features_config['rsi_window'])
        
        data['atr'] = ta.ATR(high, low, close, timeperiod=features_config['atr_window'])
        
        for window in features_config['momentum_windows']:
            data[f'momentum_{window}'] = close.pct_change(window)

        # Devolver el DataFrame con las nuevas columnas (sin normalizar)
        return data

    except Exception as e:
        logger.error(f"Error durante el cálculo de características: {e}", exc_info=True)
        # Devolver el dataframe original si falla
        return df