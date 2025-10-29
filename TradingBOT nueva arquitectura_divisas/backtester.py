# backtester.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

from rl_core.rl_environment import TradingEnv # Importar el mismo entorno
from rl_core.rl_agent import RLAgent # Importar el agente

logger = logging.getLogger(__name__)

def calculate_stats(equity_curve: pd.Series) -> Dict[str, float]:
    """Calcula estadísticas de rendimiento clave desde una curva de equity."""
    
    if equity_curve.empty or len(equity_curve) < 2:
        return {"sharpe": -10.0, "sortino": -10.0, "max_drawdown": 1.0, "total_return": 0.0, "final_equity": 0.0}

    # Asegurarse de que el índice es un DatetimeIndex para el resample
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
         # Si no tiene índice de tiempo (ej. Optuna), usar un rango numérico
         equity_curve.index = pd.to_datetime(pd.RangeIndex(start=0, stop=len(equity_curve), step=1), unit='m')

    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity / equity_curve.iloc[0]) - 1

    # Usar retornos diarios (o por período) para Sharpe/Sortino
    # Resample a diario para un Sharpe comparable
    daily_returns = equity_curve.resample('D').last().pct_change().dropna()
    
    if daily_returns.empty or len(daily_returns) < 2:
        # No hay suficientes datos para stats diarias, usar retornos por paso
        daily_returns = equity_curve.pct_change().dropna()
    
    if daily_returns.empty or len(daily_returns) < 2:
         return {"sharpe": -10.0, "sortino": -10.0, "max_drawdown": 1.0, "total_return": total_return, "final_equity": final_equity}

    # Calcular Sharpe
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) # Anualizado
    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
        sharpe_ratio = -10.0

    # Calcular Sortino (volatilidad a la baja)
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std()
    if downside_std == 0 or np.isnan(downside_std) or np.isinf(downside_std):
        sortino_ratio = sharpe_ratio # Fallback a sharpe
    else:
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252)
        if np.isnan(sortino_ratio) or np.isinf(sortino_ratio):
             sortino_ratio = -10.0

    # Calcular Max Drawdown
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() # Es un número negativo

    return {
        "sharpe": sharpe_ratio,
        "sortino": sortino_ratio,
        "max_drawdown": abs(max_drawdown), # Devolver como positivo
        "total_return": total_return,
        "final_equity": final_equity
    }

def run_portfolio_backtest(df: pd.DataFrame, agent: RLAgent, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta un backtest a nivel de portfolio usando el entorno y el agente.
    """
    logger.info(f"Iniciando backtest de portfolio en {len(df)} velas...")
    
    if df.empty:
        logger.error("El DataFrame para el backtest está vacío.")
        return {"stats": calculate_stats(pd.Series()), "equity_curve": pd.Series()}

    try:
        # Crear el entorno con los datos del backtest
        env = TradingEnv(df=df, config=config)
    except Exception as e:
        logger.error(f"Error al crear TradingEnv para backtest: {e}", exc_info=True)
        return {"stats": calculate_stats(pd.Series()), "equity_curve": pd.Series()}

    obs, _ = env.reset()
    
    equity_history = [env.initial_balance]
    dates = [df.index[0]]
    
    start_time = pd.Timestamp.now()

    try:
        while True:
            # Usar el agente entrenado para predecir
            action = agent.predict(obs) 
            
            # Avanzar el entorno
            obs, reward, terminated, truncated, info = env.step(action)
            
            equity_history.append(info['equity'])
            dates.append(df.index[info['step']])
            
            if terminated or truncated:
                break
                
    except Exception as e:
        logger.error(f"Error durante el bucle de backtest: {e}", exc_info=True)
        # Continuar para calcular stats con lo que se tenga

    end_time = pd.Timestamp.now()
    logger.info(f"Backtest completado en {(end_time - start_time).total_seconds():.2f}s")
    
    equity_curve = pd.Series(equity_history, index=pd.to_datetime(dates))
    stats = calculate_stats(equity_curve)
    
    logger.info(f"Resultados del Backtest: {stats}")
    
    return {"stats": stats, "equity_curve": equity_curve}