# train_rl.py

import sys
import os
import logging
import pandas as pd
import numpy as np
import warnings
import optuna

from pyrobot.config_loader import load_config
# DEPRECADO: from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict, Any

# --- 1. Configuración de Path y Logging ---
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

try:
    from pyrobot.utils import setup_logging
    from pyrobot.exceptions import DataError
    from rl_core.rl_processor import RLProcessor
    from rl_core.rl_environment import TradingEnv
    from rl_core.rl_agent import RLAgent
    # --- NUEVO: Importar el backtester de portfolio ---
    from backtester import run_portfolio_backtest, calculate_stats

except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que todos los archivos (rl_processor, rl_environment, etc.) existen y no hay errores de sintaxis.")
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore') # Opcional

# --- Cargar Config ---
try:
    config = load_config(os.path.join(dir_path, 'config.yaml'))
    logger.info("Configuración 'config.yaml' cargada.")
except Exception as e:
    logger.critical(f"Error al leer 'config.yaml': {e}", exc_info=True)
    sys.exit(1)

# --- Variables Globales de Datos ---
df_train = pd.DataFrame()
df_val = pd.DataFrame()
processor = None

# --- NUEVA FUNCIÓN AUXILIAR ---
def apply_train_stats_to_df(df_raw: pd.DataFrame, stats: Dict, config: Dict) -> pd.DataFrame:
    """Aplica estadísticas de normalización pre-calculadas a un DataFrame."""
    logger.info("Aplicando estadísticas de normalización de TRAIN a datos...")
    if not stats:
         logger.warning("No se encontraron estadísticas (norm_stats). Los datos no se normalizarán.")
         return df_raw.copy()

    features_config = config['features']
    cols_to_norm = features_config.get('cols_to_normalize', [])
    momentum_cols = [f'momentum_{w}' for w in features_config.get('momentum_windows', [])]
    all_cols_to_norm = cols_to_norm + momentum_cols
    df_norm = df_raw.copy()

    for col in all_cols_to_norm:
         if col in df_norm.columns:
             mean = stats.get(f"{col}_mean")
             std = stats.get(f"{col}_std")
             if mean is not None and std is not None:
                  df_norm[f"{col}_norm"] = (df_norm[col] - mean) / std
             else:
                  logger.warning(f"No se encontraron mean/std para '{col}' al normalizar datos. Columna '{col}_norm' será 0.0.")
                  if f"{col}_norm" not in df_norm.columns: df_norm[f"{col}_norm"] = 0.0

    if 'rsi' in df_norm.columns and stats.get("rsi_norm_method") == "minus_50_div_50":
         df_norm['rsi_norm'] = (df_norm['rsi'] - 50.0) / 50.0
    elif 'rsi_norm' not in df_norm.columns and 'rsi_window' in features_config:
         logger.warning("Columna 'rsi_norm' no generada para validación.")
         df_norm['rsi_norm'] = 0.0

    initial_rows = len(df_norm)
    df_norm = df_norm.dropna() # Eliminar NaNs post-normalización
    dropped_rows = initial_rows - len(df_norm)
    if dropped_rows > 0: 
        logger.info(f"{dropped_rows} filas eliminadas de datos por NaNs post-normalización.")
    
    return df_norm

def load_data():
    """Carga y procesa los datos de entrenamiento y validación."""
    global df_train, df_val, processor
    try:
        processor = RLProcessor(config)
        train_years = config['rl_params']['training_years']
        val_years = config['rl_params']['validation_years']

        logger.info(f"Cargando y procesando datos de entrenamiento ({train_years})...")
        # df_train se normaliza y sus stats se guardan en processor.norm_stats
        df_train = processor.get_data_for_years(train_years, normalize=True)

        logger.info(f"Cargando y procesando datos de validación ({val_years})...")
        # df_val se carga sin normalizar
        df_val_raw = processor.get_data_for_years(val_years, normalize=False)

        # Aplicar stats de train a val
        df_val = apply_train_stats_to_df(df_val_raw, processor.norm_stats, config)

        if df_train.empty or df_val.empty:
            raise DataError("Los datos de entrenamiento o validación están vacíos después del procesamiento.")

        logger.info("Datos de entrenamiento y validación listos.")

    except DataError as e:
        logger.critical(f"Fallo al cargar/procesar datos CSV: {e}", exc_info=False)
        if processor: processor.shutdown()
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fallo inesperado al cargar/procesar datos: {e}", exc_info=True)
        if processor: processor.shutdown()
        sys.exit(1)

# --- (El resto del archivo 'train_rl.py' no cambia) ---

def objective(trial: optuna.Trial) -> float:
    """Función que Optuna intentará maximizar."""
    logger.info(f"\n--- Iniciando Trial {trial.number} ---")

    # 1. Sugerir Hiperparámetros
    hyperparameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ppo_n_steps": trial.suggest_categorical("ppo_n_steps", [512, 1024, 2048, 4096]),
        "ppo_batch_size": trial.suggest_categorical("ppo_batch_size", [32, 64, 128, 256]),
        "ppo_n_epochs": trial.suggest_int("ppo_n_epochs", 5, 20),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 2.0),
    }

    # 2. Crear Entorno y Agente con estos HP
    if df_train.empty:
        logger.error("df_train está vacío en 'objective'. No se puede crear el entorno.")
        return -np.inf # Fallo

    try:
        train_env = TradingEnv(df_train, config)
    except ValueError as e:
        logger.error(f"Error al crear TradingEnv para entrenamiento en Trial {trial.number}: {e}")
        return -np.inf

    agent = RLAgent(config)
    try:
        model = agent.create_model(train_env, hyperparameters)
    except Exception as e:
        logger.error(f"Error al crear el modelo PPO en Trial {trial.number}: {e}", exc_info=True)
        return -np.inf

    # 3. Entrenar
    optimization_timesteps = config['rl_params'].get('optimization_timesteps', 500000)
    try:
        # El agente ahora se entrena usando model.learn
        agent.model.learn(total_timesteps=optimization_timesteps, tb_log_name=f"trial_{trial.number}")
    except Exception as e:
        logger.error(f"Trial {trial.number} falló durante el entrenamiento: {e}", exc_info=True)
        return -np.inf # Indicar fallo a Optuna

    # 4. Evaluar en el Entorno de Validación usando NUESTRO BACKTESTER
    logger.info(f"Evaluando Trial {trial.number} en datos de validación...")
    if df_val.empty:
        logger.error("df_val está vacío en 'objective'. No se puede evaluar.")
        return -np.inf

    try:
        # ¡¡CAMBIO!! Usar run_portfolio_backtest en lugar de evaluate_policy
        # Pasamos el DataFrame de validación y el agente ya entrenado
        backtest_results = run_portfolio_backtest(df_val, agent, config)
        
        stats = backtest_results['stats']
        
        # Métrica objetivo para Optuna (queremos maximizar Sharpe)
        metric_to_optimize = stats.get("sharpe", -10.0) 
        # Podrías cambiar a 'sortino' o 'final_equity'
        
        logger.info(f"Trial {trial.number}: Sharpe={stats['sharpe']:.4f}, Sortino={stats['sortino']:.4f}, MaxDD={stats['max_drawdown']:.2%}, Return={stats['total_return']:.2%}")

        if np.isnan(metric_to_optimize) or np.isinf(metric_to_optimize):
             logger.warning(f"Trial {trial.number} resultó en métrica inválida ({metric_to_optimize}). Devolviendo -inf.")
             return -np.inf

    except Exception as e:
        logger.error(f"Error durante la evaluación (backtest) en Trial {trial.number}: {e}", exc_info=True)
        return -np.inf

    # 5. Reportar métrica a Optuna
    return metric_to_optimize

def main():
    global processor # Para acceder al procesador en 'finally'

    try:
        # Cargar datos ANTES de iniciar el estudio
        load_data()

        logger.info("--- INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS (OPTUNA) ---")

        study_name = config['rl_params']['optuna_study_name']
        storage_name = config['rl_params']['optuna_storage_db']
        n_trials = config['rl_params']['optuna_n_trials']

        # Crear o cargar estudio Optuna
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize" # Maximizar Sharpe/Sortino
        )

        try:
            # Ejecutar optimización
            study.optimize(objective, n_trials=n_trials, timeout=None)
        except KeyboardInterrupt:
            logger.warning("Optimización interrumpida por el usuario.")
        except Exception as e:
            logger.critical(f"Ocurrió un error fatal durante la optimización: {e}", exc_info=True)

        # --- Resultados de Optuna ---
        logger.info("\n--- OPTIMIZACIÓN COMPLETA ---")
        logger.info(f"Número de trials finalizados: {len(study.trials)}")

        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and not np.isinf(t.value) and not np.isnan(t.value)]

        if not valid_trials:
             logger.critical("Optuna no completó ningún trial exitoso. No se puede reentrenar.")
             logger.critical("Revisa los logs anteriores en busca de errores en los trials.")
             return # Salir si no hay trials válidos

        best_trial = study.best_trial
        logger.info(f"Mejor Trial: {best_trial.number}")
        logger.info(f"  Valor (Métrica Optimizada): {best_trial.value:.4f}")
        logger.info("  Mejores Hiperparámetros:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        best_hyperparameters = best_trial.params

        # --- Reentrenamiento Final con los Mejores Hiperparámetros ---
        logger.info("\n--- REENTRENANDO MODELO FINAL CON MEJORES HIPERPARÁMETROS ---")

        if df_train.empty or df_val.empty:
            logger.error("Los DataFrames de entrenamiento o validación están vacíos. No se puede reentrenar.")
            return

        # Combinar train + val para el entrenamiento final
        df_final_train = pd.concat([df_train, df_val]).sort_index()
        df_final_train = df_final_train[~df_final_train.index.duplicated(keep='first')]
        df_final_train = df_final_train.dropna() 

        if df_final_train.empty:
            logger.error("El DataFrame combinado para el entrenamiento final está vacío después de dropna().")
            return

        logger.info(f"Datos combinados para entrenamiento final: {len(df_final_train)} velas.")

        try:
            final_env = TradingEnv(df_final_train, config)
        except ValueError as e:
            logger.critical(f"Error al crear el entorno final para reentrenamiento: {e}")
            return

        final_agent = RLAgent(config)

        try:
            # Crear el modelo con los mejores HP
            final_agent.create_model(final_env, best_hyperparameters)
            
            # Entrenar el modelo final
            final_training_timesteps = config['rl_params'].get('total_timesteps', 5000000)
            final_agent.train(final_env, total_timesteps=final_training_timesteps)

            logger.info("--- MODELO FINAL ENTRENADO Y GUARDADO ---")
            logger.info(f"Modelo guardado en: {config['rl_params']['model_save_path']}")
            logger.info(f"Estadísticas de normalización (usadas) guardadas en: {config['rl_params']['norm_stats_path']}")

            # --- Ejecutar backtest final sobre los datos de TEST (si existen) ---
            test_years = config['rl_params'].get('test_years', [])
            if test_years:
                logger.info(f"\n--- EJECUTANDO BACKTEST FINAL EN DATOS DE TEST ({test_years}) ---")
                df_test_raw = processor.get_data_for_years(test_years, normalize=False)
                df_test_norm = apply_train_stats_to_df(df_test_raw, processor.norm_stats, config)
                
                if df_test_norm.empty:
                     logger.error("Datos de test vacíos después de procesar. No se puede ejecutar backtest final.")
                else:
                    test_results = run_portfolio_backtest(df_test_norm, final_agent, config)
                    logger.info("--- RESULTADOS DEL BACKTEST DE TEST (Out-of-Sample) ---")
                    logger.info(f"Sharpe: {test_results['stats']['sharpe']:.4f}")
                    logger.info(f"Sortino: {test_results['stats']['sortino']:.4f}")
                    logger.info(f"Max Drawdown: {test_results['stats']['max_drawdown']:.2%}")
                    logger.info(f"Total Return: {test_results['stats']['total_return']:.2%}")
                    logger.info(f"Final Equity: ${test_results['stats']['final_equity']:,.2f}")
                    # Guardar curva de equity
                    test_results['equity_curve'].to_csv("final_backtest_equity_curve.csv")
                    logger.info("Curva de equity del backtest final guardada en 'final_backtest_equity_curve.csv'")

        except Exception as e:
            logger.critical(f"Fallo durante el reentrenamiento final: {e}", exc_info=True)

    finally:
        if processor:
            processor.shutdown()
            logger.info("RLProcessor (modo CSV) finalizado.")


if __name__ == "__main__":
    main()
