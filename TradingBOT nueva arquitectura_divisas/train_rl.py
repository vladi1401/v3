# train_rl.py

import sys
import os
import yaml
import logging
import pandas as pd
import numpy as np
import warnings
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict, Any

# --- 1. Configuración de Path y Logging ---
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

try:
    from pyrobot.utils import setup_logging
    # --- CORRECCIÓN: Añadir import DataError ---
    from pyrobot.exceptions import DataError
    from rl_core.rl_processor import RLProcessor
    from rl_core.rl_environment import TradingEnv
    from rl_core.rl_agent import RLAgent
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que todos los archivos (rl_processor, rl_environment, etc.) existen y no hay errores de sintaxis.")
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore') # Opcional

# --- Cargar Config ---
try:
    with open(os.path.join(dir_path, 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info("Configuración 'config.yaml' cargada.")
except Exception as e:
    logger.critical(f"Error al leer 'config.yaml': {e}", exc_info=True)
    sys.exit(1)

# --- Variables Globales de Datos ---
df_train = pd.DataFrame()
df_val = pd.DataFrame()
processor = None

def load_data():
    """Carga y procesa los datos de entrenamiento y validación."""
    global df_train, df_val, processor
    try:
        processor = RLProcessor(config)
        train_years = config['rl_params']['training_years']
        val_years = config['rl_params']['validation_years']

        logger.info(f"Cargando y procesando datos de entrenamiento ({train_years})...")
        df_train = processor.get_data_for_years(train_years, normalize=True)

        logger.info(f"Cargando y procesando datos de validación ({val_years})...")
        df_val_raw = processor.get_data_for_years(val_years, normalize=False)

        logger.info("Aplicando estadísticas de normalización de TRAIN a VALIDATION...")
        stats = processor.norm_stats
        if not stats:
             logger.warning("No se encontraron estadísticas de normalización (norm_stats). La validación podría no estar normalizada correctamente.")
             df_val = df_val_raw.copy()
        else:
            cols_to_norm = config['features'].get('cols_to_normalize', [])
            momentum_cols = [f'momentum_{w}' for w in config['features'].get('momentum_windows', [])]
            all_cols_to_norm = cols_to_norm + momentum_cols
            df_val = df_val_raw.copy()

            for col in all_cols_to_norm:
                 if col in df_val.columns:
                     mean = stats.get(f"{col}_mean")
                     std = stats.get(f"{col}_std")
                     if mean is not None and std is not None:
                          df_val[f"{col}_norm"] = (df_val[col] - mean) / std
                     else:
                          logger.warning(f"No se encontraron mean/std para '{col}' al normalizar datos de validación. Columna '{col}_norm' podría faltar o ser incorrecta.")
                          if f"{col}_norm" not in df_val.columns: df_val[f"{col}_norm"] = 0.0

            if 'rsi' in df_val.columns and stats.get("rsi_norm_method") == "minus_50_div_50":
                 df_val['rsi_norm'] = (df_val['rsi'] - 50.0) / 50.0
            elif 'rsi_norm' not in df_val.columns and 'rsi_window' in config['features']:
                 logger.warning("Columna 'rsi_norm' no generada para validación.")
                 df_val['rsi_norm'] = 0.0

            initial_val_rows = len(df_val)
            df_val = df_val.dropna()
            dropped_val_rows = initial_val_rows - len(df_val)
            if dropped_val_rows > 0: logger.info(f"{dropped_val_rows} filas eliminadas de los datos de validación por NaNs post-normalización.")

        logger.info("Datos de entrenamiento y validación listos.")

    except DataError as e: # Ahora 'DataError' está definido
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
        model.learn(total_timesteps=optimization_timesteps, tb_log_name=f"trial_{trial.number}")
    except Exception as e:
        logger.error(f"Trial {trial.number} falló durante el entrenamiento: {e}", exc_info=True)
        return -np.inf # Indicar fallo a Optuna

    # 4. Evaluar en el Entorno de Validación
    logger.info(f"Evaluando Trial {trial.number} en datos de validación...")
    if df_val.empty:
        logger.error("df_val está vacío en 'objective'. No se puede evaluar.")
        # Podríamos devolver 0 o un valor malo. -inf indica fallo.
        return -np.inf

    try:
        val_env = TradingEnv(df_val, config)
    except ValueError as e:
        logger.error(f"Error al crear TradingEnv para validación en Trial {trial.number}: {e}")
        return -np.inf

    try:
        mean_reward, std_reward = evaluate_policy(model, val_env, n_eval_episodes=1, deterministic=True)
        logger.info(f"Trial {trial.number}: Mean Reward={mean_reward:.4f} +/- {std_reward:.4f}")

        if np.isnan(mean_reward) or np.isinf(mean_reward):
             logger.warning(f"Trial {trial.number} resultó en recompensa inválida ({mean_reward}). Devolviendo -inf.")
             return -np.inf

    except Exception as e:
        logger.error(f"Error durante la evaluación en Trial {trial.number}: {e}", exc_info=True)
        return -np.inf

    # 5. Reportar métrica a Optuna
    return mean_reward

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
            direction="maximize"
        )

        try:
            # Ejecutar optimización
            study.optimize(objective, n_trials=n_trials, timeout=None) # Sin límite de tiempo
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
        logger.info(f"  Valor (Mean Reward): {best_trial.value:.4f}")
        logger.info("  Mejores Hiperparámetros:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        best_hyperparameters = best_trial.params

        # --- Reentrenamiento Final con los Mejores Hiperparámetros ---
        logger.info("\n--- REENTRENANDO MODELO FINAL CON MEJORES HIPERPARÁMETROS ---")

        if df_train.empty or df_val.empty:
            logger.error("Los DataFrames de entrenamiento o validación están vacíos. No se puede reentrenar.")
            return

        df_final_train = pd.concat([df_train, df_val]).sort_index()
        df_final_train = df_final_train[~df_final_train.index.duplicated(keep='first')]
        df_final_train = df_final_train.dropna() # Asegurar limpieza final

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
            final_agent.create_model(final_env, best_hyperparameters)
            final_training_timesteps = config['rl_params'].get('total_timesteps', 5000000)
            final_agent.train(final_env, total_timesteps=final_training_timesteps)

            logger.info("--- MODELO FINAL ENTRENADO Y GUARDADO ---")
            logger.info(f"Modelo guardado en: {config['rl_params']['model_save_path']}")
            logger.info(f"Estadísticas de normalización (usadas) guardadas en: {config['rl_params']['norm_stats_path']}")

        except Exception as e:
            logger.critical(f"Fallo durante el reentrenamiento final: {e}", exc_info=True)

    finally:
        if processor:
            processor.shutdown()
            logger.info("RLProcessor (modo CSV) finalizado.")


if __name__ == "__main__":
    main()