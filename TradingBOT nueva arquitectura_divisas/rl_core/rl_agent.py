# rl_core/rl_agent.py

import os
import logging
import numpy as np
from typing import Dict, Any
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
# Corregir import si rl_environment está en el mismo directorio
from .rl_environment import TradingEnv

class RLAgent:
    """
    Envoltorio para el Agente de Stable-Baselines3 (PPO o DQN).
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config_full = config # Guardar config completa
        self.config_rl = config['rl_params']
        self.model = None
        self.algorithm = self.config_rl.get('algorithm', 'PPO').upper()

    def create_model(self, env: TradingEnv, hyperparameters: Dict[str, Any] = None):
         """Crea una nueva instancia del modelo con hiperparámetros dados."""
         vec_env = DummyVecEnv([lambda: env])
         
         # Usar hiperparámetros si se proporcionan, sino usar config.yaml
         # 'hp' contendrá los parámetros específicos del trial de Optuna o los defaults
         hp = hyperparameters if hyperparameters else self.config_rl

         self.logger.info(f"Creando Agente {self.algorithm} con hiperparámetros: {hp}")

         if self.algorithm == 'PPO':
             # --- REFACTORIZACIÓN: Separar kwargs base de los hiperparámetros ---
             
             # Kwargs base (fijos)
             base_kwargs = {
                 'policy': self.config_rl['policy'],
                 'env': vec_env,
                 'verbose': 0, # Menos verboso durante Optuna
                 'tensorboard_log': self.config_rl['tensorboard_log'],
                 'device': "auto"
             }

             # Hiperparámetros (extraídos de 'hp')
             # Usamos .get() con los valores por defecto de tu script original
             ppo_hyperparams = {
                 'n_steps': hp.get('ppo_n_steps', 2048),
                 'batch_size': hp.get('ppo_batch_size', 64),
                 'n_epochs': hp.get('ppo_n_epochs', 10),
                 'gamma': hp.get('gamma', 0.99),
                 'gae_lambda': hp.get('gae_lambda', 0.95),
                 'clip_range': hp.get('clip_range', 0.2),
                 'ent_coef': hp.get('ent_coef', 0.0),
                 'vf_coef': hp.get('vf_coef', 0.5),
                 'max_grad_norm': hp.get('max_grad_norm', 0.5),
                 'learning_rate': hp.get('learning_rate', 3e-4),
             }
             
             # Combinar los diccionarios
             model_kwargs = {**base_kwargs, **ppo_hyperparams}
             
             self.model = PPO(**model_kwargs)

         # ... (código para DQN si fuera necesario) ...

         else:
             raise ValueError(f"Algoritmo '{self.algorithm}' no soportado.")
         return self.model


    def train(self, env: TradingEnv, total_timesteps: int = None):
        """Entrena el modelo existente."""
        if self.model is None:
             # Si no se creó antes (ej. no se usa Optuna), crearlo con config.yaml
             self.create_model(env)

        # Usar timesteps pasados, o caer a los de config.yaml
        timesteps = total_timesteps if total_timesteps else self.config_rl.get('total_timesteps', 1000000)
        model_path = self.config_rl['model_save_path']

        self.logger.info(f"Iniciando entrenamiento por {timesteps} timesteps...")

        # --- ¡El entrenamiento ocurre aquí! ---
        # Usar verbose=1 para ver progreso si se entrena directamente
        self.model.learn(
            total_timesteps=timesteps,
            log_interval=1, # Para PPO, loggea cada 'n_steps'
            tb_log_name=f"{self.algorithm}_run" # Nombre para TensorBoard
        )

        self.model.save(model_path)
        self.logger.info(f"Entrenamiento completado. Modelo guardado en {model_path}")

    def load_model(self):
        """Carga un modelo entrenado."""
        model_path = self.config_rl['model_save_path']
        if not os.path.exists(model_path):
            self.logger.critical(f"No se encontró el modelo guardado en {model_path}")
            self.logger.critical("Asegúrate de ejecutar 'train_rl.py' primero.")
            raise FileNotFoundError(model_path)

        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path)
        # ... (código para DQN) ...
        else:
            raise ValueError(f"Algoritmo '{self.algorithm}' no soportado.")
        self.logger.info(f"Agente RL ({self.algorithm}) cargado desde {model_path}")

    def predict(self, state: np.ndarray) -> int:
        """Pide al agente entrenado que tome una decisión (acción)."""
        if self.model is None:
            self.logger.error("El modelo no está cargado. Llamando a load_model().")
            # Intento de autocorrección
            try:
                self.load_model()
            except Exception as e:
                self.logger.critical(f"Fallo al autocargar el modelo: {e}. Devolviendo 'Mantener'.")
                return 0 # Mantener

        action, _states = self.model.predict(state, deterministic=True)
        return int(action)