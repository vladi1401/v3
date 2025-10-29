# rl_core/rl_environment.py

import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List # Añadido List
from collections import deque

class TradingEnv(gym.Env):
    """
    Entorno de simulación de trading para RL con slippage y recompensa Sharpe.
    """

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.df = df # DataFrame completo

        # --- Definir nombres de columnas explícitamente ---
        features_config = config['features']
        
        # Obtener lista ordenada de columnas de features normalizadas
        self.norm_feature_col_names = self._get_norm_feature_names(features_config)
        self.expected_features_count = len(self.norm_feature_col_names)

        self.logger.debug(f"Environment Init: Expected feature columns (count={self.expected_features_count}): {self.norm_feature_col_names}")

        missing_cols = [col for col in self.norm_feature_col_names if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Columnas faltantes en el DataFrame del Env: {missing_cols}")
            self.logger.error(f"Columnas disponibles: {df.columns.tolist()}")
            raise ValueError(f"Faltan columnas ({missing_cols}) en DataFrame del Env.")

        self.features_df = df[self.norm_feature_col_names]
        self.prices = df[['close', 'open', 'high', 'low', 'atr']] # ATR no normalizado es útil

        self.config_rl = config['rl_params']; self.initial_balance = self.config_rl['balance_inicial']
        self.spread_cost_pips = self.config_rl['spread_cost_pips']; self.pip_value = config['trading']['pip_value']
        self.spread_cost = self.spread_cost_pips * self.pip_value; self.slippage_pips = self.config_rl.get('slippage_pips', 0.0)
        self.max_steps = len(self.df) - 1

        self.reward_type = self.config_rl.get('reward_type', 'pnl').lower()
        self.reward_window_size = self.config_rl.get('reward_history_window', 100)
        self.step_returns_history = deque(maxlen=self.reward_window_size)
        self.risk_free_rate_daily = (1 + self.config_rl.get('sharpe_risk_free_rate', 0.0))**(1/252) - 1

        self.action_space = gym.spaces.Discrete(3) # 0: Mantener, 1: Comprar, 2: Vender
        
        self.expected_state_size = self.expected_features_count + 1 # Features + Posición actual
        self.observation_shape = (self.expected_state_size,)
        self.logger.info(f"Observation space shape DEFINIDO como: {self.observation_shape}")

        bound = 100.0 # Límite para features normalizadas (z-score)
        try:
            # --- Forzar dtype explícitamente también en la definición ---
            self.observation_space = gym.spaces.Box(
                low=-bound, high=bound, shape=self.observation_shape, dtype=np.float32
            )
            self.logger.info(f"Espacio de observación gym.Box CREADO con shape={self.observation_space.shape}, dtype={self.observation_space.dtype}")
        except Exception as e:
             self.logger.critical(f"FALLO al crear gym.spaces.Box: {e}", exc_info=True)
             raise

        self._initialize_state()

    def _get_norm_feature_names(self, features_config: Dict[str, Any]) -> List[str]:
        """Obtiene la lista ordenada de nombres de features normalizadas."""
        cols_to_norm = features_config.get('cols_to_normalize', [])
        momentum_cols = [f'momentum_{w}' for w in features_config.get('momentum_windows', [])]
        
        norm_cols = [f"{col}_norm" for col in (cols_to_norm + momentum_cols)]
        
        # Añadir 'rsi_norm' si 'rsi_window' está definida
        if 'rsi_window' in features_config:
            norm_cols.append('rsi_norm')
            
        return sorted(list(set(norm_cols))) # Ordenar y asegurar unicidad

    def _initialize_state(self):
        """Resetea el estado interno del entorno."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.last_equity = self.initial_balance
        self.position = 0 # -1: Venta, 0: Plano, 1: Compra
        self.entry_price = 0.0
        self.step_returns_history.clear()

    def reset(self, seed=None, options=None):
        """Resetea el entorno y devuelve la observación inicial."""
        super().reset(seed=seed)
        self._initialize_state()
        
        state = self._get_state()
        info = self._get_info()

        # --- Verificación en Reset ---
        if state is None: 
            raise ValueError("reset() produjo un estado None")
        if state.shape != self.observation_shape:
             self.logger.critical(f"Shape Mismatch GRAVE en reset! Esperado {self.observation_shape}, Obtenido {state.shape}")
             raise ValueError(f"Shape Mismatch GRAVE en reset! Esperado {self.observation_shape}, Obtenido {state.shape}")
        if state.dtype != np.float32:
             self.logger.warning(f"Dtype Mismatch en reset! Esperado {np.float32}, Obtenido {state.dtype}. Forzando conversión.")
             state = state.astype(np.float32)

        return state, info

    def _get_state(self) -> np.ndarray:
        """Construye el vector de estado actual, asegurando el shape y dtype correctos."""
        try:
            if self.current_step >= len(self.df):
                 self.logger.warning(f"_get_state step {self.current_step} >= len(df) {len(self.df)}. Devolviendo ceros.")
                 return np.zeros(self.observation_shape, dtype=np.float32)

            # Seleccionar fila y columnas EXACTAS y ORDENADAS
            current_row = self.df.iloc[self.current_step]
            feature_values = current_row[self.norm_feature_col_names].values.astype(np.float32)

            if feature_values.shape[0] != self.expected_features_count:
                raise ValueError(f"Shape de features incorrecto! Esperado {self.expected_features_count}, Obtenido {feature_values.shape[0]}")

            # Crear estado (features + posición)
            current_pos_float = np.float32(self.position)
            state_constructed = np.append(feature_values, current_pos_float).astype(np.float32)
            
            # Verificar shape y dtype finales
            if state_constructed.shape != self.observation_shape:
                raise ValueError(f"Shape final incorrecto. Esperado {self.observation_shape}, obtenido {state_constructed.shape}")
            if state_constructed.dtype != np.float32:
                 raise TypeError(f"Dtype final incorrecto. Esperado {np.float32}, obtenido {state_constructed.dtype}")

            return state_constructed

        except IndexError:
             self.logger.error(f"IndexError en _get_state step {self.current_step}. Max steps: {self.max_steps}.")
             return np.zeros(self.observation_shape, dtype=np.float32)
        except KeyError as e:
             self.logger.error(f"KeyError en _get_state: {e}. Step: {self.current_step}.")
             return np.zeros(self.observation_shape, dtype=np.float32)
        except Exception as e:
             self.logger.error(f"Error inesperado en _get_state: {e}", exc_info=True)
             return np.zeros(self.observation_shape, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Devuelve información de debugging sobre el estado actual."""
        return {
            "step": self.current_step, 
            "balance": self.balance, 
            "equity": self.equity, 
            "position": self.position, 
            "entry_price": self.entry_price
        }

    def _apply_slippage(self, price: float, action_type: str) -> float:
        """Aplica un deslizamiento aleatorio al precio de ejecución."""
        if self.slippage_pips <= 0: 
            return price
            
        slippage_amount = np.random.uniform(0, self.slippage_pips) * self.pip_value
        
        if action_type == 'buy': # Comprar (abrir o cerrar venta) -> Peor precio (más alto)
            return price + slippage_amount
        elif action_type == 'sell': # Vender (abrir o cerrar compra) -> Peor precio (más bajo)
            return price - slippage_amount
        return price

    def _calculate_reward(self, equity_change: float) -> float:
        """Calcula la recompensa basada en la configuración (pnl, sharpe, etc.)."""
        # Calcular retorno % del paso
        step_return = equity_change / self.last_equity if abs(self.last_equity) > 1e-9 else 0
        self.step_returns_history.append(step_return)
        
        if self.reward_type == "pnl": 
            return equity_change

        # Si no hay suficiente historial, devolver PnL (quizás penalizado)
        if len(self.step_returns_history) < self.reward_window_size: 
            return equity_change * 0.1 # PnL pequeño para incentivar supervivencia

        returns_array = np.array(self.step_returns_history)
        std_dev = np.std(returns_array)
        epsilon = 1e-9 # Evitar división por cero

        if self.reward_type == "risk_adjusted": 
            # Recompensa ajustada por volatilidad (Sortino simple)
            return equity_change / (std_dev + epsilon)
            
        elif self.reward_type == "sharpe":
             # Calcular Sharpe Ratio de la ventana
             mean_return = np.mean(returns_array)
             sharpe_ratio = (mean_return - self.risk_free_rate_daily) / (std_dev + epsilon)
             # Devolver el sharpe ratio (escalado y recortado)
             return np.clip(sharpe_ratio * 100, -10, 10) # Clip para evitar recompensas extremas
             
        else: # Default a PnL
            return equity_change

    def step(self, action: int):
        """Ejecuta un paso en el entorno."""
        
        current_prices = self.prices.iloc[self.current_step]
        # El precio de apertura del *siguiente* paso es el precio de ejecución
        next_step_idx = min(self.current_step + 1, self.max_steps)
        next_open_price = self.prices.iloc[next_step_idx]['open']
        
        terminated = self.current_step >= self.max_steps - 1
        realized_pnl = 0.0

        # === Lógica de Acción ===
        # action 0: MANTENER
        # action 1: COMPRAR
        # action 2: VENDER
        
        if action == 1: # === COMPRAR ===
            if self.position <= 0: # Si estamos planos (0) o en venta (-1)
                if self.position == -1: # Cerrar venta
                    exec_price_close = self._apply_slippage(next_open_price, 'buy') # Precio de cierre (buy)
                    pnl = (self.entry_price - exec_price_close) - self.spread_cost # PnL de venta
                    realized_pnl += pnl
                    self.balance += pnl
                
                # Abrir compra
                self.position = 1
                exec_price_open = self._apply_slippage(next_open_price, 'buy') # Precio de apertura (buy)
                self.entry_price = exec_price_open + self.spread_cost # Precio de entrada incluye spread
        
        elif action == 2: # === VENDER ===
            if self.position >= 0: # Si estamos planos (0) o en compra (1)
                if self.position == 1: # Cerrar compra
                    exec_price_close = self._apply_slippage(next_open_price, 'sell') # Precio de cierre (sell)
                    pnl = (exec_price_close - self.entry_price) - self.spread_cost # PnL de compra
                    realized_pnl += pnl
                    self.balance += pnl
                
                # Abrir venta
                self.position = -1
                exec_price_open = self._apply_slippage(next_open_price, 'sell') # Precio de apertura (sell)
                self.entry_price = exec_price_open - self.spread_cost # Precio de entrada incluye spread

        # === Cálculo de Equity y Recompensa ===
        
        current_close_price = current_prices['close']
        unrealized_pnl = 0.0
        
        if self.position == 1: # Compra abierta
            unrealized_pnl = (current_close_price - self.entry_price)
        elif self.position == -1: # Venta abierta
            unrealized_pnl = (self.entry_price - current_close_price)
            
        self.equity = self.balance + unrealized_pnl
        
        # El cambio en equity (incluyendo PnL realizado y no realizado) es la base de la recompensa
        equity_change_this_step = self.equity - self.last_equity
        reward = self._calculate_reward(equity_change_this_step)
        
        self.last_equity = self.equity
        self.current_step += 1
        
        # --- Obtener nuevo estado e info ---
        state = self._get_state()
        info = self._get_info()

        # Cierre forzado al final del episodio
        if terminated and self.position != 0: 
             final_pnl = unrealized_pnl - self.spread_cost # Pagar spread final
             self.balance += final_pnl
             self.equity = self.balance
             info['balance'] = self.balance
             info['equity'] = self.equity
             self.position = 0

        # --- Verificación final de estado ---
        if state is None: 
            raise ValueError(f"step() produjo un estado None en step {self.current_step-1}")
        if state.shape != self.observation_shape:
             self.logger.critical(f"Shape Mismatch GRAVE en step return! Esperado {self.observation_shape}, Obtenido {state.shape}")
             raise ValueError(f"Shape Mismatch GRAVE en step return! Esperado {self.observation_shape}, Obtenido {state.shape}")
        if state.dtype != np.float32:
             self.logger.warning(f"Dtype Mismatch en step return! Esperado {np.float32}, Obtenido {state.dtype}. Forzando conversión.")
             state = state.astype(np.float32)

        # truncated = False (no usamos max_episode_steps de Gymnasium)
        return state, reward, terminated, False, info