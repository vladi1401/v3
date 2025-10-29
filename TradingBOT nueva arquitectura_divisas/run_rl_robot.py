# run_rl_robot.py

import sys
import os
import time
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

# --- 1. Configuración de Path y Logging ---
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

try:
    from pyrobot.utils import setup_logging
    from pyrobot.broker import PyRobot
    from pyrobot.exceptions import ConnectionError, OrderError, DataError

    # --- ARQUITECTURA: Importar LiveProcessor refactorizado ---
    from rl_core.live_processor import LiveProcessor
    from rl_core.rl_agent import RLAgent

    import MetaTrader5 as mt5
    import talib as ta
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)

# --- Clase Principal del Bot ---
class LiveBot:
    """Orquesta el bot de RL en vivo con gestión de riesgo profesional."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_cfg = config['trading']
        self.risk_cfg = config['risk_management']
        self.feature_cfg = config['features']
        self.rl_cfg = config['rl_params']

        # --- Validar Configuración Esencial ---
        # (Sin cambios)
        required_keys = ['ticker', 'interval_mt5', 'pip_value']
        if not all(key in self.bot_cfg for key in required_keys):
            raise ValueError("Faltan claves esenciales ('ticker', 'interval_mt5', 'pip_value') en la sección 'trading' de config.yaml")
        required_risk_keys = ['risk_per_trade_percent', 'atr_period_sl_tp', 'sl_atr_multiplier', 'tp_atr_multiplier']
        if not all(key in self.risk_cfg for key in required_risk_keys):
             raise ValueError("Faltan claves esenciales en la sección 'risk_management' de config.yaml")


        # --- SEGURIDAD: Validar y Cargar Credenciales desde Entorno ---
        # (Sin cambios)
        try:
            self.login = int(os.environ['MT5_LOGIN'])
            self.password = os.environ['MT5_PASS']
            self.server = os.environ['MT5_SERVER']
            logger.info(f"Credenciales cargadas desde variables de entorno (Login: {self.login}).")
        except KeyError as e:
            logger.critical(f"¡ERROR DE SEGURIDAD! Variable de entorno no encontrada: {e}")
            logger.critical("Por favor, configura MT5_LOGIN, MT5_PASS, y MT5_SERVER antes de ejecutar el bot.")
            sys.exit(1)
        except (ValueError, TypeError):
            logger.critical("¡ERROR! MT5_LOGIN debe ser un número entero.")
            sys.exit(1)

        # --- Cargar Agente ---
        # (Sin cambios)
        self.agent = RLAgent(config)
        self.agent.load_model() # Cargar el modelo entrenado

        # --- Conectar Broker ---
        # (Sin cambios)
        risk_percent = self.risk_cfg.get('risk_per_trade_percent', 0.5)
        logger.info(f"Usando riesgo por operación: {risk_percent}%")

        self.broker = PyRobot(
            self.login, self.password, self.server,
            risk_percent=risk_percent,
            atr_period=self.risk_cfg['atr_period_sl_tp'],
            sl_atr_mult=self.risk_cfg['sl_atr_multiplier'],
            tp_atr_mult=self.risk_cfg['tp_atr_multiplier'],
            pip_value=self.bot_cfg['pip_value']
        )
        self.broker.initial_connection() # Conexión inicial robusta

        # --- Crear Procesador en Vivo ---
        # (Debe hacerse DESPUÉS de conectar el broker y cargar el agente)
        self.processor = LiveProcessor(config) # Carga norm_stats.json
        self.ticker = self.bot_cfg['ticker']

        # --- Estado Interno ---
        # (Sin cambios)
        self.current_position = 0 # -1 Venta, 0 Plano, 1 Compra
        self.last_time = None

        # --- Llenar Buffer Inicial del Procesador ---
        # (Sin cambios)
        self.processor.prime_buffer()
        self.sync_state() # Sincronizar posición con broker

    def sync_state(self):
        """Sincroniza la posición actual con el broker."""
        # (Sin cambios)
        try:
            positions = self.broker.get_portfolio_pos_time()
            if positions and self.ticker in positions:
                pos_type = positions[self.ticker]['PosType']
                new_position = 1 if pos_type == 1 else -1
                if new_position != self.current_position:
                     logger.info(f"Estado sincronizado. Posición actual: {new_position} (cambio detectado desde {self.current_position})")
                     self.current_position = new_position
                else:
                     logger.debug(f"Estado sincronizado. Posición actual: {self.current_position}")

            else:
                if self.current_position != 0:
                     logger.info(f"Estado sincronizado. Sin posición abierta (cambio detectado desde {self.current_position}).")
                     self.current_position = 0
                else:
                     logger.debug("Estado sincronizado. Sin posición abierta.")

        except ConnectionError as e:
             logger.error(f"Error de conexión al sincronizar estado: {e}")
             raise
        except Exception as e:
            logger.error(f"Error inesperado al sincronizar estado: {e}", exc_info=True)
            self.current_position = 0 # Asumir plano por seguridad

    def run(self):
        """Bucle principal del bot."""
        logger.info(f"--- BOT RL (CSV Mode Training) INICIADO --- Operando {self.ticker} ---")
        try:
            while True:
                current_dt = datetime.now()
                actual_minute = current_dt.minute

                # Ejecutar lógica solo al inicio de un nuevo minuto
                if actual_minute != self.last_time:
                    cycle_start_time = time.time()
                    self.last_time = actual_minute
                    logger.info(f"--- Nuevo Ciclo Minuto: {current_dt.strftime('%Y-%m-%d %H:%M:%S')} ---")

                    # Comprobar conexión y mercado
                    try:
                        is_connected = self.broker.is_connected()
                        if not is_connected:
                            logger.warning("Conexión perdida, _connect_with_backoff se activará en la próxima llamada.")
                            # No continuar este ciclo si no hay conexión inicial
                            time.sleep(5) # Esperar antes del siguiente intento
                            continue

                        market_open = self.broker.market_open # Usa @ensure_connection
                        liquidity_hours = self.broker.liquidity_hours # Usa @ensure_connection

                    except ConnectionError as e:
                         logger.error(f"Fallo de reconexión en chequeo de estado: {e}. Reintentando en el próximo ciclo.")
                         time.sleep(10) # Espera más larga si la reconexión falla
                         continue
                    except Exception as e:
                         logger.error(f"Error inesperado chequeando estado del mercado: {e}", exc_info=True)
                         time.sleep(5)
                         continue


                    if market_open and liquidity_hours:
                        logger.debug("Mercado abierto y en horas de liquidez.")

                        # --- Lógica Principal del Ciclo ---
                        try:
                            # 1. Sincronizar estado (por si cambió manualmente o por SL/TP)
                            self.sync_state()

                            # 2. Actualizar Datos y Obtener Estado para el Agente
                            state, current_atr = self.processor.update_and_get_state(self.current_position)

                            if state is None:
                                logger.info("No hay nueva vela o hubo error en el procesador. Esperando al próximo minuto.")
                                continue # Saltar el resto del ciclo

                            # Log del estado (opcional, puede ser muy verboso)
                            # logger.debug(f"Estado obtenido (shape {state.shape}): {state}")
                            logger.debug(f"ATR actual (no norm): {current_atr:.5f}")


                            # 3. Pedir Acción al Agente RL
                            action = self.agent.predict(state) # 0=Hold, 1=Buy, 2=Sell
                            action_map = {0: "MANTENER", 1: "COMPRAR", 2: "VENDER"}
                            logger.info(f"Acción decidida por el agente: {action} ({action_map[action]})")

                            # 4. Ejecutar Acción en el Broker
                            self.execute_action(action)

                        # --- Manejo de Errores Específicos del Ciclo ---
                        except (OrderError, DataError) as e:
                             logger.error(f"Error operativo en el ciclo: {e}", exc_info=False)
                        except ConnectionError as e:
                             logger.error(f"Error de conexión durante el ciclo: {e}. Se reintentará.")
                             # El decorador @ensure_connection manejará reintentos en la próxima llamada
                        except Exception as e:
                             logger.error(f"Error inesperado en el ciclo principal: {e}", exc_info=True)
                        # --- Fin Lógica Principal ---

                        cycle_duration = time.time() - cycle_start_time
                        logger.info(f"Ciclo completado en: {cycle_duration:.2f}s")

                    else:
                        # Mercado cerrado o fuera de horas
                        logger.info(f"Mercado cerrado o fuera de horas de liquidez. Esperando... (MarketOpen={market_open}, Liquidity={liquidity_hours})")
                        # Intentar cerrar posiciones abiertas si el mercado cierra (opcional)
                        # if not market_open and self.current_position != 0:
                        #    logger.warning("Mercado cerrado con posición abierta. Intentando cerrar...")
                        #    try:
                        #        self.broker.close_position_by_ticker(self.ticker)
                        #        self.current_position = 0
                        #    except Exception as e_close:
                        #        logger.error(f"No se pudo cerrar la posición al cierre del mercado: {e_close}")


                # Pausa corta para no saturar la CPU
                time.sleep(1) # Chequea cada segundo si el minuto ha cambiado

        except KeyboardInterrupt:
            logger.info("--- Interrupción de teclado detectada. Apagando bot... ---")
        except ConnectionError as e:
            logger.critical(f"Error de conexión irrecuperable en bucle principal: {e}. Apagando...", exc_info=True)
        except Exception as e:
            logger.critical(f"¡¡ERROR INESPERADO Y FATAL en bucle principal!!: {e}", exc_info=True)
        finally:
            logger.info("Iniciando secuencia de apagado...")
            if hasattr(self, 'broker') and self.broker:
                self.broker.shutdown()
            else:
                 # Asegurarse de que MT5 se cierre si el broker no se inicializó
                 try: mt5.shutdown()
                 except: pass
            logger.info("--- BOT RL APAGADO ---")

    def execute_action(self, action: int):
        """Ejecuta la acción decidida por el agente."""
        # (Sin cambios lógicos, pero añadiendo verificación explícita de posición antes de actuar)
        current_pos_before_action = self.current_position # Guardar estado antes

        try:
            if action == 1: # === ACCIÓN: COMPRAR ===
                if current_pos_before_action == 0:
                    logger.info("Ejecutando: Abrir COMPRA")
                    self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_BUY)
                    # No actualizamos self.current_position aquí, sync_state() lo hará
                elif current_pos_before_action == -1:
                    logger.info("Ejecutando: Cerrar VENTA y abrir COMPRA")
                    self.broker.close_position_by_ticker(self.ticker)
                    time.sleep(0.5) # Pausa
                    # Re-sincronizar ANTES de abrir la nueva orden
                    self.sync_state()
                    if self.current_position == 0: # Verificar que realmente se cerró
                         logger.info("Venta cerrada, procediendo a abrir COMPRA...")
                         self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_BUY)
                    else:
                         logger.warning("Fallo al cerrar VENTA antes de abrir COMPRA. Acción abortada.")
                else: # current_pos == 1
                    logger.info("Acción COMPRAR, pero ya estamos comprados. Manteniendo.")

            elif action == 2: # === ACCIÓN: VENDER ===
                if current_pos_before_action == 0:
                    logger.info("Ejecutando: Abrir VENTA")
                    self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_SELL)
                elif current_pos_before_action == 1:
                    logger.info("Ejecutando: Cerrar COMPRA y abrir VENTA")
                    self.broker.close_position_by_ticker(self.ticker)
                    time.sleep(0.5) # Pausa
                    self.sync_state() # Re-sincronizar
                    if self.current_position == 0: # Verificar cierre
                         logger.info("Compra cerrada, procediendo a abrir VENTA...")
                         self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_SELL)
                    else:
                         logger.warning("Fallo al cerrar COMPRA antes de abrir VENTA. Acción abortada.")
                else: # current_pos == -1
                    logger.info("Acción VENDER, pero ya estamos vendidos. Manteniendo.")

            else: # === ACCIÓN: MANTENER (action == 0) ===
                logger.info(f"Acción MANTENER. Posición actual sin cambios: {current_pos_before_action}")

            # Sincronización final después de la acción (opcional, pero bueno para logs)
            # time.sleep(0.5) # Esperar un poco a que la orden se procese
            # self.sync_state()

        # Captura de errores (sin cambios)
        except OrderError as e:
            logger.error(f"Error de Orden al ejecutar acción {action}: {e}", exc_info=False)
        except DataError as e:
             logger.error(f"Error de Datos al ejecutar acción {action}: {e}", exc_info=False)
        except ConnectionError as e:
             logger.error(f"Error de Conexión al ejecutar acción {action}: {e}", exc_info=False)
             raise # Re-lanzar para que el bucle principal maneje la reconexión
        except Exception as e:
             logger.error(f"Error inesperado en execute_action({action}): {e}", exc_info=True)


if __name__ == "__main__":
    try:
        # Cargar config (sin cambios)
        with open(os.path.join(dir_path, 'config.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        # Loggear error antes de salir
        logging.basicConfig() # Configuración básica si setup_logging falló o no se llamó
        logging.critical(f"Error fatal al leer 'config.yaml': {e}", exc_info=True)
        sys.exit(1)

    # Inicializar y correr el bot (sin cambios)
    bot = None # Definir fuera del try para usar en finally
    try:
         bot = LiveBot(config)
         bot.run()
    except Exception as e:
         logging.critical(f"Fallo fatal al inicializar o ejecutar el bot: {e}", exc_info=True)
         # Asegurarse de cerrar MT5 incluso si el bot falla muy temprano
         if bot and hasattr(bot, 'broker') and bot.broker:
              bot.broker.shutdown()
         else:
              try: mt5.shutdown()
              except: pass
         sys.exit(1)