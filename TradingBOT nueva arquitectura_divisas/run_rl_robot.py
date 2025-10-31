# run_rl_robot.py

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, time as dt_time, timezone
from typing import Dict, Any, Tuple

# --- 1. Configuración de Path y Logging ---
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

try:
    from pyrobot.utils import setup_logging
    from pyrobot.broker import PyRobot
    from pyrobot.exceptions import ConnectionError, OrderError, DataError
    # --- ARQUITECTURA: Importar LiveProcessor CORREGIDO ---
    from rl_core.live_processor import LiveProcessor
    from rl_core.rl_agent import RLAgent
    
    # --- NUEVOS MÓDULOS DE GESTIÓN Y ALERTAS ---
    from pyrobot.alerts import send_telegram_alert
    from pyrobot.news_filter import is_news_embargo_active

    import MetaTrader5 as mt5
    import talib as ta
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)

# --- Clase Principal del Bot ---
class LiveBot:
    """
    Orquesta el bot de RL en vivo con gestión de riesgo de nivel de fondeo.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_cfg = config['trading']
        self.risk_cfg = config['risk_management']
        self.feature_cfg = config['features']
        self.rl_cfg = config['rl_params']
        self.firewall_cfg = config['funding_challenge_firewall']

        # --- Validar Configuración Esencial ---
        # (Sin cambios)
        required_keys = ['ticker', 'interval_mt5', 'pip_value']
        if not all(key in self.bot_cfg for key in required_keys):
            raise ValueError("Faltan claves esenciales ('ticker', 'interval_mt5', 'pip_value') en la sección 'trading' de config.yaml")
        required_risk_keys = ['risk_per_trade_percent', 'atr_period_sl_tp', 'sl_atr_multiplier', 'tp_atr_multiplier']
        if not all(key in self.risk_cfg for key in required_risk_keys):
             raise ValueError("Faltan claves esenciales en la sección 'risk_management' de config.yaml")


        # --- Cargar Credenciales desde Entorno ---
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

        # --- Conectar Broker ---
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
        self.broker.initial_connection()
        logger.info("Broker conectado.")

        # --- Cargar Agente ---
        self.agent = RLAgent(config)
        self.agent.load_model()
        logger.info("Agente de RL cargado.")

        # --- Crear Procesador en Vivo ---
        # (Debe hacerse DESPUÉS de conectar el broker y cargar el agente)
        self.processor = LiveProcessor(config) # Carga norm_stats.json
        self.ticker = self.bot_cfg['ticker']

        # --- Llenar Buffer Inicial del Procesador ---
        self.processor.prime_buffer()
        logger.info("Buffer de datos en vivo 'primeado' (calentado).")

        # --- Inicializar Estado Interno y Cortafuegos (Firewall) ---
        self.current_position = 0 # -1 Venta, 0 Plano, 1 Compra
        self.last_time = None
        self.sync_state() # Sincronizar posición con broker

        # --- Lógica del Cortafuegos (Firewall) ---
        try:
            account_info = self.broker.get_account_info()
            self.initial_balance = account_info.balance
            self.equity_high_watermark = account_info.equity
            
            # Límites de Pérdida
            self.daily_loss_limit_pct = self.firewall_cfg['daily_loss_limit_percent'] / 100.0
            self.max_drawdown_limit_pct = self.firewall_cfg['max_drawdown_limit_percent'] / 100.0
            
            # Colchón de seguridad
            cushion_pct = self.firewall_cfg.get('drawdown_cushion_percent', 1.0) / 100.0
            
            # Límite de Pérdida Diaria (basado en 5% del balance inicial, regla común de FTMO)
            self.daily_loss_limit_usd = self.initial_balance * self.daily_loss_limit_pct
            self.daily_equity_stop_level = self.initial_balance - self.daily_loss_limit_usd
            # Aplicar colchón de seguridad
            self.daily_equity_stop_level_safe = self.daily_equity_stop_level + (self.initial_balance * cushion_pct)

            # Límite de Drawdown Total (Trailing)
            self.max_drawdown_stop_level = self.equity_high_watermark * (1 - self.max_drawdown_limit_pct)
            # Aplicar colchón de seguridad
            self.max_drawdown_stop_level_safe = self.max_drawdown_stop_level + (self.initial_balance * cushion_pct)

            self.trading_halted_today = False
            self.trading_halted_permanently = False
            self.last_day_checked = datetime.now(timezone.utc).day

            logger.info("--- CORTAFUEGOS (FIREWALL) INICIALIZADO ---")
            logger.info(f"Balance Inicial: ${self.initial_balance:,.2f}")
            logger.info(f"Equity HWM Inicial: ${self.equity_high_watermark:,.2f}")
            logger.info(f"Límite de Pérdida Diaria (Equity): < ${self.daily_equity_stop_level:,.2f} (Seguridad: ${self.daily_equity_stop_level_safe:,.2f})")
            logger.info(f"Límite de Drawdown Máx. (Equity): < ${self.max_drawdown_stop_level:,.2f} (Seguridad: ${self.max_drawdown_stop_level_safe:,.2f})")
            
            send_telegram_alert(f"✅ BOT INICIADO. Firewall Activado. Equity: ${self.equity_high_watermark:,.2f}. Límite DD Máx: ${self.max_drawdown_stop_level_safe:,.2f}", level="info")

        except Exception as e:
            logger.critical(f"¡Error fatal al inicializar el Cortafuegos!: {e}", exc_info=True)
            raise

    def sync_state(self):
        """Sincroniza la posición actual con el broker."""
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
            self.current_position = 0

    # --- FUNCIONES DEL CORTAFUEGOS (FIREWALL) ---

    def liquidate_and_halt_daily(self):
        """Liquida todas las posiciones y detiene el trading por hoy."""
        msg = f"¡¡VIOLACIÓN DE PÉRDIDA DIARIA!! Equity por debajo de ${self.daily_equity_stop_level_safe:,.2f}. Liquidando posiciones y deteniendo trading por hoy."
        logger.critical(msg)
        send_telegram_alert(f"🛑 HALT DIARIO: {msg}", level="critical")
        
        self.trading_halted_today = True
        try:
            self.broker.close_all_positions()
        except Exception as e:
            logger.error(f"Error al liquidar posiciones por halt diario: {e}")
            send_telegram_alert(f"🚨 ERROR: Fallo al liquidar posiciones en HALT DIARIO: {e}", level="error")

    def liquidate_and_halt_permanently(self):
        """Liquida todas las posiciones y detiene el bot permanentemente."""
        msg = f"¡¡VIOLACIÓN DE DRAWDOWN MÁXIMO!! Equity por debajo de ${self.max_drawdown_stop_level_safe:,.2f}. Liquidando TODO y apagando bot."
        logger.critical(msg)
        send_telegram_alert(f"🔥🔥 HALT PERMANENTE: {msg}. EL BOT SE APAGARÁ.", level="critical")
        
        self.trading_halted_permanently = True
        try:
            self.broker.close_all_positions()
        except Exception as e:
            logger.error(f"Error al liquidar posiciones por halt permanente: {e}")
            send_telegram_alert(f"🚨 ERROR: Fallo al liquidar posiciones en HALT PERMANENTE: {e}", level="error")

    def check_firewall(self, current_dt_utc: datetime):
        """
        Comprueba las reglas del cortafuegos de la cuenta.
        Devuelve True si es seguro operar, False si no.
        """
        # --- 1. Resetear el límite diario a medianoche (hora del broker/cuenta) ---
        # Usamos UTC para consistencia
        if current_dt_utc.day != self.last_day_checked:
            logger.info(f"Nuevo día de trading ({current_dt_utc.date()}). Reseteando límite de pérdida diaria.")
            send_telegram_alert(f"☀️ Nuevo día de trading. Reseteando límite de pérdida diaria.", level="info")
            self.trading_halted_today = False
            self.last_day_checked = current_dt_utc.day
            
            # IMPORTANTE: Recalcular el límite de pérdida diaria si la regla
            # es 5% del balance/equity del DÍA ANTERIOR.
            # La regla actual (5% del balance INICIAL) no necesita recalculo.
            # new_balance = self.broker.get_balance()
            # self.daily_equity_stop_level = new_balance - (new_balance * self.daily_loss_limit_pct)
            # logger.info(f"Nuevo límite de pérdida diaria (Equity): < ${self.daily_equity_stop_level:,.2f}")

        # --- 2. Comprobar si estamos en halt ---
        if self.trading_halted_permanently:
            return False # Parada total
        if self.trading_halted_today:
            return False # Parada diaria

        # --- 3. Obtener Equity y comprobar límites ---
        try:
            current_equity = self.broker.get_equity()

            # 3a. Comprobar Límite de Drawdown Máximo (Trailing)
            # Actualizar primero el High-Water Mark
            if current_equity > self.equity_high_watermark:
                self.equity_high_watermark = current_equity
                self.max_drawdown_stop_level = self.equity_high_watermark * (1 - self.max_drawdown_limit_pct)
                # Recalcular colchón
                cushion_amount = self.initial_balance * self.firewall_cfg.get('drawdown_cushion_percent', 1.0) / 100.0
                self.max_drawdown_stop_level_safe = self.max_drawdown_stop_level + cushion_amount
                
                logger.info(f"Nuevo Equity High-Water Mark: ${self.equity_high_watermark:,.2f}. Nuevo Límite DD Máx (Seguridad): ${self.max_drawdown_stop_level_safe:,.2f}")

            if current_equity < self.max_drawdown_stop_level_safe:
                self.liquidate_and_halt_permanently()
                return False

            # 3b. Comprobar Límite de Pérdida Diaria (Fijo)
            if current_equity < self.daily_equity_stop_level_safe:
                self.liquidate_and_halt_daily()
                return False

            # Si pasa todos los chequeos, es seguro operar.
            return True

        except (DataError, ConnectionError) as e:
            logger.error(f"Error de conexión/datos en check_firewall: {e}. Pausando ciclo.")
            return False # No es seguro operar si no podemos verificar el equity
        except Exception as e:
            logger.critical(f"Error inesperado en check_firewall: {e}", exc_info=True)
            return False # No es seguro operar

    # --- BUCLE PRINCIPAL DE TRADING ---

    def run_trading_cycle(self, current_dt_utc: datetime):
        """
        Ejecuta un ciclo de lógica de trading (obtener estado, decidir, actuar).
        """
        try:
            # 1. Comprobar embargo de noticias
            if is_news_embargo_active(self.ticker, current_dt_utc):
                logger.info("Embargo de noticias activo. Saltando ciclo de trading.")
                # Asegurarse de cerrar posiciones si las reglas lo exigen
                if self.current_position != 0:
                     logger.warning("Cerrando posición actual debido a embargo de noticias.")
                     send_telegram_alert("⚠️ Cerrando posición por embargo de noticias inminente.", level="warning")
                     self.broker.close_position_by_ticker(self.ticker)
                     self.sync_state()
                return

            # 2. Sincronizar estado (por si SL/TP saltó)
            self.sync_state()

            # 3. Actualizar Datos y Obtener Estado para el Agente
            state = self.processor.update_and_get_state(self.current_position)

            if state is None:
                logger.debug("No hay nueva vela o hubo error en el procesador. Esperando.")
                return # Saltar el resto del ciclo

            # 4. Pedir Acción al Agente RL
            action = self.agent.predict(state) # 0=Hold, 1=Buy, 2=Sell
            action_map = {0: "MANTENER", 1: "COMPRAR", 2: "VENDER"}
            logger.info(f"Acción decidida por el agente: {action} ({action_map[action]})")

            # 5. Ejecutar Acción en el Broker
            self.execute_action(action)

        except (OrderError, DataError) as e:
             logger.error(f"Error operativo en el ciclo: {e}", exc_info=False)
             send_telegram_alert(f"🚨 ERROR: Error operativo en el ciclo: {e}", level="error")
        except ConnectionError as e:
             logger.error(f"Error de conexión durante el ciclo: {e}. Se reintentará.")
             # El decorador @ensure_connection manejará reintentos
        except Exception as e:
             logger.error(f"Error inesperado en el ciclo principal: {e}", exc_info=True)
             send_telegram_alert(f"🚨 ERROR: Error inesperado en el ciclo: {e}", level="error")

    def run(self):
        """Bucle principal del bot."""
        logger.info(f"--- BOT RL INICIADO --- Operando {self.ticker} ---")
        try:
            while True:
                current_dt_utc = datetime.now(timezone.utc)
                actual_minute = current_dt_utc.minute

                # --- 1. Lógica de Reseteo Diario del Cortafuegos ---
                # (Se maneja dentro de check_firewall)

                # --- 2. Lógica de Ciclo por Minuto ---
                if actual_minute != self.last_time:
                    cycle_start_time = time.time()
                    self.last_time = actual_minute
                    logger.info(f"--- Nuevo Ciclo Minuto: {current_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

                    # --- 3. Comprobar Cortafuegos (Firewall) ---
                    if not self.check_firewall(current_dt_utc):
                        if self.trading_halted_permanently:
                            logger.critical("HALT PERMANENTE. Apagando el bot.")
                            break # Salir del bucle while True
                        if self.trading_halted_today:
                            logger.info("Halt diario activo. Esperando al próximo día.")
                        
                        time.sleep(10) # Pausa más larga si el firewall está activo o falló
                        continue

                    # --- 4. Comprobar Conexión y Mercado ---
                    try:
                        is_connected = self.broker.is_connected()
                        if not is_connected:
                            logger.warning("Conexión perdida, _connect_with_backoff se activará.")
                            time.sleep(5)
                            continue

                        market_open = self.broker.market_open
                        liquidity_hours = self.broker.liquidity_hours

                    except ConnectionError as e:
                         logger.error(f"Fallo de reconexión en chequeo de estado: {e}. Reintentando.")
                         time.sleep(10)
                         continue
                    except Exception as e:
                         logger.error(f"Error inesperado chequeando estado del mercado: {e}", exc_info=True)
                         time.sleep(5)
                         continue
                    
                    # --- 5. Ejecutar Lógica de Trading ---
                    if market_open and liquidity_hours:
                        logger.debug("Mercado abierto y en horas de liquidez.")
                        self.run_trading_cycle(current_dt_utc) # Lógica refactorizada
                    else:
                        logger.info(f"Mercado cerrado o fuera de horas. (MarketOpen={market_open}, Liquidity={liquidity_hours})")

                    cycle_duration = time.time() - cycle_start_time
                    logger.info(f"Ciclo completado en: {cycle_duration:.2f}s")

                # Pausa corta para no saturar la CPU
                time.sleep(1) # Chequea cada segundo si el minuto ha cambiado

        except KeyboardInterrupt:
            logger.info("--- Interrupción de teclado detectada. Apagando bot... ---")
        except ConnectionError as e:
            logger.critical(f"Error de conexión irrecuperable: {e}. Apagando...", exc_info=True)
        except Exception as e:
            logger.critical(f"¡¡ERROR FATAL en bucle principal!!: {e}", exc_info=True)
        finally:
            logger.info("Iniciando secuencia de apagado...")
            send_telegram_alert("💤 Bot apagándose...", level="info")
            if hasattr(self, 'broker') and self.broker:
                # Opcional: No cerrar posiciones al apagar manualmente
                # self.broker.close_all_positions() 
                self.broker.shutdown()
            else:
                 try: mt5.shutdown()
                 except: pass
            logger.info("--- BOT RL APAGADO ---")

    def execute_action(self, action: int):
        """Ejecuta la acción decidida por el agente."""
        current_pos_before_action = self.current_position

        try:
            if action == 1: # === ACCIÓN: COMPRAR ===
                if current_pos_before_action == 0:
                    logger.info("Ejecutando: Abrir COMPRA")
                    send_telegram_alert(f"📈 Abriendo COMPRA en {self.ticker}", level="info")
                    self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_BUY)
                elif current_pos_before_action == -1:
                    logger.info("Ejecutando: Cerrar VENTA y abrir COMPRA")
                    send_telegram_alert(f"🔄 Cerrando VENTA y abriendo COMPRA en {self.ticker}", level="info")
                    self.broker.close_position_by_ticker(self.ticker)
                    time.sleep(0.5)
                    self.sync_state()
                    if self.current_position == 0:
                         self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_BUY)
                    else:
                         logger.warning("Fallo al cerrar VENTA antes de abrir COMPRA. Acción abortada.")
                         send_telegram_alert("⚠️ Fallo al cerrar VENTA. Compra abortada.", level="warning")
                else: # current_pos == 1
                    logger.info("Acción COMPRAR, pero ya estamos comprados. Manteniendo.")

            elif action == 2: # === ACCIÓN: VENDER ===
                if current_pos_before_action == 0:
                    logger.info("Ejecutando: Abrir VENTA")
                    send_telegram_alert(f"📉 Abriendo VENTA en {self.ticker}", level="info")
                    self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_SELL)
                elif current_pos_before_action == 1:
                    logger.info("Ejecutando: Cerrar COMPRA y abrir VENTA")
                    send_telegram_alert(f"🔄 Cerrando COMPRA y abriendo VENTA en {self.ticker}", level="info")
                    self.broker.close_position_by_ticker(self.ticker)
                    time.sleep(0.5)
                    self.sync_state()
                    if self.current_position == 0:
                         self.broker.open_market_order_with_risk(self.ticker, mt5.ORDER_TYPE_SELL)
                    else:
                         logger.warning("Fallo al cerrar COMPRA antes de abrir VENTA. Acción abortada.")
                         send_telegram_alert("⚠️ Fallo al cerrar COMPRA. Venta abortada.", level="warning")
                else: # current_pos == -1
                    logger.info("Acción VENDER, pero ya estamos vendidos. Manteniendo.")

            else: # === ACCIÓN: MANTENER (action == 0) ===
                logger.info(f"Acción MANTENER. Posición actual sin cambios: {current_pos_before_action}")

            # Sincronización final
            time.sleep(1.0) # Dar tiempo al broker para procesar
            self.sync_state()

        except OrderError as e:
            logger.error(f"Error de Orden al ejecutar acción {action}: {e}", exc_info=False)
            send_telegram_alert(f"🚨 ERROR de Orden: {e}", level="error")
        except DataError as e:
             logger.error(f"Error de Datos al ejecutar acción {action}: {e}", exc_info=False)
             send_telegram_alert(f"🚨 ERROR de Datos: {e}", level="error")
        except ConnectionError as e:
             logger.error(f"Error de Conexión al ejecutar acción {action}: {e}", exc_info=False)
             raise # Re-lanzar para que el bucle principal maneje la reconexión
        except Exception as e:
             logger.error(f"Error inesperado en execute_action({action}): {e}", exc_info=True)
             send_telegram_alert(f"🚨 ERROR Inesperado (execute_action): {e}", level="error")


if __name__ == "__main__":
    try:
        from pyrobot.config_loader import load_config
        config = load_config(os.path.join(dir_path, 'config.yaml'))
    except Exception as e:
        logging.basicConfig()
        logging.critical(f"Error fatal al leer 'config.yaml': {e}", exc_info=True)
        sys.exit(1)

    bot = None
    try:
         bot = LiveBot(config)
         bot.run()
    except Exception as e:
         logging.critical(f"Fallo fatal al inicializar o ejecutar el bot: {e}", exc_info=True)
         if bot and hasattr(bot, 'broker') and bot.broker:
              bot.broker.shutdown()
         else:
              try: mt5.shutdown()
              except: pass
         sys.exit(1)
