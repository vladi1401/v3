# pyrobot/broker.py

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import numpy as np
import math
from forex_python.converter import CurrencyRates
import holidays
from typing import List, Dict, Union, Tuple
from currency_converter import CurrencyConverter
import time
import logging
import talib as ta # <-- CORRECCIÓN: Importar talib en el top-level

# Corregir import relativo si es necesario
from .utils import ensure_connection
from .exceptions import ConnectionError, OrderError, DataError


class PyRobot:

    def __init__(self, client_id: int, client_mdp: str, trading_serveur: str, risk_percent: float,
                 atr_period: int, sl_atr_mult: float, tp_atr_mult: float, pip_value: float):
        self.client_id = client_id
        self.client_mdp = client_mdp
        self.trading_serveur = trading_serveur
        # --- Nuevos parámetros de riesgo ---
        self.risk_percent = risk_percent / 100.0 # Convertir % a decimal
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.pip_value = pip_value
        # --- ---
        self.cr = CurrencyRates()
        self.cc = CurrencyConverter()
        self.logger = logging.getLogger(__name__)

    # --- Funciones de Conexión (Sin Cambios) ---
    def _connect(self): # ... (igual que antes)
        if not mt5.initialize(): raise ConnectionError(f"mt5.initialize() falló. Código: {mt5.last_error()}")
        if not mt5.login(self.client_id, self.client_mdp, self.trading_serveur): raise ConnectionError(f"mt5.login() falló. Código: {mt5.last_error()}. Revisa credenciales.")
        self.logger.info(f"Conexión MT5 establecida para login {self.client_id} en {self.trading_serveur}")
    def _connect_with_backoff(self, max_retries=5, initial_delay=5): # ... (igual que antes)
        delay = initial_delay
        for i in range(max_retries):
            try: self._connect(); return
            except ConnectionError as e:
                self.logger.warning(f"Intento de conexión {i+1}/{max_retries} fallido: {e}")
                if i == max_retries - 1: self.logger.error("Máximo de reintentos alcanzado."); raise
                self.logger.info(f"Reintentando en {delay} segundos..."); time.sleep(delay); delay *= 2
    def initial_connection(self): self._connect_with_backoff() # ... (igual que antes)
    def shutdown(self): self.logger.info("Cerrando conexión con MT5."); mt5.shutdown() # ... (igual que antes)
    def is_connected(self) -> bool: return mt5.terminal_info() is not None # ... (igual que antes)

    # --- Funciones de Mercado y Cuenta (CORREGIDO DECORADOR) ---
    @property
    @ensure_connection # <-- ensure_connection PRIMERO
    def market_open(self) -> bool:
        current_time = datetime.now(timezone.utc); us_holidays = holidays.US()
        if current_time.date() not in us_holidays:
            weekday = current_time.weekday(); hour = current_time.hour
            if weekday == 4 and hour >= 22: return False
            elif weekday == 5: return False
            elif weekday == 6 and hour < 22: return False
            else: return True
        else: return False

    @property
    @ensure_connection # <-- ensure_connection PRIMERO
    def liquidity_hours(self) -> bool:
        current_utc_hour = datetime.now(timezone.utc).hour; return 3 <= current_utc_hour < 22

    @ensure_connection
    def get_portfolio_pos_time(self): # ... (ligeramente modificado)
        portfolio = {}
        positions = mt5.positions_get()
        if positions is None:
             if mt5.last_error() != mt5.RES_S_OK: self.logger.warning(f"Error al obtener posiciones: {mt5.last_error()}")
             return None
        current_time_utc = datetime.now(timezone.utc)
        for position in positions:
            position_time_utc = datetime.fromtimestamp(position.time, tz=timezone.utc)
            position_age = current_time_utc - position_time_utc
            portfolio[position.symbol] = {
                'PosType' : 1 if position.type == mt5.POSITION_TYPE_BUY else -1, 'Time': position_age,
                'Ticket': position.ticket, 'Volume': position.volume, 'PriceOpen': position.price_open # Añadir precio
            }
        return portfolio

    # --- ¡¡NUEVA GESTIÓN DE RIESGO!! ---

    @ensure_connection
    def get_current_atr(self, ticker: str, timeframe=mt5.TIMEFRAME_M1) -> float:
        """Obtiene el valor ATR actual para el cálculo de SL/TP."""
        try:
            # Necesitamos historial suficiente para calcular ATR
            rates = mt5.copy_rates_from_pos(ticker, timeframe, 0, self.atr_period + 10)
            if rates is None or len(rates) < self.atr_period:
                raise DataError(f"Datos insuficientes para calcular ATR({self.atr_period}) en {ticker}")

            df = pd.DataFrame(rates)
            # Asegurarse de que las columnas son numéricas
            df[['high', 'low', 'close']] = df[['high', 'low', 'close']].apply(pd.to_numeric)
            
            # --- CORRECCIÓN: Usar talib importado ---
            atr_values = ta.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
            
            if atr_values.empty:
                raise DataError(f"Cálculo de ATR devolvió vacío para {ticker}")
            atr_value = atr_values.iloc[-1]

            if atr_value is None or np.isnan(atr_value) or atr_value <= 0:
                 raise DataError(f"Valor ATR inválido ({atr_value}) calculado para {ticker}")

            return atr_value
        except Exception as e:
            self.logger.error(f"Error al obtener ATR para {ticker}: {e}")
            raise DataError(f"Error al obtener ATR para {ticker}: {e}")

    @ensure_connection
    def calculate_sl_tp(self, ticker: str, order_type: int, entry_price: float) -> Tuple[float, float]:
        """Calcula los precios de SL y TP basados en el ATR."""
        atr = self.get_current_atr(ticker)
        symbol_info = mt5.symbol_info(ticker)
        if symbol_info is None: raise DataError(f"No se pudo obtener symbol_info para {ticker}")
        point = symbol_info.point
        digits = symbol_info.digits

        sl_distance_points = self.sl_atr_mult * atr
        tp_distance_points = self.tp_atr_mult * atr

        if order_type == mt5.ORDER_TYPE_BUY:
            sl_price = entry_price - sl_distance_points
            tp_price = entry_price + tp_distance_points
        else: # ORDER_TYPE_SELL
            sl_price = entry_price + sl_distance_points
            tp_price = entry_price - tp_distance_points

        # Redondear a los dígitos correctos del símbolo
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        return sl_price, tp_price

    @ensure_connection
    def calculate_volume_by_risk(self, ticker: str, sl_price: float, entry_price: float) -> float:
        """Calcula el volumen/lote basado en un riesgo porcentual fijo."""
        try:
            account_info = mt5.account_info()
            symbol_info = mt5.symbol_info(ticker)
            if account_info is None or symbol_info is None:
                raise DataError("No se pudo obtener información de cuenta o símbolo para calcular volumen.")

            account_equity = account_info.equity
            account_currency = account_info.currency
            contract_size = symbol_info.trade_contract_size
            currency_base = symbol_info.currency_base
            currency_profit = symbol_info.currency_profit # Moneda en que se expresa el beneficio
            point = symbol_info.point

            # 1. Calcular el riesgo monetario
            risk_amount = account_equity * self.risk_percent
            if risk_amount <= 0:
                self.logger.warning("Riesgo monetario calculado es cero o negativo. Usando lote mínimo.")
                return symbol_info.volume_min

            # 2. Calcular la distancia al SL en puntos (no pips)
            sl_distance_points = abs(entry_price - sl_price)
            if sl_distance_points <= 0:
                self.logger.warning("Distancia SL es cero o negativa. Usando lote mínimo.")
                return symbol_info.volume_min

            # 3. Calcular el valor de 1 punto (tick value) en la moneda de beneficio (currency_profit)
            # Para Forex: tick_value = contract_size * point (o tick_size)
            # Para otros (CFDs, etc.): mt5.symbol_info(ticker).tick_value
            tick_value = symbol_info.tick_value if symbol_info.tick_value > 0 else contract_size * point

            # 4. Ajustar el valor del tick a la moneda de la cuenta
            if currency_profit == account_currency:
                tick_value_adjusted = tick_value
            else:
                 # Necesitamos la tasa de cambio Profit -> Account
                 # Ej: EURJPY (Profit=JPY) en cuenta USD. Necesitamos JPYUSD.
                 pair_profit_account = None
                 if currency_profit + account_currency in mt5.symbols_get():
                     pair_profit_account = currency_profit + account_currency
                 elif account_currency + currency_profit in mt5.symbols_get():
                     pair_profit_account = account_currency + currency_profit # Invertir la tasa luego
                 else:
                     # Usar forex_python como fallback
                     try:
                         rate = self.cr.get_rate(currency_profit, account_currency)
                         tick_value_adjusted = tick_value * rate
                         pair_profit_account = "forex_python" # Flag
                     except Exception as e_conv:
                         raise DataError(f"No se pudo obtener tasa de cambio {currency_profit}->{account_currency}: {e_conv}")

                 if pair_profit_account != "forex_python":
                     tick_info = mt5.symbol_info_tick(pair_profit_account)
                     if tick_info is None: raise DataError(f"No se pudo obtener tick para conversión {pair_profit_account}")
                     if pair_profit_account.startswith(account_currency): # Tasa invertida
                         tick_value_adjusted = tick_value / tick_info.bid
                     else: # Tasa directa
                         tick_value_adjusted = tick_value * tick_info.bid


            if tick_value_adjusted <= 0:
                self.logger.warning("Valor de tick ajustado inválido. Usando lote mínimo.")
                return symbol_info.volume_min

            # 5. Calcular el volumen
            # Volumen = (Riesgo Monetario) / (Distancia SL en Puntos * Valor Ajustado de 1 Punto)
            volume = risk_amount / (sl_distance_points * tick_value_adjusted)

            # 6. Ajustar a los límites de volumen del broker
            volume_min = symbol_info.volume_min
            volume_max = symbol_info.volume_max
            volume_step = symbol_info.volume_step

            volume = max(volume_min, volume)
            volume = min(volume_max, volume)
            volume = math.floor(volume / volume_step) * volume_step if volume_step > 0 else volume
            volume = round(volume, len(str(volume_step).split('.')[-1]) if '.' in str(volume_step) else 0) # Redondear a decimales del step

            final_volume = max(volume, volume_min) # Asegurar mínimo absoluto
            if final_volume <= 0: # Doble check
                 self.logger.warning(f"Volumen final calculado {final_volume} es inválido. Usando lote mínimo {volume_min}.")
                 return volume_min

            return final_volume

        except Exception as e:
            self.logger.error(f"Error en calculate_volume_by_risk para {ticker}: {e}", exc_info=True)
            try: # Intentar devolver el mínimo del broker como fallback final
                symbol_info = mt5.symbol_info(ticker)
                return symbol_info.volume_min if symbol_info else 0.01
            except:
                return 0.01 # Fallback absoluto

    # --- FUNCIONES DE TRADING ACTUALIZADAS ---

    @ensure_connection
    def open_market_order_with_risk(self, ticker: str, order_type: int):
        """
        Abre una orden a mercado calculando volumen, SL y TP dinámicamente.
        """
        self.logger.info(f"Intentando abrir orden {order_type} para {ticker} con gestión de riesgo...")
        
        symbol_info = mt5.symbol_info(ticker) # Obtener para 'point'
        if symbol_info is None: raise OrderError(f"No se pudo obtener symbol_info para {ticker}")

        tick = mt5.symbol_info_tick(ticker)
        if tick is None: raise OrderError(f"No se pudo obtener el tick para {ticker}")
        entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        # Calcular SL y TP
        try:
            sl_price, tp_price = self.calculate_sl_tp(ticker, order_type, entry_price)
            self.logger.info(f"Calculado: Entry={entry_price}, SL={sl_price}, TP={tp_price}")
        except DataError as e:
            raise OrderError(f"No se pudo abrir orden por fallo en cálculo SL/TP: {e}")

        # Calcular Volumen
        try:
            volume = self.calculate_volume_by_risk(ticker, sl_price, entry_price)
            self.logger.info(f"Volumen calculado basado en riesgo ({self.risk_percent*100}%): {volume} lotes")
        except DataError as e:
             raise OrderError(f"No se pudo abrir orden por fallo en cálculo de volumen: {e}")

        if volume <= 0:
            raise OrderError(f"Volumen calculado inválido ({volume}). No se abre orden.")

        # Construir y enviar orden
        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": ticker,
            "volume": volume,
            "type": order_type,
            "price": entry_price, # Precio para órdenes a mercado
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 123456,
            "comment": "RL Bot Order Risk",
            "type_time": mt5.ORDER_TIME_GTC,
            # Usar FILLING_FOK o FILLING_IOC según prefieras
            "type_filling": mt5.ORDER_FILLING_FOK, # Fill Or Kill
        }

        result = mt5.order_send(order_request)

        # --- Verificación Post-Orden ---
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"Fallo al enviar orden para {ticker}: {result.retcode} - {result.comment}"
            self.logger.error(msg)
            raise OrderError(msg)

        self.logger.info(f"¡ORDEN ENVIADA! {ticker} Vol: {volume} @ {result.price} SL: {sl_price} TP: {tp_price}. Verificando ejecución...")

        # Verificar si la posición realmente se abrió
        time.sleep(0.5) # Pequeña pausa para que el broker procese
        # Necesitamos buscar por el 'magic' number o símbolo, ya que el 'ticket' es de la orden, no de la posición aún
        positions = mt5.positions_get(symbol=ticker)
        position_found = False
        if positions:
            for pos in positions:
                # Comprobar si coincide (aproximadamente) tipo, volumen y precio cercano
                if pos.type == order_type and abs(pos.volume - volume) < 0.001 and abs(pos.price_open - result.price) < symbol_info.point * 10:
                     position_found = True
                     self.logger.info(f"¡VERIFICACIÓN OK! Posición {pos.ticket} abierta.")
                     break # Salir si la encontramos

        if not position_found:
            msg = f"¡FALLO DE VERIFICACIÓN! Orden {result.order} para {ticker} enviada pero no se encontró la posición correspondiente."
            self.logger.error(msg)
            # Podrías intentar cerrar la orden si quedó pendiente, pero es complejo
            raise OrderError(msg)

        return result

    @ensure_connection
    def close_position_by_ticker(self, ticker: str):
        """
        Cierra la primera posición abierta para un ticker.
        (Verificación post-cierre añadida)
        """
        self.logger.info(f"Intentando cerrar posición para {ticker}...")

        positions = self.get_portfolio_pos_time() # Obtiene el ticket
        if not positions or ticker not in positions:
            self.logger.warning(f"No se encontró ninguna posición abierta para cerrar en {ticker}.")
            return None # No lanzar error si no hay nada que cerrar

        pos_info = positions[ticker]
        ticket_to_close = pos_info['Ticket'] # Usar el ticket obtenido
        volume = pos_info['Volume']
        is_buy_position = (pos_info['PosType'] == 1)
        order_type = mt5.ORDER_TYPE_SELL if is_buy_position else mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(ticker)
        if tick is None: raise OrderError(f"No se pudo obtener el tick para cerrar {ticker}")
        price = tick.bid if is_buy_position else tick.ask

        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket_to_close, # ¡El ticket de la posición a cerrar!
            "symbol": ticker, "volume": volume, "type": order_type, "price": price,
            "deviation": 20, "magic": 123456, "comment": "RL Bot Close Risk",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK, # O IOC
        }

        result = mt5.order_send(order_request)

        # --- Verificación Post-Cierre ---
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Podríamos añadir manejo si el error es "posición ya cerrada"
            msg = f"Fallo al cerrar posición {ticket_to_close} para {ticker}: {result.retcode} - {result.comment}"
            self.logger.error(msg)
            raise OrderError(msg)

        self.logger.info(f"¡ORDEN DE CIERRE ENVIADA! {ticker} Ticket: {ticket_to_close} @ {result.price}. Verificando cierre...")

        time.sleep(0.5)
        # Verificar que la posición con ESE TICKET ya no exista
        position_info = mt5.positions_get(ticket=ticket_to_close)
        if position_info is not None and len(position_info) > 0:
            msg = f"¡FALLO DE VERIFICACIÓN! Orden de cierre enviada pero la posición {ticket_to_close} sigue abierta."
            self.logger.error(msg)
            # Podrías reintentar el cierre aquí
            raise OrderError(msg)

        self.logger.info(f"¡VERIFICACIÓN OK! Posición {ticket_to_close} cerrada.")
        return result

    # La función 'cancel_order' se puede mantener igual
    @ensure_connection
    def cancel_order(self): # ... (igual que antes)
        orders = mt5.orders_get();
        if orders is None or len(orders) == 0: return
        self.logger.info(f"Cancelando {len(orders)} órdenes pendientes...")
        for order in orders:
            result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Error al cancelar orden {order.ticket}: {result.retcode} - {result.comment}")