El proyecto está dividido en dos fases principales: Entrenamiento y Operación en Vivo.

¿En qué se basa? (El Cerebro de la IA)
El "cerebro" del bot es un modelo de IA que aprende a operar en un entorno simulado.

Algoritmo de IA: Se basa en un algoritmo de Aprendizaje por Refuerzo (RL) llamado PPO (Proximal Policy Optimization). En lugar de programar reglas fijas (como "comprar si el RSI < 30"), el bot aprende por sí mismo una estrategia óptima.

Entorno de Simulación (TradingEnv): Para aprender, el bot (el "Agente") se entrena dentro de un entorno de trading simulado (rl_core/rl_environment.py).

Estado (Lo que ve el bot): En cada paso, el bot "ve" el estado actual del mercado. Este estado es un conjunto de indicadores técnicos normalizados (como EMA, RSI, ATR y Momentum) más su posición actual (comprado, vendido o fuera del mercado).

Acciones (Lo que hace el bot): El bot solo puede tomar tres decisiones: 0: Mantener, 1: Comprar, o 2: Vender.

Recompensa (El Objetivo): El bot es "recompensado" o "castigado" en función del resultado de sus acciones. El objetivo principal, según la configuración, es maximizar el Sharpe Ratio (retorno ajustado al riesgo) a lo largo del tiempo.

¿Cómo funciona? (El Flujo de Trabajo)
El proyecto tiene un flujo de trabajo claro para crear el bot y luego ponerlo a operar.

Fase 1: Entrenamiento y Optimización (Offline)
Este proceso se ejecuta una sola vez para crear el modelo de IA, usando el script train_rl.py.

Recolección de Datos: Primero, se descargan años de datos históricos (ej. 2020-2023) del par de divisas (EURUSD M1) desde MetaTrader 5 y se guardan en archivos CSV.

Cálculo de Features: Se cargan los datos CSV y se calculan todos los indicadores (EMA, RSI, etc.).

Normalización: Los valores de los indicadores se normalizan (para que la IA los entienda mejor) y las estadísticas de esta normalización (media, desviación estándar) se guardan en un archivo clave: normalization_stats.json.

Optimización (Optuna): El script usa Optuna para probar cientos de configuraciones de IA (hiperparámetros). Entrena un modelo con cada configuración usando los datos de training_years (ej. 2020-2022).

Backtesting: Cada modelo entrenado se prueba en datos de validación (validation_years, ej. 2023) usando backtester.py para ver cuál obtiene el mejor Sharpe Ratio.

Modelo Final: Una vez que Optuna encuentra la mejor configuración, el script entrena un modelo final usando todos los datos (ej. 2020-2023). Este "cerebro" entrenado se guarda como rl_forex_model_ppo_optimized.zip.

Fase 2: Operación en Vivo (Online)
Este es el bot en ejecución, operando en una cuenta real o demo usando el script run_rl_robot.py.

Conexión y Carga: El bot se inicia, carga el config.yaml, y se conecta a la cuenta de MetaTrader 5 (MT5). Carga los dos archivos clave de la Fase 1: el modelo rl_forex_model_ppo_optimized.zip y las estadísticas normalization_stats.json.

"Calentamiento" del Buffer: El bot descarga inmediatamente las últimas ~500 velas de MT5 para "calentar" sus indicadores (el RSI de 14 períodos necesita al menos 14 velas para dar un valor).

Bucle Principal (Cada Minuto): El bot entra en un bucle infinito que se ejecuta al inicio de cada nueva vela (cada minuto):

a. Cortafuegos (Firewall): Primero, revisa las reglas de riesgo de la cuenta. Comprueba si el equity actual ha violado el Límite de Pérdida Diaria o el Límite de Drawdown Máximo (Trailing). Si se viola un límite, el bot liquida todas las posiciones y se detiene (ya sea por el día o permanentemente).

b. Filtros de Mercado: Comprueba si el mercado está abierto y si hay noticias de alto impacto (usando news_filter.py). Si el mercado está cerrado o hay noticias, pausa la operativa.

c. Obtener Estado (Live): El LiveProcessor (rl_core/live_processor.py) obtiene la vela más reciente de MT5, calcula los indicadores, y los normaliza usando las estadísticas guardadas en normalization_stats.json. Añade la posición actual (ej. "Comprado" = 1) para crear el vector de estado.

d. Tomar Decisión (IA): El vector de estado se pasa al modelo de IA cargado (agent.predict), que devuelve una acción: 0 (Mantener), 1 (Comprar) o 2 (Vender).

e. Ejecutar Acción: El bot traduce esta acción en una orden de trading:

Si la IA dice "Comprar" (1) y el bot está "Vendido" (-1), primero cierra la venta y luego abre una compra.

Si la IA dice "Vender" (2) y el bot está plano (0), abre una venta.

Si la IA dice "Mantener" (0), no hace nada.

f. Gestión de Riesgo (por Operación): Al abrir una operación, el módulo pyrobot/broker.py calcula automáticamente el volumen (lotes) basado en el riesgo (ej. 0.5% del equity) y establece el Stop Loss (SL) y Take Profit (TP) basados en el ATR (Volatilidad).
