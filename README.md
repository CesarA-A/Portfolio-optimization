# Proyecto: Optimización de Portafolios mediante LSTM + PSO

## Objetivo General

Desarrollar un modelo híbrido que combine una red neuronal LSTM para la predicción de rendimientos de activos financieros y un algoritmo PSO para la asignación óptima de pesos dentro de un portafolio.

---

# Arquitectura Objetivo

```text
Datos históricos
        ↓
      LSTM
        ↓
Predicción de rendimientos
        ↓
       PSO
        ↓
 Pesos óptimos
        ↓
 Portafolio eficiente
```

---

# Fase 0: Preparación del Entorno

## Herramientas

- Python
- VS Code
- venv
- Jupyter Notebook

## Librerías

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install yfinance
pip install scikit-learn
pip install jupyter
pip install tensorflow
pip install pyswarms
```

## Estructura del Proyecto

```text
portfolio_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── models/
│
├── results/
│
└── .venv/
```

---

# Fase 1: Obtención y Comprensión de Datos

## Objetivos

- Descargar datos históricos.
- Comprender la serie temporal.
- Detectar valores faltantes.

## Activos Iniciales

```text
AAPL
MSFT
NVDA
```

## Actividades

- Descargar datos con yfinance.
- Visualizar precios.
- Analizar volumen.
- Revisar datos faltantes.

## Entregable

```text
Dataset limpio y validado.
```

---

# Fase 2: Ingeniería de Características

## Objetivos

Transformar los precios en variables útiles para el modelo.

## Actividades

### Rendimientos

```python
returns = close.pct_change()
```

### Posibles indicadores futuros

- RSI
- MACD
- SMA 20
- SMA 50

## Entregable

```text
Serie temporal de rendimientos.
```

---

# Fase 3: Series Temporales

## Objetivos

Preparar datos para LSTM.

## Conceptos a estudiar

- Tendencia
- Estacionalidad
- Ruido
- Ventanas temporales

## Ventana Inicial

```text
42 días de entrada
↓
1 día de salida
```

Ejemplo

```text
[42 observaciones]
        ↓
rendimiento día 43
```

## Entregable

```python
X.shape = (muestras,42,features)

y.shape = (muestras,)
```

---

# Fase 4: Normalización

## Objetivos

Escalar datos para mejorar el entrenamiento.

## Herramienta

```python
MinMaxScaler
```

## Entregable

Datos normalizados.

---

# Fase 5: Implementación de LSTM

## Objetivos

Predecir rendimientos futuros.

## Modelo Inicial

```python
LSTM(50)
Dense(1)
```

## Flujo

```text
42 días históricos
        ↓
      LSTM
        ↓
Predicción día siguiente
```

## Entregable

Modelo entrenado.

---

# Fase 6: Evaluación de la LSTM

## Métricas

- MAE
- RMSE

## Visualización

```text
Valor Real
vs
Valor Predicho
```

## Entregable

Análisis de desempeño.

---

# Fase 7: Predicción Multi-Activo

## Objetivos

Generar predicciones para varios activos.

## Activos

```text
AAPL
MSFT
NVDA
```

## Resultado esperado

```text
AAPL → retorno esperado
MSFT → retorno esperado
NVDA → retorno esperado
```

Ejemplo

```text
AAPL = 0.8%
MSFT = 0.5%
NVDA = 1.2%
```

---

# Fase 8: Implementación de PSO

## Objetivos

Optimizar los pesos del portafolio.

## Variables

```text
w1
w2
w3
```

## Restricción

```text
w1 + w2 + w3 = 1
```

## Entradas

- Rendimientos esperados
- Matriz de covarianza

## Salida

```text
Pesos óptimos
```

Ejemplo

```text
AAPL = 35%
MSFT = 25%
NVDA = 40%
```

---

# Fase 9: Integración LSTM + PSO

## Arquitectura Final

```text
Datos históricos
        ↓
       LSTM
        ↓
Retornos esperados
        ↓
        PSO
        ↓
Pesos óptimos
        ↓
Portafolio eficiente
```

## Objetivo

Usar predicciones para la construcción del portafolio.

---

# Fase 10: Comparación de Resultados

## Benchmark 1

Portafolio equiponderado

```text
33%
33%
33%
```

## Benchmark 2

Modelo de Markowitz

## Benchmark 3

```text
LSTM + PSO
```

## Métricas

- Retorno esperado
- Volatilidad
- Sharpe Ratio

---

# Extensión Futura

## FinBERT + LSTM + PSO

```text
Noticias financieras
        ↓
      FinBERT
        ↓
    Sentimiento
        ↓
Datos históricos
        ↓
       LSTM
        ↓
Predicción de rendimientos
        ↓
        PSO
        ↓
 Portafolio óptimo
```

### Objetivo

Combinar finanzas cuantitativas y cualitativas para mejorar la toma de decisiones.

---

# Meta para el Martes

Tener implementado:

```text
AAPL
 ↓
Rendimientos
 ↓
Ventanas temporales
 ↓
Normalización
 ↓
LSTM
 ↓
Predicción
 ↓
Gráfica Real vs Predicho
```

No implementar PSO hasta que la LSTM funcione correctamente.
