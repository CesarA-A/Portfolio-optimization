

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/raw/SnP_daily_update.csv',header=[0,1],index_col = 0)
df_cpy = df.copy()
df_cpy.head()

df_cpy.shape

df_cpy.info()

null_pct = df_cpy['Close'].isnull().mean() * 100
print("Estadísticas de valores faltantes en 'Close' por Ticker:")
print(null_pct.describe())

tickers_with_many_nulls = null_pct[null_pct > 50]
print(f"\nCantidad de tickers con mas del 50% de datos faltantes: {len(tickers_with_many_nulls)}")

df_cpy.index = pd.to_datetime(df_cpy.index)

daily_mean_close = df_cpy['Close'].mean(axis=1)
daily_total_volume = df_cpy['Volume'].sum(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

axes[0].plot(daily_mean_close.index, daily_mean_close.values, color='tab:blue', label='Precio Promedio de Cierre')
axes[0].set_title('Tendencia del Precio de Cierre Promedio de las Acciones', fontsize=14)
axes[0].set_ylabel('Precio (USD)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].fill_between(daily_total_volume.index, daily_total_volume.values, color='tab:orange', alpha=0.4, label='Volumen Total Diario')
axes[1].set_title('Volumen Total de Transacciones en el Tiempo', fontsize=14)
axes[1].set_xlabel('Fecha', fontsize=12)
axes[1].set_ylabel('Volumen', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()


correlation_matrix = df_cpy['Close'].corr()

print(correlation_matrix.iloc[:])

tickers_completos = null_pct[null_pct == 0].index.get_level_values('Ticker').unique()

volumen_medio = df_cpy['Volume'][tickers_completos].mean().sort_values(ascending=False)
top_20_tickers = volumen_medio.head(20).index.tolist()

empresas_lstm = top_20_tickers

print("La variable 'empresas_lstm' ha sido restaurada con éxito:")
print(empresas_lstm)


# Verificar nulos por característica para cada una de las 20 empresas seleccionadas
caracteristicas = df_cpy.columns.levels[0].tolist()
verificacion_nulos = {}

for col in caracteristicas:
    # Obtenemos la suma de nulos de las 20 empresas en cada columna de forma limpia
    nulos_por_ticker = df_cpy[col][empresas_lstm].isnull().sum()
    verificacion_nulos[col] = nulos_por_ticker

# Crear un DataFrame con el recuento de nulos para visualizar el resultado
df_verificacion = pd.DataFrame(verificacion_nulos)
print("Cantidad de valores nulos (NaN) en las 20 empresas seleccionadas por característica:")
print(df_verificacion)


data = df_cpy.isna().sum()


# Identificar los tickers con al menos un valor nulo en 'Close'
activos_con_nulos = null_pct[null_pct > 0].index.get_level_values('Ticker').unique().tolist()

# Guardar la lista en un archivo de texto o simplemente tenerla en memoria
print(f"Total de activos con datos faltantes: {len(activos_con_nulos)}")
print("\nLista de activos con nulos:\n", activos_con_nulos)


