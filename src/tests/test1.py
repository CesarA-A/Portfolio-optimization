import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def preparar_datos_lstm(ruta_csv='./data/30_yr_market_data.csv', lookback=21, split_ratio=0.8):
    df = pd.read_csv(ruta_csv)

    df['Date'].info()

    new_df = df[['Date', 'Apple']].set_index('Date')
    #Informacion de Apple
    # print(new_df)
    # print(new_df.info())sx
    # print(new_df.isna().sum())

    new_df['Apple'] = new_df['Apple'].pct_change()
    
    #Eliminamos datos nulos
    new_df.dropna(inplace=True)

    fechas_originales = new_df.index.values

    #Escalamos los datos
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(new_df[['Apple']])

    #Creacion de la ventana
    X, y = [], []
    fechas_y = []

    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i, 0])
        y.append(scaled_data[i, 0])
        fechas_y.append(fechas_originales[i])

    X = np.array(X) 
    y = np.array(y)
    fechas_y = np.array(fechas_y)

    # Reshape a 3D para LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    #Separacion en datos de entrenaminiento y test
    split_index = int(len(X) * split_ratio)
    
    # Conjunto de entrenamiento (Pasado)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Guardamos las fechas por separado para graficar despues
    fechas_train, fechas_test = fechas_y[:split_index], fechas_y[split_index:]

    print("Datos listos para la LSTM")
    print(f"X_train shape: {X_train.shape} \t y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape} \t y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler, fechas_test


X_train, y_train, X_test, y_test, scaler, fechas_test = preparar_datos_lstm(lookback=42)

model = Sequential()

model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_absolute_error')

print(model.summary())

early_stop = EarlyStopping(
    monitor='val_loss',    
    patience=7,           
    restore_best_weights=True 
)

print("\nIniciando el entrenamiento de la LSTM")
history = model.fit(
    X_train, y_train,   
    epochs=70, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stop]
)

print("\nEntrenamiento completado con exito")
print("\nRealizando predicciones")

predicciones_escaladas = model.predict(X_test)
predicciones = scaler.inverse_transform(predicciones_escaladas)

y_test_2d = np.reshape(y_test, (y_test.shape[0], 1))
valores_reales = scaler.inverse_transform(y_test_2d)


print("Generando la grafica comparativa...")

plt.ion() 
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(fechas_test, valores_reales, color='blue', label='Precio Real de Apple')
ax.plot(fechas_test, predicciones, color='red', linestyle='--', label='Predicción LSTM')

ax.set_title('Comparación de Precio Real vs. Predicción de la Red LSTM (Apple)', fontsize=16)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio de la Acción (USD)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)

plt.xticks(rotation=45)
ax.xaxis.set_major_locator(plt.MaxNLocator(10)) 

plt.tight_layout()

def al_cerrar_ventana(event):
    os.makedirs('results', exist_ok=True)
    ruta_guardado = 'results/Test1/Test1.png'
    plt.savefig(ruta_guardado, bbox_inches='tight')

fig.canvas.mpl_connect('close_event', al_cerrar_ventana)
plt.show()