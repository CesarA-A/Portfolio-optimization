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



def preparar_datos_lstm(ruta_csv='./data/raw/processed/dataset_aap.csv',lookback=21, split_ratio=0.8,dias_futuros=5):
        df = pd.read_csv(ruta_csv)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        df.sort_index(inplace=True)


        df["Target_PSO"] = df["Close"].shift(-dias_futuros) / df["Close"] - 1
        
        columnas_precio = ['Close', 'High', 'Low', 'Open']
        df[columnas_precio] = df[columnas_precio].pct_change()

        df['Volume'] = np.log1p(df['Volume'])

        df.dropna(inplace=True)
        fechas_originales = df.index.values

        scaler = MinMaxScaler(feature_range=(0,1))
        datos_numericos = df.select_dtypes(include=[np.number]).values
        split_index_temporal = int(len(datos_numericos) * split_ratio)
    
        # Separamos en crudo antes de escalar
        train_raw = datos_numericos[:split_index_temporal]
        test_raw = datos_numericos[split_index_temporal:]
        
        # El escalador SOLO aprende (fit) del pasado (Train)
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw) 
        
        scaled_data = np.vstack((train_scaled, test_scaled))
        #Creacion de la ventana
        X, y = [], []
        fechas_y = []

        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback : i, :])
            y.append(scaled_data[i, 0])
            fechas_y.append(fechas_originales[i])

        X = np.array(X) 
        y = np.array(y)
        fechas_y = np.array(fechas_y)

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

X_train, y_train, X_test, y_test, scaler, fechas_test = preparar_datos_lstm(lookback=21)


def desescalar_objetivo(scaler, valores_1d, num_features):
    # Creamos una matriz llena de ceros con la forma original (filas, num_columnas)
    matriz_aux = np.zeros((len(valores_1d), num_features))
    # Colocamos nuestros valores a desescalar en la primera columna (índice 0)
    matriz_aux[:, 0] = valores_1d.flatten()
    # Invertimos la transformación y recuperamos solo la primera columna desescalada
    return scaler.inverse_transform(matriz_aux)[:, 0]


model = Sequential()

model.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))
model.add(LSTM(units=16, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

print(model.summary())

early_stop = EarlyStopping(
    monitor='val_loss',    
    patience=10,           
    restore_best_weights=True 
)



print("\nIniciando el entrenamiento de la LSTM")
history = model.fit(
    X_train, y_train,   
    epochs=100, 
    batch_size=42, 
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stop]
)

print("\nEntrenamiento completado con exito")
print("\nRealizando predicciones")

num_features = X_train.shape[2]  

predicciones_escaladas = model.predict(X_test)

predicciones = desescalar_objetivo(scaler, predicciones_escaladas, num_features)
valores_reales = desescalar_objetivo(scaler, y_test, num_features)

print("Generando la grafica comparativa...")

plt.ion() 
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(fechas_test, valores_reales, color='blue', label='Retorno Real (Apple)')
ax.plot(fechas_test, predicciones, color='red', linestyle='--', label='Predicción LSTM')

ax.set_title('Comparación de Retorno Real vs. Predicción LSTM (Apple)', fontsize=16)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Retorno Diario (pct_change)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)

plt.xticks(rotation=45)
ax.xaxis.set_major_locator(plt.MaxNLocator(10)) 

plt.tight_layout()

def al_cerrar_ventana(event):
    os.makedirs('results', exist_ok=True)
    ruta_guardado = 'results/Test2/Test2.png'
    plt.savefig(ruta_guardado, bbox_inches='tight')

fig.canvas.mpl_connect('close_event', al_cerrar_ventana)
plt.show()


