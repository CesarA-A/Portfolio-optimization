import pandas as pd
import numpy as np

# Cargar el dataset MultiIndex original
df = pd.read_csv("./data/raw/SnP_daily_update.csv", header=[0,1], index_col=0)

def download_data(assets, tolerancia_nulos_pct=10.0):

    activos_validos = []
    
    print("Evaluando calidad de los activos seleccionados ===")
    for asset in assets:
        columna_cierre = ('Close', asset)
        
        if columna_cierre in df.columns:
            total_filas = len(df)
            nulos = df[columna_cierre].isna().sum()
            pct_faltantes = (nulos / total_filas) * 100
            
            if pct_faltantes <= tolerancia_nulos_pct:
                activos_validos.append(asset)
                print(f" {asset}: {pct_faltantes:.2f}% de datos faltantes. (Apto)")
            else:
                print(f"{asset}: {pct_faltantes:.2f}% de datos faltantes. (DESCARTADO por superar el {tolerancia_nulos_pct}%)")
        else:
            print(f" {asset}: No se encuentra en el archivo SnP_daily_update.csv")
            
    # Si ningún activo pasó la prueba, detenemos el proceso
    if not activos_validos:
        print("\nError: Ninguno de los activos cumple con el criterio de datos faltantes.")
        return None

    # Construir la lista final de columnas MultiIndex solo para los activos que sí pasaron el filtro
    columnas_seleccionadas = []
    for asset in activos_validos:
        columnas_seleccionadas.extend([
            ('Close', asset),
            ('High', asset),
            ('Low', asset),
            ('Open', asset),
            ('Volume', asset)
        ])
    
    # Extraer el subconjunto de datos
    df_d = df[columnas_seleccionadas].copy()
    
    # --- APLICACIÓN DE LA ESTRATEGIA 1 ---
    # Eliminamos las filas del inicio donde falten datos en CUALQUIERA de las columnas seleccionadas
    # (Esto corta automáticamente el dataset en la primera fecha donde todos los activos ya cotizan)
    df_d.dropna(how='any', inplace=True)
    
    # Exportar el dataset limpio y listo para la LSTM
    df_d.to_csv('./data/raw/processed/data_LSTM.csv')
    
    print(f"\nDataset guardado con éxito en: './data/raw/processed/data_LSTM.csv'")
    print(f"Rango de fechas final: Desde {df_d.index.min()} hasta {df_d.index.max()}")
    print(f"Total de registros disponibles para entrenamiento: {len(df_d)} días.")
    
    return df_d



if __name__ == "__main__":
    assets = [
        'NVDA', 'MSFT', 'GOOGL', 'NFLX', 'AMZN', 
        'F', 'WMT', 'YUM', 'BAC', 'WFC', 
        'PFE', 'ZBH', 'XOM', 'MPC', 'WM', 
        'VMC', 'LYB', 'XEL', 'WEC',
        'AAPL', 'JPM', 'V', 'KO' 
    ] 
    
    df_procesado = download_data(assets=assets)
    
