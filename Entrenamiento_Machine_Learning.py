import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("--- ENTRENAMIENTO FINAL DEL MODELO ---")

# --- PASO 1: Cargar la Base de Datos ---
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_filename = 'features_database_normalized.csv'
    csv_path = os.path.join(script_dir, csv_filename)
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{csv_filename}' en la carpeta {script_dir}.")
    exit()

print(f"Base de datos cargada con éxito. Usando todos los {len(data)} registros para el entrenamiento.")
print("Distribución de clases:")
print(data['Tipo'].value_counts())
print("-" * 30)

# --- PASO 2: Preparar TODOS los Datos (Features y Labels) ---
features = [
    'Duracion_s', 
    'RMS_normalized_%MVC', 
    'MAV_normalized_%MVC', 
    'MDF_Hz', 
    'MNF_Hz'
]
# 'X_all' (X_todo) contendrá todas las características de tus 125 registros
X_all = data[features]

# 'y_all' (y_todo) contendrá todas las etiquetas
y_all = data['Tipo']

# --- PASO 3: Escalar TODOS los Datos ---
# Creamos y "entrenamos" el escalador con el 100% de los datos
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
print("Todos los datos han sido escalados.")
print("-" * 30)

# --- PASO 4: Entrenar el Modelo FINAL ---
print("Entrenando el modelo Random Forest con el 100% de los datos...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_all_scaled, y_all)

print("¡Modelo final entrenado!")
print("-" * 30)

# --- PASO 5: Guardar el Modelo y Scaler Finales ---
# Estos son los archivos que serán usados en la aplicación en tiempo real
joblib.dump(model, 'modelo_emg_FINAL.joblib')
joblib.dump(scaler, 'scaler_emg_FINAL.joblib')

print("¡Modelo y Scaler FINALES guardados exitosamente!")
print("Archivos creados:")
print(f"  {os.path.join(script_dir, 'modelo_emg_FINAL.joblib')}")
print(f"  {os.path.join(script_dir, 'scaler_emg_FINAL.joblib')}")
