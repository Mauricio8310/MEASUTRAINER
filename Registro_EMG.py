import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal
from collections import deque
import os
import json

class EMGSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Análisis EMG")
        self.root.geometry("1200x800")
        
        # Variables de configuración
        self.puerto = "COM4"
        self.baudrate = 115200
        self.fs = 100  # Hz
        self.ser = None
        self.grabando = False
        self.archivo_actual = None
        self.thread_grabacion = None
        
        # Parámetros de detección (del archivo MATLAB)
        self.umbral_inicio = 150
        self.umbral_fin = 100
        self.tiempo_max_sin_contraccion = 3  # segundos
        
        # Filtro Butterworth bandpass 20-45 Hz
        self.lowcut = 20
        self.highcut = 45
        self.b, self.a = signal.butter(4, [self.lowcut, self.highcut], 
                                       btype='bandpass', fs=self.fs)
        
        # Buffer para el filtro (estado del filtro)
        self.zi = signal.lfilter_zi(self.b, self.a)
        
        # Buffer para visualización en tiempo real
        self.buffer_size = 500
        self.data_buffer = deque([0] * self.buffer_size, maxlen=self.buffer_size)
        
        # Diccionario para almacenar valores MVC por sujeto
        self.mvc_values = {}
        self.cargar_mvc_values()
        
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear pestañas
        self.tab_grabacion = ttk.Frame(self.notebook)
        self.tab_procesamiento = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_grabacion, text="🔴 Módulo de Grabación")
        self.notebook.add(self.tab_procesamiento, text="🔬 Módulo de Procesamiento")
        
        self.crear_interfaz_grabacion()
        self.crear_interfaz_procesamiento()
        
        # Intentar conexión automática
        self.conectar_serial()
        
        # Iniciar lectura continua en segundo plano
        self.thread_lectura = threading.Thread(target=self.leer_datos_continuo, daemon=True)
        self.thread_lectura.start()
        
    def cargar_mvc_values(self):
        """Carga los valores MVC guardados"""
        if os.path.exists('mvc_calibration.json'):
            with open('mvc_calibration.json', 'r') as f:
                self.mvc_values = json.load(f)
    
    def guardar_mvc_values(self):
        """Guarda los valores MVC"""
        with open('mvc_calibration.json', 'w') as f:
            json.dump(self.mvc_values, f)
    
    def crear_interfaz_grabacion(self):
        """Crea la interfaz del módulo de grabación"""
        # Frame superior - Estado de conexión
        frame_conexion = ttk.LabelFrame(self.tab_grabacion, text="Estado de Conexión", padding=10)
        frame_conexion.pack(fill='x', padx=10, pady=5)
        
        self.label_estado = ttk.Label(frame_conexion, text="Desconectado", 
                                      foreground="red", font=('Arial', 10, 'bold'))
        self.label_estado.pack()
        
        # Frame de configuración
        frame_config = ttk.LabelFrame(self.tab_grabacion, text="Configuración de Grabación", padding=10)
        frame_config.pack(fill='x', padx=10, pady=5)
        
        # ID del Sujeto
        ttk.Label(frame_config, text="ID del Sujeto:").grid(row=0, column=0, sticky='w', pady=5)
        self.entry_sujeto = ttk.Entry(frame_config, width=15)
        self.entry_sujeto.grid(row=0, column=1, padx=5, pady=5)
        self.entry_sujeto.insert(0, "S01")
        
        # Tipo de Entrenamiento
        ttk.Label(frame_config, text="Tipo de Entrenamiento:").grid(row=1, column=0, sticky='w', pady=5)
        self.combo_tipo = ttk.Combobox(frame_config, width=15, 
                                       values=["MVC", "Fuerza", "Hipertrofia", "Resistencia"])
        self.combo_tipo.grid(row=1, column=1, padx=5, pady=5)
        self.combo_tipo.current(0)
        
        # Número de Serie
        ttk.Label(frame_config, text="Número de Serie:").grid(row=2, column=0, sticky='w', pady=5)
        self.entry_serie = ttk.Entry(frame_config, width=15)
        self.entry_serie.grid(row=2, column=1, padx=5, pady=5)
        self.entry_serie.insert(0, "1")
        
        # Botón de grabación
        self.btn_grabar = ttk.Button(frame_config, text="▶️ Grabar", 
                                     command=self.toggle_grabacion)
        self.btn_grabar.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Frame de visualización
        frame_plot = ttk.LabelFrame(self.tab_grabacion, text="Señal EMG en Tiempo Real", padding=10)
        frame_plot.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Crear gráfica
        self.fig_grab = Figure(figsize=(10, 4), dpi=100)
        self.ax_grab = self.fig_grab.add_subplot(111)
        self.ax_grab.set_ylim([0, 800])
        self.ax_grab.set_xlim([0, self.buffer_size])
        self.ax_grab.set_xlabel('Muestras')
        self.ax_grab.set_ylabel('Amplitud EMG Filtrada')
        self.ax_grab.grid(True)
        self.ax_grab.axhline(y=self.umbral_inicio, color='r', linestyle='--', label='Umbral')
        
        self.line_grab, = self.ax_grab.plot([], [], 'b-', linewidth=1.5)
        self.ax_grab.legend()
        
        self.canvas_grab = FigureCanvasTkAgg(self.fig_grab, frame_plot)
        self.canvas_grab.draw()
        self.canvas_grab.get_tk_widget().pack(fill='both', expand=True)
        
        # Label de depuración
        self.label_debug = ttk.Label(frame_plot, text="Esperando datos...", foreground="blue")
        self.label_debug.pack()
        
        # Iniciar actualización de gráfica
        self.actualizar_grafica_grabacion()
    
    def crear_interfaz_procesamiento(self):
        """Crea la interfaz del módulo de procesamiento"""
        # Frame Paso 1 - Calibración MVC
        frame_mvc = ttk.LabelFrame(self.tab_procesamiento, text="Paso 1: Calibrar Sujeto (MVC)", padding=10)
        frame_mvc.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(frame_mvc, text="Cargar archivo MVC del sujeto:").grid(row=0, column=0, sticky='w', pady=5)
        self.btn_cargar_mvc = ttk.Button(frame_mvc, text="📁 Cargar MVC", 
                                         command=self.cargar_mvc)
        self.btn_cargar_mvc.grid(row=0, column=1, padx=5, pady=5)
        
        self.label_mvc_status = ttk.Label(frame_mvc, text="Sin calibración", foreground="orange")
        self.label_mvc_status.grid(row=0, column=2, padx=5, pady=5)
        
        # Frame Paso 2 - Procesamiento
        frame_proc = ttk.LabelFrame(self.tab_procesamiento, text="Paso 2: Procesar Ejercicio", padding=10)
        frame_proc.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(frame_proc, text="Cargar archivo de ejercicio:").grid(row=0, column=0, sticky='w', pady=5)
        self.btn_cargar_ejercicio = ttk.Button(frame_proc, text="📁 Cargar Ejercicio", 
                                               command=self.cargar_ejercicio)
        self.btn_cargar_ejercicio.grid(row=0, column=1, padx=5, pady=5)
        
        # Frame de características
        frame_features = ttk.LabelFrame(self.tab_procesamiento, text="Características Extraídas", padding=10)
        frame_features.pack(fill='x', padx=10, pady=5)
        
        self.text_features = tk.Text(frame_features, height=10, width=80)
        self.text_features.pack(fill='x', padx=5, pady=5)
        
        # Botón para añadir a base de datos
        self.btn_add_db = ttk.Button(frame_features, text="💾 Añadir a la Base de Datos", 
                                     command=self.anadir_a_db, state='disabled')
        self.btn_add_db.pack(pady=5)
        
        # Frame de visualización
        frame_plot_proc = ttk.LabelFrame(self.tab_procesamiento, text="Visualización de Señal", padding=10)
        frame_plot_proc.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.fig_proc = Figure(figsize=(10, 4), dpi=100)
        self.ax_proc = self.fig_proc.add_subplot(111)
        
        self.canvas_proc = FigureCanvasTkAgg(self.fig_proc, frame_plot_proc)
        self.canvas_proc.get_tk_widget().pack(fill='both', expand=True)
        
        # Variable para almacenar características actuales
        self.features_actuales = None
    
    def conectar_serial(self):
        """Intenta conectar con el puerto serie"""
        try:
            self.ser = serial.Serial(self.puerto, self.baudrate, timeout=1)
            time.sleep(2)  # Esperar estabilización
            # Reiniciar el estado del filtro
            self.zi = signal.lfilter_zi(self.b, self.a)
            self.label_estado.config(text=f"Conectado a {self.puerto}", foreground="green")
            messagebox.showinfo("Conexión", f"Conectado exitosamente a {self.puerto}")
        except Exception as e:
            self.label_estado.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Error de Conexión", f"No se pudo conectar a {self.puerto}\n{str(e)}")
    
    def leer_datos_continuo(self):
        """Lee datos continuamente del puerto serie para visualización y grabación"""
        contador = 0
        while True:
            try:
                if self.ser and self.ser.is_open:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode('utf-8').strip()
                        try:
                            valor_raw = float(line)
                            timestamp = time.time()
                            
                            # Centrar la señal
                            valor_centrado = valor_raw - 2048
                            
                            # Aplicar filtro
                            valor_filtrado_array, self.zi = signal.lfilter(
                                self.b, self.a, [valor_centrado], zi=self.zi
                            )
                            
                            # Tomar valor absoluto
                            valor_filtrado = abs(valor_filtrado_array[0])
                            
                            # Actualizar buffer para visualización
                            self.data_buffer.append(valor_filtrado)
                            
                            # Si está grabando, guardar en archivo
                            if self.grabando and self.archivo_actual:
                                self.archivo_actual.write(f"{timestamp},{valor_raw},{valor_filtrado}\n")
                                self.archivo_actual.flush()
                            
                            # Actualizar label de depuración cada 50 muestras
                            contador += 1
                            if contador % 50 == 0:
                                self.root.after(0, lambda v_raw=valor_raw, v_filt=valor_filtrado: 
                                    self.label_debug.config(
                                        text=f"Datos: Raw={v_raw:.0f}, Filtrado={v_filt:.2f}, Buffer={len(self.data_buffer)}"
                                    ))
                            
                        except ValueError:
                            continue
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error en lectura continua: {e}")
                time.sleep(0.5)
    
    def toggle_grabacion(self):
        """Inicia o detiene la grabación"""
        if not self.grabando:
            self.iniciar_grabacion()
        else:
            self.detener_grabacion()
    
    def iniciar_grabacion(self):
        """Inicia la grabación de datos"""
        if self.ser is None or not self.ser.is_open:
            messagebox.showerror("Error", "No hay conexión con el puerto serie")
            return
        
        # Validar campos
        sujeto = self.entry_sujeto.get().strip()
        tipo = self.combo_tipo.get()
        serie = self.entry_serie.get().strip()
        
        if not sujeto or not serie:
            messagebox.showerror("Error", "Complete todos los campos")
            return
        
        # Crear nombre de archivo
        nombre_archivo = f"{tipo}_{sujeto}_S{serie}.csv"
        
        try:
            self.archivo_actual = open(nombre_archivo, 'w')
            self.archivo_actual.write("timestamp,raw_value,filtered_value\n")
            
            self.grabando = True
            self.btn_grabar.config(text="⏹️ Detener")
            
            messagebox.showinfo("Grabación", f"Grabando en: {nombre_archivo}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo crear el archivo: {str(e)}")
    
    def detener_grabacion(self):
        """Detiene la grabación de datos"""
        self.grabando = False
        if self.archivo_actual:
            self.archivo_actual.close()
            self.archivo_actual = None
        self.btn_grabar.config(text="▶️ Grabar")
        messagebox.showinfo("Grabación", "Grabación detenida")
    

    
    def actualizar_grafica_grabacion(self):
        """Actualiza la gráfica en tiempo real"""
        try:
            if len(self.data_buffer) > 0:
                datos = list(self.data_buffer)
                x_data = range(len(datos))
                self.line_grab.set_data(x_data, datos)
                self.ax_grab.set_xlim([0, self.buffer_size])
                
                # Ajustar ylim dinámicamente si hay datos
                if max(datos) > 0:
                    max_val = max(datos)
                    self.ax_grab.set_ylim([0, max(800, max_val * 1.2)])
                
                self.canvas_grab.draw()
                self.canvas_grab.flush_events()
        except Exception as e:
            print(f"Error actualizando gráfica: {e}")
        
        self.root.after(100, self.actualizar_grafica_grabacion)
    
    def cargar_mvc(self):
        """Carga y procesa un archivo MVC"""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo MVC",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Leer archivo
            df = pd.read_csv(filename)
            
            # Extraer ID del sujeto del nombre del archivo
            base_name = os.path.basename(filename)
            parts = base_name.replace('.csv', '').split('_')
            sujeto_id = parts[1] if len(parts) > 1 else "Unknown"
            
            # Procesar señal
            signal_filtered = df['filtered_value'].values
            
            # Calcular RMS
            rms_mvc = np.sqrt(np.mean(signal_filtered**2))
            
            # Guardar valor MVC
            self.mvc_values[sujeto_id] = rms_mvc
            self.guardar_mvc_values()
            
            self.label_mvc_status.config(
                text=f"✅ {sujeto_id}: RMS_MVC = {rms_mvc:.2f}",
                foreground="green"
            )
            
            messagebox.showinfo("Calibración", 
                              f"MVC calibrado para {sujeto_id}\nRMS: {rms_mvc:.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar MVC: {str(e)}")
    
    def cargar_ejercicio(self):
        """Carga y procesa un archivo de ejercicio"""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de ejercicio",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Leer archivo
            df = pd.read_csv(filename)
            
            # Extraer información del nombre del archivo
            base_name = os.path.basename(filename)
            parts = base_name.replace('.csv', '').split('_')
            tipo_ejercicio = parts[0]
            sujeto_id = parts[1] if len(parts) > 1 else "Unknown"
            
            # Verificar que exista calibración MVC
            if sujeto_id not in self.mvc_values:
                messagebox.showwarning("Advertencia", 
                                      f"No hay calibración MVC para {sujeto_id}")
                return
            
            # Procesar señal
            signal_filtered = df['filtered_value'].values
            
            # Calcular características
            features = self.extraer_caracteristicas(signal_filtered, sujeto_id, tipo_ejercicio)
            
            # Mostrar características
            self.mostrar_caracteristicas(features)
            
            # Visualizar señal
            self.visualizar_senal(df)
            
            # Habilitar botón de guardar
            self.btn_add_db.config(state='normal')
            self.features_actuales = features
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar ejercicio: {str(e)}")
    
    def extraer_caracteristicas(self, signal_data, sujeto_id, tipo_ejercicio):
        """Extrae características de la señal"""
        # Características crudas
        duracion = len(signal_data) / self.fs
        rms_valor = np.sqrt(np.mean(signal_data**2))
        mav_valor = np.mean(np.abs(signal_data))
        
        # Características normalizadas (% MVC)
        mvc_rms = self.mvc_values[sujeto_id]
        rms_normalizado = (rms_valor / mvc_rms) * 100
        mav_normalizado = (mav_valor / mvc_rms) * 100
        
        # Características de frecuencia
        f, psd = signal.welch(signal_data, self.fs, nperseg=min(256, len(signal_data)))
        
        # Frecuencia mediana (MDF)
        cumsum_psd = np.cumsum(psd)
        mdf = f[np.where(cumsum_psd >= cumsum_psd[-1]/2)[0][0]]
        
        # Frecuencia media (MNF)
        mnf = np.sum(f * psd) / np.sum(psd)
        
        return {
            'Sujeto': sujeto_id,
            'Tipo': tipo_ejercicio,
            'Duracion_s': duracion,
            'RMS_raw': rms_valor,
            'MAV_raw': mav_valor,
            'RMS_normalized_%MVC': rms_normalizado,
            'MAV_normalized_%MVC': mav_normalizado,
            'MDF_Hz': mdf,
            'MNF_Hz': mnf
        }
    
    def mostrar_caracteristicas(self, features):
        """Muestra las características en el área de texto"""
        self.text_features.delete(1.0, tk.END)
        
        texto = "=" * 60 + "\n"
        texto += "CARACTERÍSTICAS EXTRAÍDAS\n"
        texto += "=" * 60 + "\n\n"
        
        texto += f"Sujeto: {features['Sujeto']}\n"
        texto += f"Tipo de Ejercicio: {features['Tipo']}\n\n"
        
        texto += "--- Características Temporales ---\n"
        texto += f"Duración: {features['Duracion_s']:.2f} s\n\n"
        
        texto += "--- Características Crudas ---\n"
        texto += f"RMS: {features['RMS_raw']:.2f}\n"
        texto += f"MAV: {features['MAV_raw']:.2f}\n\n"
        
        texto += "--- Características Normalizadas (%MVC) ---\n"
        texto += f"RMS Normalizado: {features['RMS_normalized_%MVC']:.2f}%\n"
        texto += f"MAV Normalizado: {features['MAV_normalized_%MVC']:.2f}%\n\n"
        
        texto += "--- Características de Frecuencia ---\n"
        texto += f"Frecuencia Mediana (MDF): {features['MDF_Hz']:.2f} Hz\n"
        texto += f"Frecuencia Media (MNF): {features['MNF_Hz']:.2f} Hz\n"
        
        self.text_features.insert(1.0, texto)
    
    def visualizar_senal(self, df):
        """Visualiza la señal cargada"""
        self.ax_proc.clear()
        
        tiempo = np.arange(len(df)) / self.fs
        
        self.ax_proc.plot(tiempo, df['filtered_value'], 'b-', label='Señal Filtrada', linewidth=0.8)
        self.ax_proc.axhline(y=self.umbral_inicio, color='r', linestyle='--', 
                            label=f'Umbral Inicio ({self.umbral_inicio})')
        
        self.ax_proc.set_xlabel('Tiempo (s)')
        self.ax_proc.set_ylabel('Amplitud EMG Filtrada')
        self.ax_proc.set_title('Señal EMG Procesada')
        self.ax_proc.legend()
        self.ax_proc.grid(True, alpha=0.3)
        
        self.canvas_proc.draw()
    
    def anadir_a_db(self):
        """Añade las características actuales a la base de datos"""
        if self.features_actuales is None:
            return
        
        archivo_db = 'features_database_normalized.csv'
        
        # Crear DataFrame con las características
        df_new = pd.DataFrame([self.features_actuales])
        
        # Añadir a archivo o crear nuevo
        if os.path.exists(archivo_db):
            df_new.to_csv(archivo_db, mode='a', header=False, index=False)
        else:
            df_new.to_csv(archivo_db, mode='w', header=True, index=False)
        
        messagebox.showinfo("Base de Datos", 
                          f"Características añadidas a {archivo_db}")
        
        # Deshabilitar botón y limpiar características
        self.btn_add_db.config(state='disabled')
        self.features_actuales = None
    
    def __del__(self):
        """Limpieza al cerrar"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        if self.archivo_actual:
            self.archivo_actual.close()

# Iniciar aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = EMGSystem(root)
    root.mainloop()