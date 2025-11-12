import sys
import numpy as np
from scipy import signal
from scipy.signal import welch, butter
import joblib
import serial
import threading
import time
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, QFrame, QProgressBar, QScrollArea)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class EMGDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Measutrainer - EMG Detection System")
        self.setGeometry(50, 50, 1600, 950)
        
        # Aplicar tema oscuro moderno
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #eee;
            }
        """)
        
        # Variables globales del sistema
        self.mvc_rms_usuario = None
        self.calibracion_completada = False
        self.detectando = False
        self.grabando_serie = False
        
        # Variables de bioimpedancia
        self.midiendo_impedancia = False
        self.esperando_impedancia = False
        self.z_pre = None
        self.z_post = None
        self.timeout_impedancia = 10  # segundos
        self.tiempo_inicio_medicion = None
        
        # Buffers
        self.signal_buffer = deque(maxlen=1000)
        self.serie_buffer_actual = []
        self.calibracion_buffer = []
        
        # Variables de detección
        self.UMBRAL_INICIO = 150
        self.UMBRAL_FIN = 100
        self.TIEMPO_PAUSA_MAX = 3.0
        self.tiempo_pausa = 0
        self.ultimo_tiempo = time.time()
        
        # Contador de series detectadas
        self.series_detectadas = 0
        self.ultima_prediccion = None
        
        # Configuración del filtro Butterworth 20-45 Hz
        self.fs = 100
        self.lowcut = 20
        self.highcut = 45
        self.b, self.a = butter(4, [self.lowcut, self.highcut], btype='bandpass', fs=self.fs)
        self.zi = signal.lfilter_zi(self.b, self.a)
        
        # Variable para control de lectura
        self.running = True
        
        # Cargar modelo y scaler
        try:
            self.model = joblib.load('modelo_emg.joblib')
            self.scaler = joblib.load('scaler_emg.joblib')
            self.modelo_cargado = True
            print("✓ Modelo y scaler cargados correctamente")
        except Exception as e:
            print(f"✗ Error cargando modelo: {e}")
            self.modelo_cargado = False
        
        # Configurar conexión Bluetooth
        self.puerto_conectado = False
        self.ser = None
        self.puerto_bt = None
        
        # Configurar GUI
        self.init_ui()
        
        # Intentar conectar después de crear la GUI
        QTimer.singleShot(500, self.conectar_bluetooth)
        
        # Timer para actualizar gráfica
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_grafica)
        self.timer.start(50)
        
        # Timer para verificar timeout de impedancia
        self.timer_impedancia = QTimer()
        self.timer_impedancia.timeout.connect(self.verificar_timeout_impedancia)
        self.timer_impedancia.start(500)
    
    def conectar_bluetooth(self):
        """Intenta conectar con el ESP32 vía Bluetooth"""
        try:
            import serial.tools.list_ports
            
            # Buscar puertos COM disponibles
            ports = serial.tools.list_ports.comports()
            
            self.log_mensaje("🔍 Buscando dispositivo Bluetooth...")
            print("\n🔍 Puertos disponibles:")
            for port in ports:
                print(f"   - {port.device}: {port.description}")
            
            # Intentar encontrar el dispositivo Bluetooth
            for port in ports:
                if "Bluetooth" in port.description or "BT" in port.description or "Standard Serial" in port.description:
                    print(f"\n🔍 Probando puerto Bluetooth: {port.device}")
                    try:
                        # Intentar abrir el puerto
                        self.ser = serial.Serial(port.device, 115200, timeout=1)
                        time.sleep(2)
                        
                        # Limpiar buffer inicial
                        self.ser.reset_input_buffer()
                        self.ser.reset_output_buffer()
                        
                        # VERIFICACIÓN CRÍTICA
                        print("⏳ Verificando comunicación (3 segundos)...")
                        time.sleep(1)
                        
                        datos_recibidos = False
                        intentos = 0
                        max_intentos = 30
                        
                        while intentos < max_intentos and not datos_recibidos:
                            if self.ser.in_waiting > 0:
                                try:
                                    test_data = self.ser.readline().decode('utf-8', errors='ignore').strip()
                                    # Verificar que sea un número válido
                                    test_value = float(test_data)
                                    print(f"✅ Datos válidos recibidos: {test_value}")
                                    datos_recibidos = True
                                except:
                                    pass
                            time.sleep(0.1)
                            intentos += 1
                        
                        if not datos_recibidos:
                            print(f"❌ No se recibieron datos válidos de {port.device}")
                            self.ser.close()
                            continue
                        
                        # Si llegamos aquí, la conexión es válida
                        self.zi = signal.lfilter_zi(self.b, self.a)
                        self.puerto_conectado = True
                        self.puerto_bt = port.device
                        print(f"✓ Conectado exitosamente a {port.device}")
                        
                        self.actualizar_card(self.card_conexion, "Conectado BT", "#10b981")
                        
                        self.serial_thread = threading.Thread(target=self.leer_datos_continuo, daemon=True)
                        self.serial_thread.start()
                        
                        self.log_mensaje(f"✅ Conectado vía Bluetooth: {port.device}")
                        self.log_mensaje(f"📡 Dispositivo: {port.description}")
                        self.log_mensaje("✅ Comunicación verificada - Recibiendo datos")
                        return
                        
                    except Exception as e:
                        print(f"Error al conectar a {port.device}: {e}")
                        if self.ser and self.ser.is_open:
                            self.ser.close()
                        continue
            
            # Si no se encontró ningún puerto válido
            raise Exception("No se encontró el dispositivo ESP32 transmitiendo datos")
            
        except Exception as e:
            print(f"✗ Error conectando Bluetooth: {e}")
            self.puerto_conectado = False
            self.actualizar_card(self.card_conexion, "Desconectado", "#ef4444")
            self.log_mensaje(f"✗ Error de conexión Bluetooth")
            self.log_mensaje("⚠️ Soluciones:")
            self.log_mensaje("   1. Verifique que ESP32 esté ENCENDIDO")
            self.log_mensaje("   2. Verifique que el código esté cargado")
            self.log_mensaje("   3. Si persiste: Desvincular y volver a emparejar")
            self.log_mensaje("      'Measutrainer_EMG' en configuración de Bluetooth")
    
    def leer_datos_continuo(self):
        """Lee datos continuamente del puerto serie"""
        contador = 0
        sin_datos_contador = 0
        ultima_advertencia = 0
        
        print("🔄 Hilo de lectura iniciado")
        
        while self.running:
            try:
                if self.ser and self.ser.is_open:
                    if self.ser.in_waiting > 0:
                        sin_datos_contador = 0
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        # Debug: mostrar datos crudos solo en consola
                        if contador % 100 == 0:
                            print(f"Datos recibidos: '{line}' (in_waiting: {self.ser.in_waiting})")
                        
                        # PROTOCOLO DE COMUNICACIÓN:
                        # Formato de datos del ESP32:
                        # - Datos EMG normales: número simple (ej: "2048")
                        # - Impedancia: "Z:valor" (ej: "Z:523.45")
                        
                        if line.startswith("Z:"):
                            # Recibir dato de impedancia
                            self.procesar_dato_impedancia(line)
                        else:
                            # Procesar dato EMG normal
                            try:
                                valor_raw = float(line)
                                
                                valor_centrado = valor_raw - 2048
                                
                                valor_filtrado_array, self.zi = signal.lfilter(
                                    self.b, self.a, [valor_centrado], zi=self.zi
                                )
                                
                                amplitud_actual = abs(valor_filtrado_array[0])
                                
                                self.signal_buffer.append(amplitud_actual)
                                
                                if hasattr(self, 'calibrando') and self.calibrando:
                                    self.calibracion_buffer.append(amplitud_actual)
                                
                                if self.detectando and self.grabando_serie:
                                    self.serie_buffer_actual.append(amplitud_actual)
                                
                                contador += 1
                                
                            except ValueError:
                                continue
                    else:
                        sin_datos_contador += 1
                        if sin_datos_contador >= 5000 and (sin_datos_contador - ultima_advertencia) >= 5000:
                            print("⚠️ No se están recibiendo datos del ESP32")
                            ultima_advertencia = sin_datos_contador
                        time.sleep(0.001)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error en lectura continua: {e}")
                time.sleep(0.5)
    
    def procesar_dato_impedancia(self, line):
        """
        Procesa datos de impedancia recibidos del ESP32
        Formato esperado: "Z:valor" donde valor es la impedancia en ohmios
        """
        try:
            # Extraer valor después de "Z:"
            valor_str = line.split("Z:")[1]
            valor_impedancia = float(valor_str)
            
            if self.esperando_impedancia:
                if self.midiendo_impedancia and self.z_pre is None:
                    # Es medición PRE
                    self.z_pre = valor_impedancia
                    self.label_z_pre.setText(f"{self.z_pre:.1f} Ω")
                    self.log_mensaje(f"✅ Impedancia PRE recibida: {self.z_pre:.1f} Ω")
                    self.finalizar_medicion_pre()
                    
                elif self.midiendo_impedancia and self.z_pre is not None and self.z_post is None:
                    # Es medición POST
                    self.z_post = valor_impedancia
                    self.label_z_post.setText(f"{self.z_post:.1f} Ω")
                    self.log_mensaje(f"✅ Impedancia POST recibida: {self.z_post:.1f} Ω")
                    self.finalizar_medicion_post()
                
                self.esperando_impedancia = False
            
            print(f"📊 Impedancia recibida: {valor_impedancia:.2f} Ω")
            
        except Exception as e:
            print(f"Error procesando impedancia: {e}")
            self.log_mensaje(f"⚠️ Error procesando dato de impedancia: {line}")
    
    def enviar_comando_medir_impedancia(self):
        """
        Envía comando al ESP32 para iniciar medición de impedancia
        Comando: "MEASURE_Z\n"
        """
        if self.ser and self.ser.is_open:
            try:
                comando = "MEASURE_Z\n"
                self.ser.write(comando.encode('utf-8'))
                self.ser.flush()
                print(f"📤 Comando enviado al ESP32: {comando.strip()}")
                return True
            except Exception as e:
                print(f"Error enviando comando: {e}")
                self.log_mensaje(f"❌ Error enviando comando al ESP32")
                return False
        return False
    
    def iniciar_medicion_impedancia_pre(self):
        """Inicia la medición de impedancia PRE-ejercicio"""
        self.midiendo_impedancia = True
        self.esperando_impedancia = True
        self.tiempo_inicio_medicion = time.time()
        self.btn_detectar.setEnabled(False)
        self.btn_calibrar.setEnabled(False)
        
        self.label_bio_estado.setText("⚡ Midiendo impedancia\nANTES del ejercicio\n\n⚠️ No te muevas")
        self.label_bio_estado.setStyleSheet("""
            QLabel {
                background-color: #764ba2;
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }
        """)
        
        self.progress_bio.setVisible(True)
        self.progress_bio.setRange(0, 0)  # Modo indeterminado
        
        self.log_mensaje("⚡ Iniciando medición de impedancia PRE...")
        self.log_mensaje("📤 Enviando comando al ESP32...")
        
        # Enviar comando al ESP32
        if not self.enviar_comando_medir_impedancia():
            self.cancelar_medicion_impedancia("Error de comunicación con ESP32")
    
    def finalizar_medicion_pre(self):
        """Finaliza la medición PRE y comienza la grabación de EMG"""
        self.progress_bio.setVisible(False)
        self.progress_bio.setRange(0, 100)  # Restaurar rango normal
        
        self.label_bio_estado.setText("✅ Impedancia PRE medida\n\n🏋️ Listo para iniciar")
        self.label_bio_estado.setStyleSheet("""
            QLabel {
                background-color: #10b981;
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }
        """)
        
        self.log_mensaje("🏋️ Iniciando grabación de EMG...")
        
        # Iniciar grabación de EMG
        self.detectando = True
        self.grabando_serie = True
        self.serie_buffer_actual = []
        self.midiendo_impedancia = False
        
        self.btn_detectar.setEnabled(True)
        self.btn_detectar.setText("⏹️ FINALIZAR SERIE")
        self.btn_detectar.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ee0979, stop:1 #ff6a00);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ee0979);
            }
        """)
        
        self.label_estado.setText("🔴 GRABANDO SERIE - Realice su ejercicio")
        self.label_estado.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ee0979, stop:1 #ff6a00);
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-weight: bold;
            }
        """)
    
    def iniciar_medicion_impedancia_post(self):
        """Inicia la medición de impedancia POST-ejercicio"""
        self.midiendo_impedancia = True
        self.esperando_impedancia = True
        self.tiempo_inicio_medicion = time.time()
        self.btn_detectar.setEnabled(False)
        
        self.label_bio_estado.setText("⚡ Midiendo impedancia\nDESPUÉS del ejercicio\n\n⚠️ No te muevas")
        self.label_bio_estado.setStyleSheet("""
            QLabel {
                background-color: #f59e0b;
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }
        """)
        
        self.progress_bio.setVisible(True)
        self.progress_bio.setRange(0, 0)  # Modo indeterminado
        
        self.log_mensaje("⚡ Midiendo impedancia POST...")
        self.log_mensaje("📤 Enviando comando al ESP32...")
        
        # Enviar comando al ESP32
        if not self.enviar_comando_medir_impedancia():
            self.cancelar_medicion_impedancia("Error de comunicación con ESP32")
    
    def finalizar_medicion_post(self):
        """Finaliza la medición POST y compara resultados"""
        self.progress_bio.setVisible(False)
        self.progress_bio.setRange(0, 100)  # Restaurar rango normal
        
        # Calcular cambio porcentual
        cambio_porcentual = ((self.z_post - self.z_pre) / self.z_pre) * 100
        cambio_absoluto = self.z_post - self.z_pre
        
        # Comparar y mostrar resultado
        # Típicamente la impedancia DISMINUYE con el ejercicio debido al aumento de flujo sanguíneo
        if cambio_porcentual < -2:  # Disminución mayor al 2%
            self.label_bio_resultado.setText(
                f"✅ ÉXITO\n\n"
                f"Disminución detectada:\n"
                f"{abs(cambio_porcentual):.1f}%\n"
                f"({abs(cambio_absoluto):.1f} Ω)\n\n"
                f"Se confirma actividad muscular"
            )
            self.label_bio_resultado.setStyleSheet("""
                QLabel {
                    background-color: #10b981;
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-weight: bold;
                }
            """)
            self.log_mensaje(f"✅ Cambio: {cambio_porcentual:.1f}% ({cambio_absoluto:.1f} Ω)")
            self.log_mensaje("✅ ¡Actividad muscular confirmada!")
        elif cambio_porcentual > 2:  # Aumento mayor al 2%
            self.label_bio_resultado.setText(
                f"⚠️ ADVERTENCIA\n\n"
                f"Aumento: {cambio_porcentual:.1f}%\n"
                f"({cambio_absoluto:.1f} Ω)\n\n"
                f"Resultado inesperado"
            )
            self.label_bio_resultado.setStyleSheet("""
                QLabel {
                    background-color: #f59e0b;
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-weight: bold;
                }
            """)
            self.log_mensaje(f"⚠️ Cambio: +{cambio_porcentual:.1f}% (+{cambio_absoluto:.1f} Ω)")
            self.log_mensaje("⚠️ Advertencia: Aumento inesperado de impedancia")
        else:  # Cambio menor al 2%
            self.label_bio_resultado.setText(
                f"⚠️ CAMBIO MÍNIMO\n\n"
                f"Cambio: {cambio_porcentual:.1f}%\n"
                f"({cambio_absoluto:.1f} Ω)\n\n"
                f"No se detectó cambio significativo"
            )
            self.label_bio_resultado.setStyleSheet("""
                QLabel {
                    background-color: #f59e0b;
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-weight: bold;
                }
            """)
            self.log_mensaje(f"⚠️ Cambio: {cambio_porcentual:.1f}% ({cambio_absoluto:.1f} Ω)")
            self.log_mensaje("⚠️ Cambio no significativo")
        
        self.label_bio_resultado.setVisible(True)
        
        self.label_bio_estado.setText("✅ Medición POST completa")
        self.label_bio_estado.setStyleSheet("""
            QLabel {
                background-color: #10b981;
                color: white;
                padding: 10px;
                border-radius: 8px;
            }
        """)
        
        self.midiendo_impedancia = False
        
        # Proceder al análisis de EMG
        self.analizar_serie_manual()
    
    def verificar_timeout_impedancia(self):
        """Verifica si ha pasado el timeout esperando medición de impedancia"""
        if self.esperando_impedancia and self.tiempo_inicio_medicion is not None:
            tiempo_transcurrido = time.time() - self.tiempo_inicio_medicion
            if tiempo_transcurrido > self.timeout_impedancia:
                self.log_mensaje(f"❌ Timeout: No se recibió impedancia en {self.timeout_impedancia}s")
                self.cancelar_medicion_impedancia("Timeout esperando respuesta del ESP32")
    
    def cancelar_medicion_impedancia(self, motivo):
        """Cancela la medición de impedancia en curso"""
        self.esperando_impedancia = False
        self.midiendo_impedancia = False
        self.tiempo_inicio_medicion = None
        
        self.progress_bio.setVisible(False)
        self.progress_bio.setRange(0, 100)
        
        self.label_bio_estado.setText(f"❌ Error en medición\n\n{motivo}")
        self.label_bio_estado.setStyleSheet("""
            QLabel {
                background-color: #ef4444;
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }
        """)
        
        self.log_mensaje(f"❌ Medición cancelada: {motivo}")
        
        # Resetear estado
        self.detectando = False
        self.grabando_serie = False
        self.serie_buffer_actual = []
        
        self.btn_detectar.setText("🚀 INICIAR SERIE")
        self.btn_detectar.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #11998e, stop:1 #38ef7d);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #38ef7d, stop:1 #11998e);
            }
        """)
        self.btn_detectar.setEnabled(True)
        self.btn_calibrar.setEnabled(True)
    
    def init_ui(self):
        """Crear interfaz gráfica mejorada"""
        # Crear scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #1a1a2e;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #16213e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #667eea;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #764ba2;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Widget central con el contenido
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(25, 20, 25, 25)
        
        # Configurar scroll area
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        
        # ===== HEADER COMPACTO =====
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 12px;
                padding: 12px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(2)
        
        titulo = QLabel("⚡ MEASUTRAINER")
        titulo.setFont(QFont('Segoe UI', 22, QFont.Bold))
        titulo.setStyleSheet("color: white; background: transparent;")
        titulo.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(titulo)
        
        subtitulo = QLabel("Real-time Biceps Training Classification with Bioimpedance")
        subtitulo.setFont(QFont('Segoe UI', 11))
        subtitulo.setStyleSheet("color: rgba(255, 255, 255, 0.85); background: transparent;")
        subtitulo.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitulo)
        
        main_layout.addWidget(header_frame)
        
        # ===== CARDS DE ESTADO =====
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)
        
        self.card_conexion = self.crear_card_compacta("🔌", "Desconectado", "#ef4444")
        cards_layout.addWidget(self.card_conexion)
        
        self.card_modelo = self.crear_card_compacta(
            "🧠",
            "Cargado" if self.modelo_cargado else "Error",
            "#10b981" if self.modelo_cargado else "#ef4444"
        )
        cards_layout.addWidget(self.card_modelo)
        
        self.card_mvc = self.crear_card_compacta("📊", "Pendiente", "#f59e0b")
        cards_layout.addWidget(self.card_mvc)
        
        self.card_series = self.crear_card_compacta("🎯", "0 Series", "#6366f1")
        cards_layout.addWidget(self.card_series)
        
        main_layout.addLayout(cards_layout)
        
        # ===== ESTADO PRINCIPAL =====
        self.label_estado = QLabel("⏳ Esperando calibración...")
        self.label_estado.setFont(QFont('Segoe UI', 13, QFont.Bold))
        self.label_estado.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4facfe, stop:1 #00f2fe);
                color: white;
                padding: 15px;
                border-radius: 10px;
            }
        """)
        self.label_estado.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label_estado)
        
        # ===== LAYOUT HORIZONTAL PARA GRÁFICAS =====
        graficas_layout = QHBoxLayout()
        graficas_layout.setSpacing(15)
        
        # ===== GRÁFICA EMG =====
        grafica_emg_frame = QFrame()
        grafica_emg_frame.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-radius: 12px;
                padding: 12px;
            }
        """)
        grafica_emg_layout = QVBoxLayout(grafica_emg_frame)
        grafica_emg_layout.setSpacing(8)
        
        grafica_emg_titulo = QLabel("📈 Señal EMG en Tiempo Real")
        grafica_emg_titulo.setFont(QFont('Segoe UI', 12, QFont.Bold))
        grafica_emg_titulo.setStyleSheet("color: #eee; background: transparent;")
        grafica_emg_layout.addWidget(grafica_emg_titulo)
        
        self.figure = Figure(figsize=(9, 6), facecolor='#16213e')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#0f3460')
        self.ax.set_title('Amplitud de Señal Filtrada (20-45 Hz)', 
                         fontsize=11, color='white', pad=12, fontweight='bold')
        self.ax.set_xlabel('Muestras', fontsize=10, color='white')
        self.ax.set_ylabel('Amplitud', fontsize=10, color='white')
        self.ax.set_ylim(0, 800)
        self.ax.set_xlim(0, 1000)
        self.ax.grid(True, alpha=0.25, color='white', linestyle='--', linewidth=0.8)
        self.ax.tick_params(colors='white', labelsize=9)
        
        for spine in self.ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1.5)
        
        self.umbral_inicio_line = self.ax.axhline(y=self.UMBRAL_INICIO, 
                                                  color='#10b981', linestyle='--', 
                                                  linewidth=2, alpha=0.8, label='Umbral Inicio')
        self.umbral_fin_line = self.ax.axhline(y=self.UMBRAL_FIN, 
                                               color='#f59e0b', linestyle='--', 
                                               linewidth=2, alpha=0.8, label='Umbral Fin')
        
        self.line, = self.ax.plot([], [], '#00f2fe', linewidth=2, label='Señal EMG')
        self.ax.legend(loc='upper right', facecolor='#0f3460', 
                      edgecolor='white', fontsize=9, framealpha=0.9)
        
        self.figure.tight_layout()
        grafica_emg_layout.addWidget(self.canvas)
        graficas_layout.addWidget(grafica_emg_frame, 65)  # 65% del espacio
        
        # ===== PANEL DE BIOIMPEDANCIA =====
        bioimpedancia_frame = QFrame()
        bioimpedancia_frame.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-radius: 12px;
                padding: 12px;
            }
        """)
        bioimpedancia_layout = QVBoxLayout(bioimpedancia_frame)
        bioimpedancia_layout.setSpacing(12)
        
        bio_titulo = QLabel("⚡ Bioimpedancia")
        bio_titulo.setFont(QFont('Segoe UI', 12, QFont.Bold))
        bio_titulo.setStyleSheet("color: #eee; background: transparent;")
        bioimpedancia_layout.addWidget(bio_titulo)
        
        # Estado de bioimpedancia
        self.label_bio_estado = QLabel("Esperando serie...")
        self.label_bio_estado.setFont(QFont('Segoe UI', 10))
        self.label_bio_estado.setStyleSheet("""
            QLabel {
                background-color: #0f3460;
                color: #aaa;
                padding: 10px;
                border-radius: 8px;
            }
        """)
        self.label_bio_estado.setAlignment(Qt.AlignCenter)
        self.label_bio_estado.setWordWrap(True)
        bioimpedancia_layout.addWidget(self.label_bio_estado)
        
        # Progress bar
        self.progress_bio = QProgressBar()
        self.progress_bio.setStyleSheet("""
            QProgressBar {
                background-color: #0f3460;
                border: 2px solid #1a4d7a;
                border-radius: 8px;
                text-align: center;
                color: white;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 6px;
            }
        """)
        self.progress_bio.setVisible(False)
        bioimpedancia_layout.addWidget(self.progress_bio)
        
        # Medición PRE
        frame_pre = QFrame()
        frame_pre.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border: 2px solid #1a4d7a;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout_pre = QVBoxLayout(frame_pre)
        
        label_pre_titulo = QLabel("📊 Impedancia PRE")
        label_pre_titulo.setFont(QFont('Segoe UI', 10, QFont.Bold))
        label_pre_titulo.setStyleSheet("color: #00f2fe; background: transparent;")
        layout_pre.addWidget(label_pre_titulo)
        
        self.label_z_pre = QLabel("-- Ω")
        self.label_z_pre.setFont(QFont('Segoe UI', 16, QFont.Bold))
        self.label_z_pre.setStyleSheet("color: #10b981; background: transparent;")
        self.label_z_pre.setAlignment(Qt.AlignCenter)
        layout_pre.addWidget(self.label_z_pre)
        
        bioimpedancia_layout.addWidget(frame_pre)
        
        # Medición POST
        frame_post = QFrame()
        frame_post.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border: 2px solid #1a4d7a;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout_post = QVBoxLayout(frame_post)
        
        label_post_titulo = QLabel("📊 Impedancia POST")
        label_post_titulo.setFont(QFont('Segoe UI', 10, QFont.Bold))
        label_post_titulo.setStyleSheet("color: #00f2fe; background: transparent;")
        layout_post.addWidget(label_post_titulo)
        
        self.label_z_post = QLabel("-- Ω")
        self.label_z_post.setFont(QFont('Segoe UI', 16, QFont.Bold))
        self.label_z_post.setStyleSheet("color: #f59e0b; background: transparent;")
        self.label_z_post.setAlignment(Qt.AlignCenter)
        layout_post.addWidget(self.label_z_post)
        
        bioimpedancia_layout.addWidget(frame_post)
        
        # Resultado de comparación
        self.label_bio_resultado = QLabel("")
        self.label_bio_resultado.setFont(QFont('Segoe UI', 9, QFont.Bold))
        self.label_bio_resultado.setStyleSheet("""
            QLabel {
                background-color: #0f3460;
                padding: 10px;
                border-radius: 8px;
            }
        """)
        self.label_bio_resultado.setAlignment(Qt.AlignCenter)
        self.label_bio_resultado.setWordWrap(True)
        self.label_bio_resultado.setVisible(False)
        bioimpedancia_layout.addWidget(self.label_bio_resultado)
        
        bioimpedancia_layout.addStretch()
        
        graficas_layout.addWidget(bioimpedancia_frame, 35)  # 35% del espacio
        
        main_layout.addLayout(graficas_layout)
        
        # ===== BOTONES DE CONTROL =====
        botones_layout = QHBoxLayout()
        botones_layout.setSpacing(15)
        
        self.btn_calibrar = QPushButton("🎯 CALIBRAR MVC")
        self.btn_calibrar.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.btn_calibrar.setMinimumHeight(55)
        self.btn_calibrar.setCursor(Qt.PointingHandCursor)
        self.btn_calibrar.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #764ba2, stop:1 #667eea);
            }
            QPushButton:pressed {
                background: #5a4d8a;
            }
            QPushButton:disabled {
                background: #2d2d44;
                color: #666;
            }
        """)
        self.btn_calibrar.clicked.connect(self.iniciar_calibracion)
        botones_layout.addWidget(self.btn_calibrar)
        
        self.btn_detectar = QPushButton("🚀 INICIAR SERIE")
        self.btn_detectar.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.btn_detectar.setMinimumHeight(55)
        self.btn_detectar.setCursor(Qt.PointingHandCursor)
        self.btn_detectar.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #11998e, stop:1 #38ef7d);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #38ef7d, stop:1 #11998e);
            }
            QPushButton:pressed {
                background: #0d7a6f;
            }
            QPushButton:disabled {
                background: #2d2d44;
                color: #666;
            }
        """)
        self.btn_detectar.setEnabled(False)
        self.btn_detectar.clicked.connect(self.toggle_deteccion)
        botones_layout.addWidget(self.btn_detectar)
        
        main_layout.addLayout(botones_layout)
        
        # ===== PANEL DE RESULTADOS =====
        resultados_frame = QFrame()
        resultados_frame.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-radius: 12px;
                padding: 12px;
            }
        """)
        resultados_layout = QVBoxLayout(resultados_frame)
        resultados_layout.setSpacing(8)
        
        resultados_titulo = QLabel("📋 Registro de Actividad")
        resultados_titulo.setFont(QFont('Segoe UI', 12, QFont.Bold))
        resultados_titulo.setStyleSheet("color: #eee; background: transparent;")
        resultados_layout.addWidget(resultados_titulo)
        
        self.texto_resultados = QTextEdit()
        self.texto_resultados.setReadOnly(True)
        self.texto_resultados.setFont(QFont('Consolas', 9))
        self.texto_resultados.setStyleSheet("""
            QTextEdit {
                background-color: #0f3460;
                color: #00f2fe;
                border: 2px solid #1a4d7a;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self.texto_resultados.setMaximumHeight(120)
        resultados_layout.addWidget(self.texto_resultados)
        
        main_layout.addWidget(resultados_frame)
        
        # Mensaje inicial
        self.log_mensaje("╔═══════════════════════════════════════════════════════╗")
        self.log_mensaje("⚡ MEASUTRAINER - SISTEMA INICIADO")
        self.log_mensaje("╚═══════════════════════════════════════════════════════╝")
        self.log_mensaje(f"🧠 Modelo: {'Cargado' if self.modelo_cargado else 'Error'}")
        self.log_mensaje("📡 Buscando dispositivo Bluetooth...")
        self.log_mensaje("╚═══════════════════════════════════════════════════════╝")
        self.log_mensaje("")
    
    def crear_card_compacta(self, emoji, valor, color):
        """Crear card informativa compacta"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #16213e;
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 8px 12px;
            }}
        """)
        card_layout = QHBoxLayout(card)
        card_layout.setSpacing(8)
        card_layout.setContentsMargins(5, 5, 5, 5)
        
        card_emoji = QLabel(emoji)
        card_emoji.setFont(QFont('Segoe UI', 16))
        card_emoji.setStyleSheet("background: transparent;")
        card_layout.addWidget(card_emoji)
        
        card_valor = QLabel(valor)
        card_valor.setFont(QFont('Segoe UI', 11, QFont.Bold))
        card_valor.setStyleSheet(f"color: {color}; background: transparent;")
        card_valor.setObjectName("card_valor")
        card_layout.addWidget(card_valor)
        
        return card
    
    def actualizar_card(self, card, valor, color):
        """Actualizar valor y color de un card"""
        card_valor = card.findChild(QLabel, "card_valor")
        if card_valor:
            card_valor.setText(valor)
            card_valor.setStyleSheet(f"color: {color}; background: transparent;")
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #16213e;
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 8px 12px;
            }}
        """)
    
    def log_mensaje(self, mensaje):
        """Agregar mensaje al log con timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.texto_resultados.append(f"[{timestamp}] {mensaje}")
    
    def analizar_serie_manual(self):
        """Analizar serie completa al detener manualmente"""
        if len(self.serie_buffer_actual) < 10:
            self.label_estado.setText("⚠️ Serie muy corta para analizar")
            self.label_estado.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f59e0b, stop:1 #fb923c);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                }
            """)
            self.log_mensaje("⚠️ Serie muy corta - No se puede analizar")
            
            # Resetear botón
            self.btn_detectar.setText("🚀 INICIAR SERIE")
            self.btn_detectar.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #11998e, stop:1 #38ef7d);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 12px 25px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #38ef7d, stop:1 #11998e);
                }
            """)
            self.btn_detectar.setEnabled(True)
            self.btn_calibrar.setEnabled(True)
            return
        
        self.label_estado.setText("⚙️ ANALIZANDO SERIE...")
        self.label_estado.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f093fb, stop:1 #f5576c);
                color: white;
                padding: 15px;
                border-radius: 10px;
            }
        """)
        
        try:
            serie_array = np.array(self.serie_buffer_actual)
            
            duracion_s = len(serie_array) / self.fs
            rms_crudo = np.sqrt(np.mean(serie_array**2))
            mav_crudo = np.mean(np.abs(serie_array))
            
            freqs, psd = welch(serie_array, fs=self.fs, nperseg=min(256, len(serie_array)))
            cumsum_psd = np.cumsum(psd)
            total_power = cumsum_psd[-1]
            mdf_hz = freqs[np.where(cumsum_psd >= total_power/2)[0][0]]
            mnf_hz = np.sum(freqs * psd) / np.sum(psd)
            
            rms_normaliz = (rms_crudo / self.mvc_rms_usuario) * 100.0
            mav_normaliz = (mav_crudo / self.mvc_rms_usuario) * 100.0
            
            X_input = np.array([[duracion_s, rms_normaliz, mav_normaliz, mdf_hz, mnf_hz]])
            X_scaled = self.scaler.transform(X_input)
            prediccion = self.model.predict(X_scaled)[0]
            
            self.series_detectadas += 1
            self.ultima_prediccion = prediccion
            
            self.actualizar_card(self.card_series, f"{self.series_detectadas} Series", "#6366f1")
            
            emoji_prediccion = {"Fuerza": "💪", "Hipertrofia": "🏋️", "Resistencia": "🔥"}
            emoji = emoji_prediccion.get(prediccion, "🎯")
            
            self.log_mensaje("╔═══════════════════════════════════════════════════════╗")
            self.log_mensaje(f"⚡ SERIE #{self.series_detectadas} ANALIZADA")
            self.log_mensaje("╠═══════════════════════════════════════════════════════╣")
            self.log_mensaje(f"⏱️  Duración: {duracion_s:.2f} s")
            self.log_mensaje(f"📊 RMS Normalizado: {rms_normaliz:.2f}% MVC")
            self.log_mensaje(f"📊 MAV Normalizado: {mav_normaliz:.2f}% MVC")
            self.log_mensaje(f"🎵 MDF: {mdf_hz:.2f} Hz")
            self.log_mensaje(f"🎵 MNF: {mnf_hz:.2f} Hz")
            self.log_mensaje("─────────────────────────────────────────────────────────")
            self.log_mensaje(f"🎯 RESULTADO: {emoji} {prediccion.upper()}")
            self.log_mensaje("╚═══════════════════════════════════════════════════════╝")
            self.log_mensaje("")
            
            self.label_estado.setText(f"✅ {emoji} RESULTADO: {prediccion.upper()}")
            self.label_estado.setStyleSheet(f"""
                QLabel {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #11998e, stop:1 #38ef7d);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    font-weight: bold;
                }}
            """)
            
            # Resetear botón para siguiente serie
            self.btn_detectar.setText("🚀 INICIAR SERIE")
            self.btn_detectar.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #11998e, stop:1 #38ef7d);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 12px 25px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #38ef7d, stop:1 #11998e);
                }
            """)
            self.btn_detectar.setEnabled(True)
            self.btn_calibrar.setEnabled(True)
            
        except Exception as e:
            self.log_mensaje(f"✗ Error en análisis: {e}")
            self.label_estado.setText("⚠️ Error en análisis")
            
            # Resetear botón
            self.btn_detectar.setText("🚀 INICIAR SERIE")
            self.btn_detectar.setEnabled(True)
            self.btn_calibrar.setEnabled(True)
    
    def iniciar_calibracion(self):
        """Iniciar calibración MVC"""
        self.btn_calibrar.setEnabled(False)
        self.log_mensaje("🎯 INICIANDO CALIBRACIÓN MVC")
        self.log_mensaje("⏳ Prepárese para realizar contracción máxima en 3 segundos...")
        
        self.label_estado.setText("⏳ PREPARANDO CALIBRACIÓN (3s)...")
        self.label_estado.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f093fb, stop:1 #f5576c);
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-weight: bold;
            }
        """)
        
        QTimer.singleShot(3000, self.ejecutar_calibracion)
    
    def ejecutar_calibracion(self):
        """Ejecutar calibración de 5 segundos"""
        self.calibrando = True
        self.calibracion_buffer = []
        
        self.label_estado.setText("🔴 ¡CONTRACCIÓN MÁXIMA AHORA! (5s)")
        self.label_estado.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ee0979, stop:1 #ff6a00);
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        self.log_mensaje("⚡ ¡AHORA! Contrayendo durante 5 segundos...")
        
        QTimer.singleShot(5000, self.finalizar_calibracion)
    
    def finalizar_calibracion(self):
        """Finalizar calibración"""
        self.calibrando = False
        
        if len(self.calibracion_buffer) > 0:
            calibracion_array = np.array(self.calibracion_buffer)
            self.mvc_rms_usuario = np.sqrt(np.mean(calibracion_array**2))
            
            self.calibracion_completada = True
            self.btn_detectar.setEnabled(True)
            
            self.actualizar_card(self.card_mvc, f"MVC: {self.mvc_rms_usuario:.1f}", "#10b981")
            
            self.log_mensaje("✅ ¡CALIBRACIÓN COMPLETADA!")
            self.log_mensaje(f"📊 MVC RMS: {self.mvc_rms_usuario:.2f}")
            self.log_mensaje("🚀 Sistema listo")
            self.log_mensaje("")
            
            self.label_estado.setText(f"✅ CALIBRACIÓN COMPLETA - Listo para iniciar")
            self.label_estado.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #11998e, stop:1 #38ef7d);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    font-weight: bold;
                }
            """)
        else:
            self.log_mensaje("✗ Error: No se capturaron datos")
            self.label_estado.setText("⚠️ Error en calibración - Verifique la conexión")
        
        self.btn_calibrar.setEnabled(True)
    
    def toggle_deteccion(self):
        """Iniciar/detener serie manualmente"""
        if not self.detectando:
            # INICIAR PROCESO: Primero medir impedancia PRE
            self.log_mensaje("🔴 SERIE INICIADA")
            
            # Resetear valores de bioimpedancia
            self.z_pre = None
            self.z_post = None
            self.label_z_pre.setText("-- Ω")
            self.label_z_post.setText("-- Ω")
            self.label_bio_resultado.setVisible(False)
            
            # Iniciar medición de impedancia PRE
            self.iniciar_medicion_impedancia_pre()
            
        else:
            # FINALIZAR SERIE: Detener EMG y medir impedancia POST
            self.detectando = False
            self.grabando_serie = False
            
            self.log_mensaje("⏹️ Serie finalizada")
            
            self.label_estado.setText("⏹️ Serie detenida - Midiendo impedancia POST...")
            self.label_estado.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f59e0b, stop:1 #fb923c);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                }
            """)
            
            # Iniciar medición de impedancia POST
            QTimer.singleShot(500, self.iniciar_medicion_impedancia_post)
    
    def actualizar_grafica(self):
        """Actualizar gráfica en tiempo real"""
        if len(self.signal_buffer) > 0:
            datos = list(self.signal_buffer)
            x_data = range(len(datos))
            self.line.set_data(x_data, datos)
            self.ax.set_xlim(0, len(self.signal_buffer))
            
            if len(datos) > 10 and max(datos) > 0:
                max_val = max(datos)
                self.ax.set_ylim(0, max(800, max_val * 1.2))
            
            self.canvas.draw()
            self.canvas.flush_events()
    
    def closeEvent(self, event):
        """Cerrar puerto serie al cerrar aplicación"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✓ Puerto serie cerrado")
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ventana = EMGDetectorApp()
    ventana.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()