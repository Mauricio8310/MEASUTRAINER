# 🧠 MEASUTRAINER
**Monitor de Actividad Muscular para Ejercicio y Rehabilitación**

## 👥 Autores
Mauricio Gael Uribe Ramírez  
Axel Jared Herrera Moreno  
Carlos Antonio Cortes Ríos  
**Asesora:** Dra. Laura Paulina Osuna Carrasco  
Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI), Universidad de Guadalajara

---

## 📖 Descripción del Proyecto

**MEASUTRAINER** es un dispositivo portátil diseñado para analizar la actividad muscular del bíceps braquial mediante la **integración simultánea de electromiografía (EMG)** y **bioimpedancia eléctrica (BIA)**.  
El sistema permite diferenciar de manera objetiva los tipos de entrenamiento muscular —**fuerza, hipertrofia y resistencia**— a través del procesamiento de señales y visualización en tiempo real mediante una interfaz desarrollada en **Python**.

A diferencia de los métodos de evaluación tradicionales basados en observación clínica, MEASUTRAINER proporciona **métricas cuantitativas y reproducibles** que ayudan a **optimizar procesos de rehabilitación** y entrenamiento físico.  
Su enfoque combina hardware especializado (módulos **OLIMEX EKG-EMG** y **AD5933**) con un **ESP32** como microcontrolador principal, lo que garantiza portabilidad, conectividad y precisión en la adquisición de datos.

---

## ⚙️ Metodología

El desarrollo del proyecto se estructuró en cuatro fases principales:

1. **Investigación y análisis de requerimientos:**  
   Revisión de literatura sobre EMG y bioimpedancia muscular, identificación de parámetros fisiológicos relevantes.
2. **Diseño del sistema:**  
   Selección de módulos electrónicos, arquitectura de adquisición y procesamiento de señales.
3. **Implementación y desarrollo:**  
   Programación de microcontrolador (ESP32), adquisición de señales, y desarrollo de la interfaz de análisis en Python.
4. **Validación experimental:**  
   Pruebas controladas en bíceps braquial para comparar la capacidad de clasificación entre fuerza, hipertrofia y resistencia.

---

## 🧩 Resultados

- Base de datos con **125 registros EMG** balanceados entre los tres tipos de entrenamiento.
- Clasificador **Random Forest** entrenado y validado (precisión promedio **78.95%**).  
- Identificación de parámetros más relevantes:  
  **RMS_normalized_%MVC**, **MNF_Hz**, y **MAV_normalized_%MVC**.
- Visualización en tiempo real de la actividad muscular, espectros de bioimpedancia y detección de fatiga.

El sistema demostró **precisión superior al 75%** en la diferenciación de modalidades de entrenamiento y **sensibilidad un 34% mayor** que los métodos EMG convencionales en la detección de fatiga muscular.

---

## 💡 Innovación y Aplicaciones

- Integración simultánea de EMG y BIA en un solo dispositivo portátil.  
- Análisis dual de señales para caracterización completa del estado muscular.  
- Aplicaciones directas en:
  - Rehabilitación física y fisioterapia  
  - Entrenamiento deportivo  
  - Evaluación de fatiga y prevención de lesiones  
  - Telemedicina y dispositivos vestibles

---

## 🌍 Impacto

El proyecto **democratiza el acceso** a tecnologías avanzadas de evaluación muscular, facilitando su uso en clínicas, centros deportivos y entornos domésticos.  
Su enfoque de bajo costo y alto valor clínico contribuye a mejorar los resultados de rehabilitación y reducir el riesgo de lesiones.

---

## 🔬 Referencias

[1] S. H. Park, S. Lee, J. Kim and Y. T. Kim, "A Wearable Multi-Frequency Device to Measure Muscle Activity Combining Simultaneous Electromyography and Electrical Impedance Myography," *2022 IEEE International Symposium on Medical Measurements and Applications (MeMeA)*, Messina, Italy, 2022, pp. 1-6, doi: 10.1109/MeMeA54994.2022.9856554.

[2] S. H. Park, S. Lee and Y. T. Kim, "A Wearable Dual-Channel Bioimpedance Spectrometer for Real-Time Muscle Contraction Detection," *IEEE Transactions on Biomedical Circuits and Systems*, vol. 18, no. 1, pp. 35–46, Feb. 2024, doi: 10.1109/TBCAS.2023.3332174.

[3] T. J. Freeborn and B. Fu, "Time-course bicep tissue bioimpedance changes throughout a fatiguing exercise protocol," *Medical Engineering & Physics*, vol. 69, pp. 109–115, Jul. 2019, doi:10.1016/j.medengphy.2019.04.006. [Online]. Available: https://pubmed.ncbi.nlm.nih.gov/31056402

[4] D. Borms, I. Ackerman, P. Smets, G. Van den Berge, and A. M. Cools, "Biceps Disorder Rehabilitation for the Athlete: A Continuum of Moderate- to High-Load Exercises," *The American Journal of Sports Medicine*, vol. 45, no. 3, pp. 642–650, Mar. 2017, doi:10.1177/03635465166.

---
