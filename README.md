# 🧠 Proyecto Final: Modelo Predictivo de Burnout

Este repositorio contiene el código y los recursos utilizados para el **Proyecto Final Integrador del Diplomado en Ciencia de Datos y Análisis Avanzado**.  
El objetivo es **predecir el riesgo de burnout en una población de estudio (médicos) en CABA**, trabajando con técnicas avanzadas de Machine Learning y Deep Learning, siguiendo un flujo completo utilizando la metodología CRISP-DM para el enfoque y la resolución del problema planteado.

---

## 📂 Contenido del Repositorio

| Archivo | Descripción |
|----------|--------------|
| **Exploracion de datos.ipynb** | Script Python para la *Exploración de Datos*. Realiza la carga de la base, limpieza inicial (eliminación de columnas irrelevantes, manejo de valores 999), cálculo de estadísticas descriptivas (media, varianza, asimetría, curtosis), identificación de duplicados y generación de visualizaciones de distribuciones y correlaciones. |
| **DNN-final.ipynb** | Implementación del modelo **Deep Neural Network (DNN)** para clasificación (*MBI_Burnout*) y regresión (*MBI_RiesgoBurnout1*). Incluye preprocesamiento, escalado, aplicación de SMOTE y entrenamiento con *K-Fold Cross Validation*. |
| **KNN Proyecto Final.ipynb** | Implementación del modelo **K-Nearest Neighbors (KNN)**. Detalla la selección de características basada en importancia (*Random Forest*), reducción de dimensionalidad y el uso de *RobustScaler* y *RandomOverSampler*. |
| **SVM Proyecto Final.ipynb** | Implementación del modelo **Support Vector Machine (SVM)**. Se enfoca en la selección de las 30 mejores características mediante *SelectKBest* y la optimización de hiperparámetros (*GridSearchCV*). |
| **xgboost-final.ipynb** | Implementación del modelo **XGBoost** para clasificación y regresión. Realiza la imputación y conversión de tipos, transformación logarítmica para manejar la asimetría y análisis de la importancia de características. |
| **Modelo arbol decision.csv** | Implementación del modelo **Árbol de decisión**. Muestra los nodos con las decisiones de clasificación basadas en los factores más relevantes. Se prioriza la interpretabilidad del modelo, permitiendo visualizar las divisiones y umbrales clave que determinan los niveles de riesgo. |
| **Regresion Logistica.csv** | Implementación del modelo **Regresión Logistíca**. Se analizan los coeficientes para identificar el peso de cada predictor sobre la probabilidad de presentar Burnout, aportando una base comparativa frente a modelos más complejos. |
| **Proyecto_final.csv** | Dataset de entrada utilizado para todo el análisis. |
| **Encuesta_Final_SinProcesar.csv** | Datos iniciales sin procesamiento. Contiene la explicación de las variables. |



---

## ⚙️ Requisitos

Para ejecutar los archivos de tipo notebook, se requiere un entorno configurado con las siguientes dependencias:

- **Python 3.11 o superior (recomendado)**
- **Librerías de análisis y visualización:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Librerías de Machine Learning:** `scikit-learn`
- **Librerías de Modelos avanzados:** `xgboost`, `tensorflow`, `keras`
- **Librerías de Desbalanceo:** `imblearn` (*para SMOTE y RandomOverSampler*)
- **Utilidades:** `pickle`

---

## 🧩 Instalación

> 💡 Se recomienda crear un entorno virtual antes de instalar las dependencias.

Ejecutar el siguiente comando de pip para realizar la instalación de las dependencias necesarias para ejecutar los notebooks del proyecto.
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imblearn tensorflow keras xgboost
```

## 🗃️ Descarga de Datos
Debes asegurarte de que el dataset Proyecto_final.csv esté disponible en la ruta de ejecución o en la ruta especificada dentro del script de exploración, como **/content/drive/MyDrive/Proyecto_final.csv** si se usa Google Colab como entorno virtual.

---

## 🚀 Ejecución del Código

El proyecto sigue un flujo que comienza con la inspección de los datos y continúa con el entrenamiento y la evaluación de modelos.

### 1️⃣ Exploración y Limpieza

- Abre y ejecuta **`Exploracion de datos.ipynb`**.  
- Este script carga el dataset y lo limpia inicialmente, eliminando identificadores y tratando los valores **999** (que representan nulos o valores especiales).  
- Genera estadísticas descriptivas (como **asimetría** y **curtosis**) y gráficos de distribución y correlación para entender las relaciones entre las variables, especialmente las relacionadas con el **MBI** (*Maslach Burnout Inventory*).

---

### 2️⃣ Modelado y Entrenamiento (Independiente)

- Ejecuta cada uno de los archivos de modelo (**`*final.ipynb`** y **`*Proyecto Final.ipynb`**).  
- Cada script se encarga de realizar el preprocesamiento específico que requiere, incluyendo:
  - Selección de características clave (como *SelectKBest* en SVM o la importancia de *Random Forest* en KNN).  
  - Aplicación de técnicas de balanceo (*SMOTE* o *RandomOverSampler*).  
  - Entrenamiento robusto mediante validación cruzada (*Cross Validation*).  


---

## 📈 Métricas de Evaluación Utilizadas

Los modelos fueron entrenados y evaluados utilizando las siguientes métricas, adecuadas para sus respectivas tareas de **Clasificación** (predicción de `MBI_Burnout`) y **Regresión** (predicción de `MBI_RiesgoBurnout1`):

### 🔹 Clasificación (DNN, KNN, SVM, XGBoost)
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **ROC-AUC** (*Area Under the Receiver Operating Characteristic Curve*)  
- **Loss** (*binary_crossentropy*)

### 🔹 Regresión (DNN, XGBoost)
- **R² Score**  
- **MAE** (*Mean Absolute Error*)  
- **MSE** (*Mean Squared Error*)  
- **RMSE** (*Root Mean Squared Error*)

---

## 📚 Citaciones

- **Datos:** [PLACEHOLDER: Fuente original del dataset `Proyecto_final.csv`]  

- **Modelos de Referencia:** [PLACEHOLDER: Referencias clave para MBI, PERMA, LGS, etc.]
- **MBI (Maslach Burnout Inventory):**  
  Maslach, C., Jackson, S. E., & Leiter, M. P. (1996). *Maslach Burnout Inventory Manual* (3rd ed.). Consulting Psychologists Press.

- **PERMA (Modelo de Bienestar):**  
  Seligman, M. E. P. (2011). *Flourish: A Visionary New Understanding of Happiness and Well-being.* Free Press.

- **LGS (Loyola Generativity Scale):**  
  McAdams, D. P., & de St. Aubin, E. (1992). *A theory of generativity and its assessment through self-report, behavioral acts, and narrative themes in autobiography.* Journal of Personality and Social Psychology, 62(6), 1003–1015.

- **Tecnologías:** Documentación de [Python](https://www.python.org/), [Scikit-learn](https://scikit-learn.org/stable/), [TensorFlow / Keras](https://www.tensorflow.org/) y [XGBoost](https://xgboost.readthedocs.io/).

---
