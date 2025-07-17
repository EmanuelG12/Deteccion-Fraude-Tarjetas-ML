# Proyecto de clasificación - detección de fraude

### 🎯 Descripción del proyecyo

> Proyecto de clasificación de detección de fraude en tarjetas de crédito utilizando modelos de Machine Learning tradicionales (LogisticRegression, RandomForestClassifier, XGBClassifier), con visualizaciones interactivas y métricas de evaluación detalladas.

---

## 📁 Estructura del repositorio

```
├── artifacts/                    # ✅ Modelos y artefactos entrenados
│   ├── best_model.pkl
│   └── preprocessor.pkl
│
├── data/                         # ✅ Datos crudos y procesados
│   ├── raw/
│   └── processed/
│
├── notebooks/                    # ✅ Exploración y experimentación
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 03_Modelado.ipynb
│   ├── 04_Evaluación.ipynb
│   └── 05_Deploy.ipynb
│
├── src/                          
│   ├── data/
│   │   └── make_dataset.py       # Función para cargar/split datos
│   ├── features/
│   │   └── build_features.py     # Preprocesador / feature engineering
│   ├── models/
│   │   ├── train_model.py        # Script de entrenamiento
│   │   ├── evaluate_model.py     # Evalúa el modelo guardado
│   │   └── predict_model.py      # Predice para nuevos datos
│   └── utils/
│       └── helpers.py            # Funciones auxiliares (métricas, validaciones, etc.)
│
├── venv/                         # ⚠️ Virtualenv local (no subir a Git)
├── requirements.txt              # ✅ Dependencias del proyecto
├── .gitignore                    # ✅ Para ignorar venv/, .pkl, etc.
├── README.md                     # ✅ Explicación del proyecto
```

---

## 🧪 Tecnologías y herramientas

- Python 3.13
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- XGBoost / LightGBM
- Streamlit

---

##  Dataset

**Nombre:** Credit Card Fraud Detection Predictive Models

**Fuente:** [Kaggle]. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Tamaño:** 284807 filas, 31 columnas

**Descripción:** Clasificación de transacciones como Fraudulentas o No fraudulentas.

---

## 🔍 Exploración de Datos (EDA)

- Distribuciones y outliers
- Análisis de correlaciones
- Valores faltantes
- Ingeniería de features
- Desbalanceo de los datos
- Encoding y escalado

---

## 📫 Contacto

📧 Emanuelgalloit@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/emanuel-gallo-0abab71ba/)

🐙 [GitHub](https://github.com/EmanuelG12)
