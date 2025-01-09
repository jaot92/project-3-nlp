# Clasificador de Traducciones NLP

## Descripción
Proyecto de procesamiento de lenguaje natural para clasificar traducciones entre automáticas y humanas.

## Estructura del Proyecto
```
.
├── README.md
├── requirements.txt
├── setup.py
└── src/
    ├── data/
    │   ├── TRAINING_DATA.txt
    │   └── REAL_DATA.txt
    ├── models/
    │   └── train_models.py
    └── notebooks/
        └── 01_exploratory_analysis.ipynb
```

## Configuración del Entorno

### Prerequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- git

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd project-3-nlp
```

2. Crear y activar entorno virtual:
```bash
# En Windows:
python -m venv venv
venv\Scripts\activate

# En macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt

# Instalar modelo de spaCy en español
python -m spacy download es_core_news_md
```

### Verificación de la Instalación
```bash
python
>>> import spacy
>>> nlp = spacy.load('es_core_news_md')
>>> exit()
```

## Uso del Proyecto

1. **Análisis Exploratorio**:
   - Abrir Jupyter Notebook:
   ```bash
   jupyter notebook src/notebooks/01_exploratory_analysis.ipynb
   ```

2. **Entrenamiento de Modelos**:
   ```bash
   python src/models/train_models.py
   ```

## Solución de Problemas Comunes

1. **Error con spaCy**:
   ```bash
   python -m spacy download es_core_news_md
   ```

2. **Error con Jupyter**:
   ```bash
   pip install notebook
   ```

## Contacto
[Jose Ortiz] - [jose.ortiz@ironhack.com]