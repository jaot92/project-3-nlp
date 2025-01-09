import joblib
import pandas as pd
import logging
from train_models import TranslationClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models(models_dir):
    """Cargar modelos entrenados y pipeline"""
    logger.info("Cargando modelos...")
    
    classifier = TranslationClassifier()
    
    # Cargar pipeline
    classifier.pipeline = joblib.load(f"{models_dir}/pipeline.joblib")
    
    # Cargar modelos
    for name in classifier.models.keys():
        classifier.models[name] = joblib.load(f"{models_dir}/{name}_model.joblib")
    
    return classifier

def predict_file(input_file, output_file, models_dir='../models'):
    """
    Realizar predicciones sobre un archivo de texto
    
    Args:
        input_file (str): Ruta al archivo de entrada (debe contener una oración por línea)
        output_file (str): Ruta donde guardar las predicciones
        models_dir (str): Directorio donde están guardados los modelos
    """
    logger.info(f"Procesando archivo: {input_file}")
    
    # Cargar textos
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    
    # Cargar modelos
    classifier = load_models(models_dir)
    
    # Realizar predicciones
    logger.info("Realizando predicciones...")
    predictions = classifier.predict(texts)
    
    # Guardar resultados
    logger.info(f"Guardando resultados en: {output_file}")
    results_df = pd.DataFrame({
        'text': texts,
        'prediction': predictions
    })
    results_df.to_csv(output_file, sep='\t', index=False)
    
    logger.info("¡Proceso completado!")
    return results_df

def main():
    # Ejemplo de uso
    input_file = '../data/REAL_DATA.txt'
    output_file = '../data/predictions.txt'
    
    predict_file(input_file, output_file)

if __name__ == "__main__":
    main() 