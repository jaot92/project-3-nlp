import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import spacy
import nltk
from nltk.corpus import stopwords
import re
import time
import logging
import joblib

# Descargar recursos necesarios
nltk.download('stopwords')
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelo de spaCy
try:
    nlp = spacy.load('es_core_news_md')
except OSError:
    logger.info("Descargando modelo de spaCy...")
    spacy.cli.download('es_core_news_md')
    nlp = spacy.load('es_core_news_md')

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extractor de características lingüísticas del texto"""
    
    def __init__(self):
        self.spanish_stopwords = set(stopwords.words('spanish'))
    
    def get_linguistic_features(self, text):
        """Extraer características lingüísticas usando spaCy"""
        doc = nlp(text)
        
        return {
            'n_tokens': len(doc),
            'n_sust': len([token for token in doc if token.pos_ == 'NOUN']),
            'n_verb': len([token for token in doc if token.pos_ == 'VERB']),
            'n_adj': len([token for token in doc if token.pos_ == 'ADJ']),
            'n_entities': len(doc.ents),
            'avg_word_length': np.mean([len(token.text) for token in doc]),
            'n_puntuacion': len([token for token in doc if token.is_punct]),
            'n_mayusculas': sum(1 for c in text if c.isupper()),
            'n_numeros': len([token for token in doc if token.like_num])
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            features.append(self.get_linguistic_features(text))
        return pd.DataFrame(features)

class TranslationClassifier:
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'svm': LinearSVC(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Crear pipeline con TF-IDF y características lingüísticas
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )),
                ('linguistic', TextFeatureExtractor())
            ]))
        ])
        
        self.results = {}

    def load_data(self, file_path):
        """Cargar y preparar los datos"""
        logger.info(f"Cargando datos desde {file_path}")
        df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
        return df

    def preprocess_text(self, text):
        """Preprocesar el texto"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar números
        text = re.sub(r'\d+', '', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenización y eliminación de stopwords
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop]
        
        # Lematización
        lemmatized = [token.lemma_ for token in nlp(" ".join(tokens))]
        
        return " ".join(lemmatized)

    def train_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Entrenar y evaluar todos los modelos"""
        logger.info("Transformando características...")
        X_train_features = self.pipeline.fit_transform(X_train)
        X_test_features = self.pipeline.transform(X_test)

        for name, model in self.models.items():
            logger.info(f"Entrenando modelo: {name}")
            start_time = time.time()
            
            model.fit(X_train_features, y_train)
            y_pred = model.predict(X_test_features)
            
            train_time = time.time() - start_time
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            self.results[name] = {
                'accuracy': accuracy,
                'training_time': train_time,
                'classification_report': report
            }

            logger.info(f"Resultados para {name}:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Tiempo de entrenamiento: {train_time:.2f} segundos")
            logger.info("\nReporte de clasificación:")
            logger.info(report)

    def predict(self, texts):
        """Predecir la clase de nuevos textos usando el mejor modelo"""
        if not self.results:
            raise ValueError("Los modelos deben ser entrenados antes de hacer predicciones")
        
        # Encontrar el mejor modelo basado en accuracy
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = self.models[best_model_name]
        
        # Preprocesar textos
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transformar características
        X_features = self.pipeline.transform(processed_texts)
        
        # Realizar predicción
        predictions = best_model.predict(X_features)
        
        return predictions

    def save_models(self, output_dir):
        """Guardar los modelos entrenados y el pipeline"""
        for name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/{name}_model.joblib")
        joblib.dump(self.pipeline, f"{output_dir}/pipeline.joblib")
        
        # Guardar resultados
        with open(f"{output_dir}/results.txt", 'w') as f:
            for model_name, result in self.results.items():
                f.write(f"\nResultados para {model_name}:\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"Tiempo de entrenamiento: {result['training_time']:.2f} segundos\n")
                f.write("\nReporte de clasificación:\n")
                f.write(result['classification_report'])

def main():
    classifier = TranslationClassifier()
    
    # Cargar datos
    logger.info("Cargando datos de entrenamiento...")
    df = classifier.load_data('../data/TRAINING_DATA.txt')
    
    # Preprocesar textos
    logger.info("Preprocesando textos...")
    df['text'] = df['text'].apply(classifier.preprocess_text)
    
    # Split de datos
    logger.info("Dividiendo datos en train y test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Entrenar y evaluar modelos
    logger.info("Iniciando entrenamiento y evaluación de modelos...")
    classifier.train_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Guardar modelos
    logger.info("Guardando modelos y resultados...")
    classifier.save_models('../models')
    
    logger.info("¡Proceso completado exitosamente!")

if __name__ == "__main__":
    main() 