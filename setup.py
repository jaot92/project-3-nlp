from setuptools import setup, find_packages

setup(
    name="nlp-translation-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'pandas==2.0.3',
        'scikit-learn==1.3.0',
        'spacy==3.6.1',
        'nltk==3.8.1',
        'joblib==1.3.2',
        'matplotlib==3.7.2',
        'seaborn==0.12.2',
        'jupyter==1.0.0',
        'notebook==7.0.3'
    ],
    python_requires='>=3.8,<3.12',
    author="Tu Nombre",
    author_email="tu.email@ejemplo.com",
    description="Clasificador de traducciones automáticas vs humanas",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
)