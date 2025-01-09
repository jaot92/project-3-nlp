from setuptools import setup, find_packages

setup(
    name="nlp-translation-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'spacy>=3.0.0',
        'nltk>=3.6.0',
        'joblib>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'jupyter>=1.0.0',
    ],
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A classifier to distinguish between machine and human translations",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
) 