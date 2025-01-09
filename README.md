![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Natural Language Processing Challenge

## Introduction

Learning how to process text is a skill required for Data Scientists. In this project, you will put these skills into practice to identify whether a sentence was automatically translated or translated by a human.

## Project Overview

In this repository you will find dataset containing sentences in Spanish and their tags: 0, if the sentences was translated by a Machine, 1, if the sentence was translated by a professional translator. Your goal is to build a classifier that is able to distinguish between the two.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── data
│   │   ├── TRAINING_DATA.txt
│   │   └── REAL_DATA.txt
│   ├── models
│   │   └── train_models.py
│   └── notebooks
│       └── 01_exploratory_analysis.ipynb
└── models/
```

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Exploratory Data Analysis**:
   - Open and run the Jupyter notebook: `src/notebooks/01_exploratory_analysis.ipynb`

2. **Train Models**:
```bash
cd src/models
python train_models.py
```

3. **Model Performance**:
   The script will train and evaluate multiple models:
   - Naive Bayes
   - Support Vector Machine
   - Random Forest
   
   Results will be logged and models will be saved in the `models/` directory.

## Deliverables

1. **Python Code:** Well-documented Python code that conducts the analysis.
2. **Accuracy estimation:** Estimation of model performance.
3. **Classified Dataset**: Ability to classify new datasets.

## Model Selection

We implement and compare multiple models:
- Classical ML approaches (Naive Bayes, SVM, Random Forest)
- Feature engineering techniques (TF-IDF, Word Embeddings)
- Performance metrics (Accuracy, F1-score, Training time)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
