# Fake News Detection – NLP Classification Pipeline

This project builds a full Natural Language Processing (NLP) pipeline to classify news headlines as real or fake.
It includes data preprocessing, text normalization, custom tokenization, vectorization, multiple model baselines, hyperparameter tuning, and final evaluation.

## Project Structure
├── training_data.csv
├── testing_data.csv
├── best_model.pkl
├── NLP_project.ipynb
├── NLP.pdf
└── README.md

## 1. Setup & Dependencies

The project uses:
- Python 3.x
- pandas
- scikit-learn
- xgboost
- joblib
- regex / unicodedata

Install dependencies:
pip install pandas scikit-learn xgboost joblib

## 2. Data Preprocessing
Key preprocessing steps:
- Convert text to lowercase
- Unicode normalization (NFKC)
- Replace censored slurs with a placeholder (censored_slur)
- Normalize or mask numbers (num)
- Remove extra whitespace
- Custom normalization for quotes and punctuation
- Optionally remove:
    - political names
    - very short headlines
    - special punctuation problems

A custom tokenizer was created to handle:
- censored slur patterns
- numbers
- punctuation
- words with hyphens or apostrophes

## 3. Vectorization Methods

The project tests three main vectorizers:

### TF-IDF (word-level)
- custom tokenizer
- n-grams (1,2)
- min_df=2, max_df=0.9

### Bag of Words
- custom tokenizer
- n-grams (1,2)
- min_df=0.1, max_df=0.9

### TF-IDF (character-level)
- n-grams (3,6)
- excellent for stylistic features (fake-news traits)

## 4. Models Tested
The following classifiers were evaluated:
- LinearSVC
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- XGBoost
Each model was paired with multiple vectorizers to identify the strongest combination.

## 5. Evaluation
The project uses:
- Accuracy
- F1-score
- Confusion Matrix
- Classification Report
A helper function (evaluate_model()) runs the full training + prediction + scoring pipeline and stores results.

## 6. Hyperparameter Tuning
GridSearchCV was applied to:
- TF-IDF (char-level) + LinearSVC

The best-performing model was:
- LinearSVC + char-level TF-IDF
    After tuning:
        - F1-score ≈ 0.968
        - Accuracy ≈ 0.968
This became the final selected model.

## 7. Error & Confusion Analysis
Misclassifications showed:
- frequent mentions of political names
- very short headlines
- punctuation inconsistencies

Several targeted preprocessing strategies were tested:
- removing political entities
- filtering out headlines <= 3 words
- normalizing quotes more aggressively
These steps yielded small but measurable F1 improvements.

## 8. Final Results
A summary of the main models:

Rank  ||	Model                           ||	Accuracy  ||     F1
1	        <=3 Headliners Best Model	        0.968	        0.967
2	        Best Model              	        0.968	        0.967
3	        Remove Pol Names + Best Model      	0.965	        0.964
…	        (Others)	                        …	            …
20+	        BoW models	                        ~0.78	        ~0.80
Last	    MNB + BoW	                        0.69	        0.64

Character TF-IDF + LinearSVC consistently outperformed all other combinations.

## 9. Exporting the Final Model
The best model is saved using:
joblib.dump(best_model, "best_model.pkl")

It can later be loaded via:
import joblib
model = joblib.load("best_model.pkl")

## 10. Testing on Custom Headlines

The final model was evaluated on:
- existing fake & real news (0.8 accuracy)
- an external test dataset
Performance remained very high and aligned with cross-validation.

The existing fake & real news dataset was also tested on a pre-trained transfer model (Fake News Bert Detect) which could classify everything correctly. 

## 11. Summary
What worked best:
- Character-level TF-IDF
- Linear Support Vector Classifier
- Custom normalization & tokenization
- Hyperparameter tuning for C and n-gram ranges

Key insights:
- Simple linear models + good preprocessing outperform complex models on sparse text data.
- Short headlines are inherently ambiguous and hurt model learning.
