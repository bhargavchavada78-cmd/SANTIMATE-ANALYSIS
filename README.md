# SENTIMENT ANALYSIS USING RANDOM FOREST AND TF-IDF

## Project Overview

This project performs Sentiment Analysis on text data using Machine Learning.

The model classifies text into sentiment categories such as:

* Positive
* Negative
* Neutral (if available in the dataset)

For text processing, TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical features.

Random Forest Classifier is used as the Machine Learning model for sentiment prediction.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Random Forest Classifier
* Matplotlib / Seaborn (optional for visualization)

## Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Text Preprocessing
4. Remove unnecessary or missing values
5. Convert text into numerical features using TF-IDF
6. Split the dataset into training and testing data
7. Train Random Forest Classifier
8. Predict sentiment
9. Evaluate the model
10. Test the model with new text

## TF-IDF

TF-IDF stands for:

Term Frequency - Inverse Document Frequency

Machine Learning algorithms cannot directly understand text.

Therefore, TF-IDF converts text into numerical vectors.

Example:

Text:

"I really like this product"

TF-IDF converts the important words from the sentence into numerical feature values that can be used by the Machine Learning model.

## Random Forest

Random Forest is a supervised Machine Learning algorithm.

It creates multiple Decision Trees and combines their predictions to produce the final prediction.

In this project, Random Forest is used to classify the sentiment of text.

## Example

Input:

"This product is amazing and I really like it."

Output:

Positive

Input:

"The service was very bad."

Output:

Negative

## Model Training Example

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df["text"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model = RandomForestClassifier(
n_estimators=100,
random_state=42
)

model.fit(X_train, y_train)

## Prediction Example

text = ["I am very happy with this product"]

text_vector = tfidf.transform(text)

prediction = model.predict(text_vector)

print(prediction)

## Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

## Project Structure

sentiment-analysis/
│
├── dataset/
│   └── sentiment_data.csv
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── sentiment_analysis.py
├── requirements.txt
└── README.txt

## Install Required Libraries

Run:

pip install pandas numpy scikit-learn matplotlib seaborn

## Run Project

Run the Python file:

python sentiment_analysis.py

## Future Improvements

The project can be improved by adding:

* Streamlit user interface
* Real-time sentiment prediction
* Logistic Regression comparison
* Naive Bayes comparison
* Deep Learning models
* LSTM
* BERT
* Better text preprocessing
* Model deployment

## Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be used to classify text sentiment.

TF-IDF is used to transform text into numerical features, while Random Forest is used to learn patterns from those features and predict the sentiment of new text.

## Author

Chavada Bhargav
