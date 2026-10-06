import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def main():

    # Project root directory
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../')
    )

    # Load trained model
    with open(os.path.join(root_dir, 'lgbm_model.pkl'), 'rb') as file:
        model = pickle.load(file)

    # Load TF-IDF vectorizer
    with open(os.path.join(root_dir, 'tfidf_vectorizer.pkl'), 'rb') as file:
        vectorizer = pickle.load(file)

    # Load test data
    test_data = pd.read_csv(
        os.path.join(root_dir, 'data/interim/test_processed.csv')
    )

    # Handle missing values
    test_data.fillna('', inplace=True)

    # Prepare test data
    X_test = vectorizer.transform(
        test_data['clean_comment']
    )

    y_test = test_data['category']

    # Make predictions
    y_pred = model.predict(X_test)

    # Classification report
    report = classification_report(
        y_test,
        y_pred
    )

    print("Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    main()