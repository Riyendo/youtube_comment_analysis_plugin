import os
import pickle
import pandas as pd
import mlflow
import dagshub

from sklearn.metrics import classification_report, confusion_matrix


def main():

    # --------------------------------------------------
    # Project root directory
    # --------------------------------------------------

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../')
    )


    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    model_path = os.path.join(
        root_dir,
        'lgbm_model.pkl'
    )

    with open(model_path, 'rb') as file:
        model = pickle.load(file)


    # --------------------------------------------------
    # Load TF-IDF vectorizer
    # --------------------------------------------------

    vectorizer_path = os.path.join(
        root_dir,
        'tfidf_vectorizer.pkl'
    )

    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)


    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    test_data_path = os.path.join(
        root_dir,
        'data/interim/test_processed.csv'
    )

    test_data = pd.read_csv(test_data_path)

    test_data.fillna('', inplace=True)


    # --------------------------------------------------
    # Prepare test data
    # --------------------------------------------------

    X_test = vectorizer.transform(
        test_data['clean_comment']
    )

    y_test = test_data['category']


    # --------------------------------------------------
    # Make predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)


    # --------------------------------------------------
    # MLflow - Parameter Logging Only
    # --------------------------------------------------
    dagshub.init(
    repo_owner="gauravrajt167iwari",
    repo_name="yt_comment",
    mlflow=True
)

# Set experiment
    mlflow.set_experiment("my-experiment")
    mlflow.set_experiment("model-evaluation")

    with mlflow.start_run():

        # Basic evaluation parameters
        mlflow.log_param(
            "model_type",
            "LightGBM"
        )

        mlflow.log_param(
            "vectorizer",
            "TF-IDF"
        )

        mlflow.log_param(
            "test_dataset",
            "test_processed.csv"
        )

        mlflow.log_param(
            "input_column",
            "clean_comment"
        )

        mlflow.log_param(
            "target_column",
            "category"
        )

        mlflow.log_param(
            "test_samples",
            len(test_data)
        )

        mlflow.log_param(
            "number_of_features",
            X_test.shape[1]
        )

        # Model parameters
        if hasattr(model, "get_params"):

            model_params = model.get_params()

            for parameter, value in model_params.items():

                if value is not None:
                    mlflow.log_param(
                        parameter,
                        str(value)
                    )

        # TF-IDF parameters
        if hasattr(vectorizer, "get_params"):

            vectorizer_params = vectorizer.get_params()

            for parameter, value in vectorizer_params.items():

                if value is not None:
                    mlflow.log_param(
                        f"tfidf_{parameter}",
                        str(value)
                    )


if __name__ == '__main__':
    main()