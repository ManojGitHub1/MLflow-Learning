import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='ManojGitHub1', repo_name='MLflow-Learning', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/ManojGitHub1/MLflow-Learning.mlflow")


# Load the dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# define params for RF model
max_depth = 12
n_estimators = 10


mlflow.set_experiment(experiment_name="testing-set_experiment2")
# Start MLflow run
with mlflow.start_run():
    # Train the model
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Log parameters and metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot
    plt.savefig("confusion_matrix.png")

    # log artifact using mlflow
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({
        "Author": "Manoj jivanagi",
        "Project": "Wine Quality Prediction",
    })

    # Log the model
    # mlflow.sklearn.log_model(model, "RandomForestClassifier")
    # Dagshub only supports mlflow version 2.0.0 and below
    mlflow.sklearn.log_model(model, artifact_path="RandomForestClassifier")

    


