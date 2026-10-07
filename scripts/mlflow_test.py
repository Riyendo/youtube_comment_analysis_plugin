import mlflow
import dagshub
import sys
import os

# Fix Windows UTF-8 console issue
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# Initialize DagsHub + MLflow
dagshub.init(
    repo_owner="gauravrajt167iwari",
    repo_name="yt_comment",
    mlflow=True
)

# Set experiment
mlflow.set_experiment("my-experiment")

# Test MLflow
with mlflow.start_run():
    mlflow.log_param("test_parameter", "hello")
    mlflow.log_metric("accuracy", 0.90)

print("MLflow run successful!")




