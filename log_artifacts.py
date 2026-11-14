
import mlflow
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Connect to existing MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Get existing experiment
experiment = mlflow.get_experiment_by_name(
    "Whisper_Champion_Challenger"
)

if experiment is None:
    raise ValueError("MLflow experiment not found!")

# Create a run for visualization artifacts
with mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name="comparison_visualization_artifacts"
):

    # Log comparison graphs
    mlflow.log_artifact(
        str(OUTPUT_DIR / "wer_comparison.png")
    )

    mlflow.log_artifact(
        str(OUTPUT_DIR / "inference_time_comparison.png")
    )

    # Log comparison and disagreement files
    mlflow.log_artifact(
        str(OUTPUT_DIR / "comparison.csv")
    )

    mlflow.log_artifact(
        str(OUTPUT_DIR / "disagreements.csv")
    )

print("All comparison artifacts logged to MLflow successfully!")
