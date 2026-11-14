
import re
import time
from pathlib import Path

import mlflow
import pandas as pd
import jiwer
import soundfile as sf
from transformers import pipeline


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "evaluation.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# Models to compare
MODELS = {
    "A": "openai/whisper-tiny",
    "B": "openai/whisper-base"
}


# Normalize text before calculating WER
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Load evaluation data
data = pd.read_csv(DATA_FILE)

required_columns = ["sample_id", "audio_path", "reference_text"]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Missing column: {column}")


# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Whisper_Champion_Challenger")


def evaluate_model(role, model_name):

    print(f"\nEvaluating Model {role}: {model_name}")

    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=-1
    )

    predictions = []
    total_time = 0

    for index, row in data.iterrows():

        audio_path = BASE_DIR / row["audio_path"]

        audio, sampling_rate = sf.read(str(audio_path))

        start_time = time.time()

        result = recognizer({
            "raw": audio,
            "sampling_rate": sampling_rate
        })

        inference_time = time.time() - start_time
        total_time += inference_time

        predictions.append({
            "sample_id": row["sample_id"],
            "audio_path": row["audio_path"],
            "reference": row["reference_text"],
            "prediction": result["text"],
            "inference_time_seconds": inference_time
        })

        print(f"Processed {index + 1}/{len(data)}")

    # Normalize references and predictions
    references = [
        normalize_text(item["reference"])
        for item in predictions
    ]

    hypotheses = [
        normalize_text(item["prediction"])
        for item in predictions
    ]

    # Calculate WER
    wer_score = jiwer.wer(references, hypotheses)

    average_time = total_time / len(data)

    # Save predictions
    predictions_file = OUTPUT_DIR / f"{role}_predictions.csv"

    pd.DataFrame(predictions).to_csv(
        predictions_file,
        index=False
    )

    # Log model evaluation to MLflow
    with mlflow.start_run(
        run_name=f"{role}_speech_whisper_normalized"
    ):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "speech_recognition")
        mlflow.log_param("dataset", "LibriSpeech test-clean")
        mlflow.log_param("number_of_samples", len(data))
        mlflow.log_param(
            "text_normalization",
            "lowercase_and_remove_punctuation"
        )

        mlflow.set_tag("model_role", role)
        mlflow.set_tag("evaluation_type", "normalized_evaluation")

        mlflow.log_metric("wer", wer_score)
        mlflow.log_metric(
            "average_inference_seconds",
            average_time
        )

        mlflow.log_artifact(str(predictions_file))

    print(f"\nModel {role} WER: {wer_score:.6f}")
    print(f"Model {role} Average Time: {average_time:.4f} seconds")

    return {
        "role": role,
        "model_name": model_name,
        "wer": wer_score,
        "average_inference_seconds": average_time,
        "predictions": predictions
    }


# Evaluate both models
result_a = evaluate_model("A", MODELS["A"])
result_b = evaluate_model("B", MODELS["B"])


# Create comparison table
comparison = pd.DataFrame([
    {
        "model": result_a["model_name"],
        "wer": result_a["wer"],
        "average_inference_seconds": result_a["average_inference_seconds"]
    },
    {
        "model": result_b["model_name"],
        "wer": result_b["wer"],
        "average_inference_seconds": result_b["average_inference_seconds"]
    }
])

comparison_file = OUTPUT_DIR / "comparison.csv"
comparison.to_csv(comparison_file, index=False)


# Select champion using predefined rule
champion = min(
    [result_a, result_b],
    key=lambda result: (
        result["wer"],
        result["average_inference_seconds"]
    )
)

challenger = (
    result_b if champion["role"] == "A" else result_a
)


# Analyze disagreements
comments = {
    "6930-75918-0000": (
        "Model A is more accurate. It predicts 'Concord', "
        "which matches the reference, while Model B predicts 'Concorde'."
    ),

    "6930-75918-0001": (
        "Model B is more accurate. It correctly recognizes "
        "'plentiful provision', while Model A predicts "
        "'plantable provision'."
    ),

    "6930-75918-0002": (
        "Model B is closer to the reference. It recognizes "
        "'were poured in' more accurately than Model A, "
        "although both models make errors."
    ),

    "6930-75918-0003": (
        "Model B is more accurate. It correctly recognizes "
        "'an object of worship' and 'had suffered', while "
        "Model A makes errors in these phrases."
    )
}


disagreements = []

for item_a, item_b in zip(
    result_a["predictions"],
    result_b["predictions"]
):

    if normalize_text(item_a["prediction"]) != normalize_text(item_b["prediction"]):

        sample_id = item_a["sample_id"]

        disagreements.append({
            "sample_id": sample_id,
            "reference": item_a["reference"],
            "model_a_prediction": item_a["prediction"],
            "model_b_prediction": item_b["prediction"],
            "comment": comments.get(
                sample_id,
                "The models produced different transcriptions. "
                "Manual review is required."
            )
        })


# Save disagreements
disagreements_file = OUTPUT_DIR / "disagreements.csv"

pd.DataFrame(disagreements).to_csv(
    disagreements_file,
    index=False
)


# Log comparison run
with mlflow.start_run(
    run_name="compare_speech_whisper_normalized"
):

    mlflow.log_param("model_a", result_a["model_name"])
    mlflow.log_param("model_b", result_b["model_name"])

    mlflow.log_param(
        "selection_rule",
        "Lowest WER; tie broken by faster inference"
    )

    mlflow.log_param(
        "number_of_disagreements",
        len(disagreements)
    )

    mlflow.log_metric("model_a_wer", result_a["wer"])
    mlflow.log_metric("model_b_wer", result_b["wer"])

    mlflow.log_metric(
        "model_a_average_time",
        result_a["average_inference_seconds"]
    )

    mlflow.log_metric(
        "model_b_average_time",
        result_b["average_inference_seconds"]
    )

    mlflow.set_tag("champion", champion["model_name"])
    mlflow.set_tag("challenger", challenger["model_name"])

    mlflow.log_artifact(str(comparison_file))
    mlflow.log_artifact(str(disagreements_file))


# Final results
print("\n========== FINAL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nChampion:", champion["model_name"])
print("Challenger:", challenger["model_name"])
print("Total disagreements:", len(disagreements))

print("\nEvaluation completed successfully!")
