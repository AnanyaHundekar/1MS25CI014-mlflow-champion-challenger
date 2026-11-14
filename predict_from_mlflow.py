
import mlflow
import pandas as pd
import soundfile as sf
import time
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

# Load the registered champion model
MODEL_URI = "models:/Whisper_Speech_Recognizer@champion"

print("Loading champion model from MLflow...")

recognizer = mlflow.transformers.load_model(MODEL_URI)

print("Champion model loaded successfully!")

# Read new audio sample details
new_data = pd.read_csv(DATA_DIR / "new_data.csv")

results = []

print("\nGenerating predictions on new audio samples...\n")

for _, row in new_data.iterrows():

    audio_path = BASE_DIR / row["audio_path"]

    audio, sampling_rate = sf.read(str(audio_path))

    start_time = time.time()

    prediction = recognizer({
        "raw": audio,
        "sampling_rate": sampling_rate
    })

    inference_time = time.time() - start_time

    results.append({
        "sample_id": row["sample_id"],
        "prediction": prediction["text"],
        "inference_seconds": inference_time
    })

    print("Sample:", row["sample_id"])
    print("Prediction:", prediction["text"])
    print("Time:", round(inference_time, 3), "seconds")
    print("-" * 50)

# Save predictions
results_df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "champion_new_predictions.csv"

results_df.to_csv(output_path, index=False)

print("\n========== PREDICTION COMPLETE ==========")
print("Model:", MODEL_URI)
print("Total samples:", len(results_df))
print("Saved to:", output_path)
