
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Detect project folders
SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "outputs":
    OUTPUT_DIR = SCRIPT_DIR
else:
    OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Read comparison results
data = pd.read_csv(OUTPUT_DIR / "comparison.csv")

# Prepare model names
models = data["model"].str.replace(
    "openai/whisper-", "Whisper ", regex=False
).str.title()


# Graph 1: WER Comparison
plt.figure(figsize=(8, 5))

plt.bar(models, data["wer"])

plt.title("Word Error Rate Comparison")
plt.xlabel("Models")
plt.ylabel("Word Error Rate")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "wer_comparison.png")
plt.close()


# Graph 2: Inference Time Comparison
plt.figure(figsize=(8, 5))

plt.bar(models, data["average_inference_seconds"])

plt.title("Average Inference Time Comparison")
plt.xlabel("Models")
plt.ylabel("Time (seconds)")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "inference_time_comparison.png")
plt.close()


print("Comparison graphs created successfully!")
print("Saved: wer_comparison.png")
print("Saved: inference_time_comparison.png")
