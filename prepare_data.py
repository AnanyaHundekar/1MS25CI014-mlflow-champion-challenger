
from datasets import load_dataset, Audio
from pathlib import Path
import csv

# Project folders
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

EVAL_DIR = DATA_DIR / "evaluation_audio"
NEW_DIR = DATA_DIR / "new_audio"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
NEW_DIR.mkdir(parents=True, exist_ok=True)

# Download the English LibriSpeech test dataset
print("Loading LibriSpeech dataset...")

dataset = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="test",
    streaming=True
)

# Read audio without decoding it
dataset = dataset.cast_column("audio", Audio(decode=False))

evaluation_records = []
new_records = []

# Collect 15 recordings
for index, sample in enumerate(dataset):
    if index >= 15:
        break

    if index < 10:
        folder = EVAL_DIR
        records = evaluation_records
        sample_name = f"eval_{index + 1:02d}.flac"
    else:
        folder = NEW_DIR
        records = new_records
        sample_name = f"new_{index - 9:02d}.flac"

    audio_path = folder / sample_name

    # Save the original FLAC audio
    audio_path.write_bytes(sample["audio"]["bytes"])

    records.append({
        "sample_id": sample["id"],
        "audio_path": str(audio_path.relative_to(BASE_DIR)),
        "reference_text": sample["text"]
    })

    print(f"Saved {sample_name}")

# Save evaluation CSV
with open(DATA_DIR / "evaluation.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=evaluation_records[0].keys())
    writer.writeheader()
    writer.writerows(evaluation_records)

# Save new-data CSV
with open(DATA_DIR / "new_data.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=new_records[0].keys())
    writer.writeheader()
    writer.writerows(new_records)

print("\nDataset preparation completed!")
print("Evaluation recordings:", len(evaluation_records))
print("New recordings:", len(new_records))
