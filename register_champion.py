
import mlflow
from mlflow import MlflowClient
from transformers import pipeline


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

client = MlflowClient()

MODEL_NAME = "Whisper_Speech_Recognizer"

MODELS = {
    "champion": "openai/whisper-base",
    "challenger": "openai/whisper-tiny"
}


# Register both models
for alias, model_name in MODELS.items():

    print(f"\nRegistering {alias}: {model_name}")

    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=-1
    )

    with mlflow.start_run(
        run_name=f"register_{alias}_{model_name.split('/')[-1]}"
    ) as run:

        mlflow.log_param("huggingface_model", model_name)
        mlflow.log_param("task", "speech_recognition")
        mlflow.set_tag("model_role", alias)

        model_info = mlflow.transformers.log_model(
            transformers_model=recognizer,
            name="speech_recognition_model",
            task="automatic-speech-recognition",
            registered_model_name=MODEL_NAME
        )

        print("Model logged and registered.")

    # Find the registered version
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    matching_versions = [
        version for version in versions
        if version.run_id == run.info.run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            f"Could not find registered version for {model_name}"
        )

    version = matching_versions[0].version

    # Assign alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version
    )

    print(f"Registered version: {version}")
    print(f"Alias assigned: {alias}")


print("\n========== REGISTRATION COMPLETE ==========")
print("Registered model:", MODEL_NAME)
print("Champion: Whisper Base")
print("Challenger: Whisper Tiny")
