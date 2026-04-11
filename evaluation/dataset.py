import json
from config import settings


def load_ground_truth():
    with open(settings.EVAL_DATASET_PATH, "r") as f:
        return json.load(f)


def save_ground_truth(data):
    with open(settings.EVAL_DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)