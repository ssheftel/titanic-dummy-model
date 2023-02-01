from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"


def train_model_step(passenger_features: pd.DataFrame, target: pd.Series) -> RandomForestClassifier:
    rand_forest = RandomForestClassifier(max_depth=2)
    rand_forest.fit(passenger_features, y=target.astype("bool"))
    return rand_forest


def persist_model_step(model: RandomForestClassifier, model_name: str, model_dir: Path = DATA_DIR) -> None:
    model_save_path = model_dir / model_name

    with model_save_path.open("wb") as f:
        pickle.dump(model, f)
