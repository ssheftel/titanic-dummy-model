from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from edscommon.job_api import Context, DefaultResource


MODULE_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = MODULE_ROOT_DIR / "data"


def featch_data_step(context: Context[DefaultResource]) -> pd.DataFrame:
    """
    Featches titanic data
    """
    passengers_file = DATA_DIR / "train.csv"

    passengers = pd.read_csv(passengers_file, index_col=False)

    context.log.debug(f"Passenger data ({passengers_file}) loaded")

    return passengers
