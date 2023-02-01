"""
survivorship random forest Job module
"""
from pathlib import Path

import pandas as pd
from edscommon.job_api.context import Context
from edscommon.job_api.jobs import create_job
from edscommon.job_api.resources.base import DefaultResource

from titanic_dummy_model.jobs.survivorship_random_forest.steps.featch_data import featch_data_step
from titanic_dummy_model.jobs.survivorship_random_forest.steps.featureize import (
    featureize_step,
    make_model_input_step,
)
from titanic_dummy_model.jobs.survivorship_random_forest.steps.train_model import (
    train_model_step,
    persist_model_step,
)

job_config_root = Path(__file__).parent.absolute()

job = create_job(
    "survivorship-random-forest", project_root_dir=job_config_root, resource_type=DefaultResource
)


@job.def_runner
def run(context: Context[DefaultResource]) -> None:
    """
    survivorship random forest job steps and logic.
    """
    context.log.debug("Starting Job: survivorship-random-forest")

    # Run Steps
    passengers = featch_data_step(context)
    passenger_features = featureize_step(context, passengers)
    X, y = make_model_input_step(passenger_features)
    model = train_model_step(X, y)
    persist_model_step(model, "random_forest_classifier.pkl")

    context.log.debug("Finished Job: survivorship-random-forest")


if __name__ == "__main__":
    job()
