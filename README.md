
# Titanic Dummy Model

---

Model for evaluation of mlops tools


## Contents

This repo has the following directories:
* `ci`: Jenkins build and automation scripts.
* `notebooks`: Jupyter notebooks used for exploratory data analysis, model development, and ROI calculations.
* `titanic_dummy_model`: Root code and configurations.
* `tests`: Unit tests project code.


## Setup

The first time you use this repo:
1. Clone the repo and navigate to the root folder.
2. Create a conda environment for this project, and activate it:
```bash
conda create --name titanic-dummy-model-py38 python=3.8
conda activate titanic-dummy-model-py38
```
3. Install required python packages from the root of this repo:
```bash
pip install -r requirements-dev.txt
```
