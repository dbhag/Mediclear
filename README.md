# MediClear

MediClear is a Python project for two medical NLP tasks:

1.  Medical text simplification using a T5-based sequence-to-sequence model.
2.  Health claim credibility classification using a DistilBERT-based classifier.

It also includes a Streamlit web app for interactive inference.

## What Each Script Does

 src/mediclear/cli/prepare_data.py downloads and cleans the simplification and health-fact datasets.
 src/mediclear/cli/train_simplifier.py trains the T5 simplification model.
 src/mediclear/cli/train_credibility_classifier.py trains the health claim classifier.
 src/mediclear/cli/evaluate_neural_models.py evaluates both models and saves metrics/results CSV files.
 app/streamlit_app.py launches the MediClear web UI.
 src/mediclear/neural_pipeline.py loads trained models and runs inference.


## Requirements

 Python 3.10 or 3.11 recommended


## Build and Run on Windows 

Open PowerShell in the project folder.

### 1. Create and activate a virtual environment

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1


If PowerShell blocks activation, run this once in the current terminal:


Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


### 2. Upgrade pip

python -m pip install --upgrade pip


### 3. Install the project


pip install -e .

pip install easse


### 4. Prepare the datasets


python -m mediclear.cli.prepare_data

You can also prepare only one dataset:

python -m mediclear.cli.prepare_data --only simplification
python -m mediclear.cli.prepare_data --only health_fact


### 5. Train the models

Train the simplifier:

python -m train_simplifier

Train the credibility classifier:

python -m mediclear.cli.train_credibility_classifier


### 6. Evaluate the models


python -m mediclear.cli.evaluate_neural_models


### 7. Run the Streamlit app


streamlit run app/streamlit_app.py



## Build and Run on macOS 

Open Terminalin the project folder.

### 1. Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate


### 2. Upgrade pip


python -m pip install --upgrade pip


### 3. Install the project


pip install -e .


pip install easse


### 4. Prepare the datasets


python -m mediclear.cli.prepare_data


Or run a single preparation step:

python -m mediclear.cli.prepare_data --only simplification
python -m mediclear.cli.prepare_data --only health_fact


### 5. Train the models

python -m train_simplifier
python -m mediclear.cli.train_credibility_classifier


### 6. Evaluate the models

python -m mediclear.cli.evaluate_neural_models


### 7. Run the Streamlit app


streamlit run app/streamlit_app.py


## Expected Generated Files

After running the workflow, you should see files like these:

### Data

  data/cochrane_simplification_train.csv
  data/cochrane_simplification_validation.csv
  data/cochrane_simplification_test.csv
  data/health_fact_train.csv
  data/health_fact_validation.csv
  data/health_fact_test.csv

### Models

 models/t5_simplifier/
 models/pubhealth_classifier/

### Results

 results/neural_simplification_outputs.csv
 results/neural_simplification_metrics.csv
 results/neural_health_fact_outputs.csv
 results/neural_health_fact_metrics.csv


## Common Commands

Run the interactive CLI pipeline:

python -m mediclear

Use custom file paths while training:

python -m mediclear.cli.train_simplifier --train_file data/cochrane_simplification_train.csv --val_file data/cochrane_simplification_validation.csv --output_dir models/t5_simplifier

python -m mediclear.cli.train_credibility_classifier --train_file data/health_fact_train.csv --val_file data/health_fact_validation.csv --output_dir models/pubhealth_classifier


Use custom file paths while evaluating:


python -m mediclear.cli.evaluate_neural_models --simplifier_dir models/t5_simplifier --classifier_dir models/pubhealth_classifier --simplification_csv data/cochrane_simplification_test.csv --health_fact_csv data/health_fact_test.csv


## Troubleshooting

python -m pip install -e .
python -m pytest tests/test_package_layout.py
### Missing model folder

Make sure you trained both models first, or place trained model files into:

	models/t5_simplifier/
	models/pubhealth_classifier/

### Streamlit command not found


python -m streamlit run app/streamlit_app.py


### SARI shows as blank or `None`


pip install easse

