# Mental Health Score Prediction

A student machine learning project that estimates a **DASS-42 stress score (0–42)** from ten personality responses, age and education.

This is an educational prototype, not a diagnostic or clinically validated screening tool. It does not predict future mental health.

## Features

- Home, Questions, Result, About and Model performance pages
- Plain-English question explanations
- Answers retained in memory during navigation in the same session
- Saved Random Forest pipeline with input validation
- Model comparison, actual-versus-predicted, error distribution and score distribution charts
- Dark theme, line illustrations and credited photography

## Dataset

[OpenPsychometrics DASS dataset](https://openpsychometrics.org/_rawdata/): 39,775 voluntary online survey responses collected in 2017–2019.

Download DASS_data_21.02.19.zip from the source. Extract data.csv and codebook.txt into a local folder named data. Raw data is excluded from this repository; consult the source for reuse terms.

The stress target uses items 1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35 and 39. Each dataset response is converted from 1–4 to 0–3 before summing.

Inputs are TIPI1–TIPI10, age and education. The DASS answers that determine the target are excluded from model inputs.

## Preprocessing

Invalid personality and education responses become missing values. Ages outside 13–100 become missing, a project cleaning rule rather than a clinical cutoff.

Within each training pipeline, numeric inputs use median imputation and standardization. Education uses most-frequent imputation and one-hot encoding. Preprocessing is fitted only on the training data.

## Results

Random seed: 42. Initial split: 60% training, 20% validation, 20% testing.

Validation average absolute errors (lower is better):

- Mean baseline: 8.730 points
- Ridge regression: 6.317 points
- Random Forest: 6.084 points

Random Forest was selected using validation results and retrained on the combined 31,820 development rows.

Final results on 7,955 held-out test rows:

- MAE: 6.059 points
- RMSE: 7.608 points
- R²: 0.484

R² is not an accuracy percentage. Average error is not an individual confidence interval.

The charts page reconstructs the original test split and checks the metrics against saved metadata before showing test charts. Do not use these test charts to tune the model.

## Run locally on Windows

Developed using Python 3.14. Run these commands in PowerShell:

    git clone https://github.com/teju7-ai/mental-health-score-prediction.git
    cd mental-health-score-prediction
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Download and extract the dataset into data as described above, then run:

    .\.venv\Scripts\python.exe explore_data.py
    .\.venv\Scripts\python.exe prepare_data.py
    .\.venv\Scripts\python.exe train_model.py
    .\.venv\Scripts\python.exe -m streamlit run app.py

Open the local address printed by Streamlit. Keep the terminal running.

The compressed trained model is included for deployment. Training scripts remain available to reproduce it. Load only trusted joblib files.

## Files

- app.py: interface, predictions and charts
- explore_data.py: initial data inspection
- prepare_data.py: cleaning and target calculation
- train_model.py: model comparison, evaluation and saving
- requirements.txt: installed package versions
- .streamlit/config.toml: theme configuration
- models/model_metadata.json: saved model information and metrics
- data/: local datasets, excluded from Git
- models/stress_model.joblib: compressed pipeline included for deployment

## Limitations and privacy

The self-selected survey sample may not represent everyone. Performance has not been established specifically for Indian students. Checking exact duplicates does not establish that every response comes from a different participant.

Plain-English explanations have not been validated as a new questionnaire. Test results come from the original survey, not a clinical evaluation of the app.

The app does not write user answers to a file or database. Answers are held in session memory. The remote photograph needs internet access.

## Credits

- [OpenPsychometrics dataset](https://openpsychometrics.org/_rawdata/)
- [DASS information](https://dass.psy.unsw.edu.au/)
- Personality items: Gosling, Rentfrow & Swann (2003), Ten-Item Personality Inventory
- [Photograph by Simon Reza](https://unsplash.com/photos/lduDz1Ra0sY), under the [Unsplash License](https://unsplash.com/license)
- Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit and joblib

## Hosted app

The deployment includes the compressed trained model and exported evaluation charts.
The raw dataset is not uploaded. Charts were checked against the saved test metrics before export.
To run the included app after installing requirements, run Streamlit directly; downloading data and training are only needed to reproduce the model.
Use Python 3.14 in Streamlit Community Cloud advanced settings, branch main, entrypoint app.py.
