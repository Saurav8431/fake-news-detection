# Fake News Detection

A local machine-learning demo that trains separate TF-IDF + Logistic Regression classifiers for English and Hindi news, with a Streamlit interface for trying them on article text.

> **Important:** This is an educational text classifier, not a fact-checking service. It learns patterns in a historical dataset and can be wrong or biased. Do not use its predictions as evidence that a claim is true or false.

## Requirements

- Python 3.10 or newer
- Internet access for the one-time dataset download
- No AI API key is needed

## Run the already-trained project

Open PowerShell in this project folder and run:

```powershell
python -m streamlit run app.py
```

Or double-click `run_app.bat` on Windows. Streamlit opens the simple interface in your browser. Select English or Hindi, open the dataset link to find an article, then enter its topic and a short brief and click **Check news**.

The model prediction appears under the form. Accuracy, precision, recall, F1, ROC AUC, and the confusion matrix stay hidden under **Show model evaluation results** until you open that section.

## Deploy from GitHub

The trained model files are included so the hosted app can start without downloading or training the datasets. The raw datasets are not included in the repository.

1. Push this project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select the repository and branch, set **Main file path** to `app.py`, then deploy.

The app links to the source datasets so visitors can browse news. Model predictions are not fact-checks.

## First-time setup

From this directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Training downloads the datasets to `data\raw\`, creates stratified holdout splits, prints evaluation metrics, and saves the fitted pipelines and metrics under `models\`. The app uses the model for the selected language; run training again after deleting the models or changing the datasets.

## Dataset

The project uses the [WELFake dataset on Hugging Face](https://huggingface.co/datasets/davanstrien/WELFake), which reports 72,134 labeled English-language articles. Labels are `0 = fake` and `1 = real`. The dataset card describes it as a merge of several datasets, but does not state a license. The project downloads it for local use and does not redistribute it; check the source and underlying dataset terms before other uses.

For Hindi, the project uses the [Hindi Fake News Detection Dataset](https://github.com/BSAkash/Hindi-Fake-News-Detection-Dataset). It provides separate fake and authentic Excel files with titles and summaries. The repository reports about 13,000 rows and includes an MIT license. The dataset paper asks users to cite: Akash Bonagiri et al., “A poster on learnings from an attempt to build an NLP-based fake news classification system for Hindi,” ACM COMPASS 2021, [DOI: 10.1145/3460112.3471974](https://doi.org/10.1145/3460112.3471974).

The Hindi data's fake examples are largely fact-check articles about false claims, while its authentic examples come from news publishers. The classifier can therefore learn source and writing-style differences rather than truth. Its Hindi predictions are only an educational experiment, not fact verification.

## Model and evaluation

Article title and text are combined and classified with Logistic Regression. English uses word-level TF-IDF (unigrams and bigrams); Hindi uses character n-gram TF-IDF to handle Devanagari text. The holdout reports include accuracy, precision, recall, F1, ROC AUC, and confusion matrices. These metrics measure performance on random splits of these datasets—not performance on current news or news outside their sources.

## Files created by training

- `data\raw\welfake.parquet` — downloaded source data
- `data\raw\hindi_fake.xlsx`, `data\raw\hindi_authentic.xlsx` — Hindi source data
- `models\fake_news_pipeline.joblib`, `models\metrics.json` — English model and metrics, included for deployment
- `models\hindi_fake_news_pipeline.joblib`, `models\hindi_metrics.json` — Hindi model and metrics, included for deployment

Raw datasets and other generated files are excluded from version control.
