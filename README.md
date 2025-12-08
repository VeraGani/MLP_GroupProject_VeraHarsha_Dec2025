# Sentiment Analysis in the Era of Large Language Models
## Team Members: Vera Ganicheva and Harsha Vardhani Mundeshwar

This repository contains all project materials, code, datasets, and documentation related to the research project **"Sentiment Analysis in the Era of Large Language Models."** The project explores how modern LLMs transform traditional sentiment analysis approaches by comparing classical machine learning techniques with LLM-based zero-shot methods.

---

## Repository Structure

```text
├── notebooks/
│   └── phase1_env_setup.ipynb               # Initial Colab environment setup
│
├── outputs/
│   ├── contrib_logreg/                      # Logistic Regression predictions (contribution)
│   │   └── imdb/
│   │        ├── .gitkeep
│   │        └── prediction.csv              # Logistic Regression prediction file
│   │
│   ├── contrib_mlp/                         # MLP Classifier predictions (contribution)
│   │   └── imdb/
│   │        ├── .gitkeep
│   │        └── prediction.csv              # MLP prediction file
│
├── LLM_Sentiment_Contribution.ipynb         # Contribution: TF-IDF + LogReg + MLP (IMDB)
├── LLM_sentiment_zero_shot.ipynb            # Zero-shot evaluation pipeline reproduction
├── README.md                                # Project documentation



phase1_env_setup.ipynb

Notebook: notebooks/phase1_env_setup.ipynb
This notebook prepares the environment required to run the sentiment analysis experiments based on the LLM-Sentiment repository. It is intended to be executed in Google Colab.

Purpose:
- Mounts Google Drive to access datasets or save outputs.
- Clones the official LLM-Sentiment GitHub repository.
- Navigates to the repository directory and lists its contents.
- Installs all required Python libraries.
- Fixes compatibility issues with older dependency versions (especially numpy and scikit-learn) by:
  - Creating a modified requirements_colab.txt
  - Installing updated versions of packages suitable for Python 3.12 (Colab).
- Installs additional NLP frameworks including transformers and torch.
- Verifies that all required libraries import successfully.

How to Run This Notebook:

Step 1 — Open in Google Colab
- Click the Colab badge at the top of the notebook or upload the notebook manually into Colab.

Step 2 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

Step 3 — Clone the Repository
!git clone https://github.com/DAMO-NLP-SG/LLM-Sentiment.git
If the folder already exists, it will simply navigate into it.

Step 4 — Install Dependencies
- Creates a modified requirements_colab.txt
- Replaces incompatible dependency pins like:
  - numpy==1.21.2
  - scikit-learn==1.1.1
- Upgrades to modern versions supported by Colab.

Final installation command:
!pip install -q \
  "numpy>=1.24" \
  "pandas>=2.0" \
  "requests>=2.28" \
  "scikit-learn>=1.4" \
  "openai>=0.27.2" \
  "tenacity>=8.2.2" \
  "tqdm>=4.65" \
  "transformers>=4.36" \
  "torch>=2.0"

Step 5 — Validate Environment
print("All good!")
Confirming that all libraries imported successfully.

Output:
- This notebook does not produce model outputs.
- It simply sets up the environment for running:
  - LLM_Sentiment_Contribution.ipynb
  - LLM_sentiment_zero_shot.ipynb


LLM_Sentiment_Contribution.ipynb

Notebook: LLM_Sentiment_Contribution.ipynb
This notebook implements classical machine learning baselines for sentiment analysis on the IMDB dataset and compares them with zero-shot LLM predictions. It is intended to be executed in Google Colab after setting up the environment with phase1_env_setup.ipynb.

Purpose:
- Load the IMDB dataset from the LLM-Sentiment repository.
- Split the dataset into train (80%) and test (20%) sets for supervised learning.
- Transform review texts into TF-IDF features using unigrams and bigrams.
- Train a Logistic Regression classifier as a baseline model.
- Evaluate Logistic Regression using accuracy, classification report, and confusion matrix.
- Train an MLP (Multi-Layer Perceptron) Classifier on the same features.
- Evaluate MLP using accuracy, classification report, and confusion matrix.
- Compare Logistic Regression and MLP performance using a bar chart.
- Visualize results:
  - Confusion matrix heatmaps for both models.
  - MLP training loss curve.
- Save model predictions to outputs/ folder for further analysis.

How to Run This Notebook:

Step 1 — Ensure Environment is Set Up
- Run phase1_env_setup.ipynb first to mount Google Drive, clone the repository, and install all required packages.

Step 2 — Load IMDB Dataset
import pandas as pd
df_imdb = pd.read_csv("data/sc/imdb/test.csv")

Step 3 — Preprocess Data
Split into train/test sets.
Transform text using TF-IDF vectorization.

Step 4 — Train Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
logreg.fit(X_train_vec, y_train)

Step 5 — Evaluate Logistic Regression
Accuracy, classification report, confusion matrix.
Save predictions to outputs/contrib_logreg/imdb/prediction.csv

Step 6 — Train MLP Classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=20, random_state=42)
mlp.fit(X_train_vec, y_train)

Step 7 — Evaluate MLP
Accuracy, classification report, confusion matrix.

Step 8 — Visualize Performance
Bar chart comparing Logistic Regression vs MLP accuracy.
Confusion matrix heatmaps for both models.
MLP training loss curve.

Output:
Predictions CSV files for Logistic Regression and MLP.
Accuracy comparison chart.
Confusion matrix heatmaps.
MLP training loss curve.
Ready-to-analyze results comparing classical ML models with zero-shot LLM predictions.


LLM_sentiment_zero_shot.ipynb

Notebook: LLM_sentiment_zero_shot.ipynb
This notebook reproduces the zero-shot sentiment analysis experiments from the paper “Sentiment Analysis in the Era of Large Language Models: A Reality Check”. It runs the LLM-Sentiment pipeline using OpenAI models and generates predictions for multiple sentiment analysis tasks. The notebook is intended to be executed in Google Colab.

Purpose:
- Mount Google Drive to access datasets or save outputs.
- Clone the official LLM-Sentiment GitHub repository if not already present.
- Navigate to the repository folder and list its contents.
- Install all required Python libraries with compatible versions for Colab.
- Configure the OpenAI API key for accessing LLMs.
- Test a simple API call to verify that the environment works.
- Run the original zero-shot experiment script from the LLM-Sentiment repository.
- Generate model predictions for multiple sentiment analysis datasets.

How to Run This Notebook:

Step 1 — Open in Google Colab
- Upload the notebook to Colab or open using the Colab badge.

Step 2 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
This allows saving outputs and accessing datasets from Drive.

Step 3 — Clone the LLM-Sentiment Repository
import os
repo_path = "/content/LLM-Sentiment"
if not os.path.exists(repo_path):
    !git clone https://github.com/DAMO-NLP-SG/LLM-Sentiment.git
%cd /content/LLM-Sentiment
!ls

Step 4 — Install Dependencies
!pip install -q --upgrade pip
!pip install -q \
  "openai==0.27.2" \
  "tqdm>=4.65" \
  "tenacity>=8.2.2" \
  "numpy>=1.24" \
  "pandas>=2.0" \
  "scikit-learn>=1.4" \
  "transformers>=4.36" \
  "torch>=2.0"

Step 5 — Configure OpenAI API Key
import os, openai
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
openai.api_key = os.environ["OPENAI_API_KEY"]
Note: Replace "YOUR_API_KEY_HERE" with your actual OpenAI API key. You can generate it from https://platform.openai.com/account/api-keys.

Step 6 — Test API Call
resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello from Colab"}]
)
print(resp["choices"][0]["message"]["content"])

Step 7 — Run Zero-Shot Experiments
%cd /content/LLM-Sentiment
!bash script/run_zero_shot.sh
This executes the original zero-shot pipeline and generates predictions for all supported sentiment datasets.

Output:
Zero-shot predictions stored in the outputs/zero-shot/ folder of the repository.
Ready-to-use results for evaluation and comparison with classical ML models in LLM_Sentiment_Contribution.ipynb.
