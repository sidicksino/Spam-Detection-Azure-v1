# 3. Model Training

In this guide, we will train our Multinomial Naive Bayes model (highly suited for spam classification) directly in our **Jupyter Notebook** running on our single **Compute Instance** (`v2spamdetection-instance`). 

We do **not** need to create separate training clusters or setup external CLI configurations. All training execution, tracking, and model registering will happen inside the notebook cells.

---

## Part A — Model Training in Notebook Cells

Run the following code blocks in the cells of your Jupyter Notebook:

### Cell 1: Load Clean Dataset & Split Data
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the cleaned dataset saved in Guide 2
clean_data_path = "spam_clean.csv"
df = pd.read_csv(clean_data_path)

# 2. Split features and labels
X = df['message'].values.astype('U')  # Message text
y = df['label'].values                # 'ham' or 'spam' label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
```

### Cell 2: Train & Evaluate the Model
We Vectorize the text using TF-IDF and train a Naive Bayes classifier. We wrap the training loop with **MLflow** so that parameters and metrics are automatically tracked inside our Azure ML Workspace history!

```python
import os
import shutil
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Start MLflow run to log performance metrics
mlflow.start_run()

print("Building and training Pipeline (TF-IDF Vectorizer + Multinomial NB)...")
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Fit pipeline on raw text directly
pipeline.fit(X_train, y_train)

# 2. Evaluate model
predictions = pipeline.predict(X_test)
accuracy = (predictions == y_test).mean()
precision = precision_score(y_test, predictions, pos_label='spam')
recall = recall_score(y_test, predictions, pos_label='spam')
f1 = f1_score(y_test, predictions, pos_label='spam')

print(f"Model Accuracy:  {accuracy:.4f}")
print(f"Spam Precision:  {precision:.4f}")
print(f"Spam Recall:     {recall:.4f}")
print(f"Spam F1-Score:   {f1:.4f}")

# 3. Log metrics to Azure ML run history
mlflow.log_metric("Accuracy", accuracy)
mlflow.log_metric("Spam_Precision", precision)
mlflow.log_metric("Spam_Recall", recall)
mlflow.log_metric("Spam_F1_Score", f1)
mlflow.log_param("Model Type", "Multinomial Naive Bayes")
mlflow.log_param("Vectorizer Type", "TfidfVectorizer")

# 4. Save the model locally as an MLflow model (saves the entire pipeline!)
# This bypasses the Azure ML server REST API compatibility issues
if os.path.exists("spam_nb_model"):
    shutil.rmtree("spam_nb_model")

mlflow.sklearn.save_model(
    sk_model=pipeline,
    path="spam_nb_model"
)

mlflow.end_run()
print("Model training run complete!")
```

### Cell 3: Register the Model in the Workspace Registry
Registering the model stores it in the central registry so it can be versioned and deployed as an endpoint.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# 1. Connect to Azure ML Workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# 2. Define the Model parameters (pointing to the local saved folder)
model_name = "spam_naive_bayes_model"
registered_model = Model(
    path="spam_nb_model",
    name=model_name,
    description="Multinomial Naive Bayes model for spam classification.",
    type=AssetTypes.MLFLOW_MODEL
)

# 3. Register model to workspace registry
print(f"Registering model '{model_name}' in the Azure ML registry...")
ml_client.models.create_or_update(registered_model)
print("Model registered successfully!")
```

---

## 📦 Required Packages

Since we are running this inside the preconfigured **Python 3.10 - SDK v2** conda environment on the Compute Instance, these packages are already installed. If you want to configure them locally or in a custom environment:
```bash
pip install azure-ai-ml azure-identity "mlflow<2.8" azureml-mlflow pandas scikit-learn
```
