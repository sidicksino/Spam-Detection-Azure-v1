# 4. Evaluation & Tuning

Once your training cells have run, you can track and evaluate model performance metrics, compare runs, and optimize hyperparameters. 

Since we are running everything in Azure, we track metrics using the **Azure ML Jobs Dashboard** and can perform hyperparameter optimization directly in our notebook.

---

## Part A — Metric Tracking & Evaluation (Azure ML Studio UI)

Any metrics logged in your training script via `mlflow.log_metric()` (like accuracy) are saved automatically in your workspace history.

### How to View Metrics Manually
1. Open [Azure ML Studio](https://ml.azure.com/).
2. On the left sidebar, click **Jobs**.
3. Select your experiment name (e.g., `spam-detection-training`).
4. Click on your training run (named with a random adjective-noun string).
5. Click on the **Metrics** tab to view logged charts (accuracy, custom loss, F1-scores).
6. To compare multiple runs side-by-side: Go back to the experiment runs list, check the boxes next to different runs, and click **Compare** at the top.

---

## Part B — Hyperparameter Tuning (Simple Notebook Sweep)

Instead of setting up complicated Sweep clusters, we can perform a hyperparameter sweep directly inside our notebook using a Python loop and MLflow. 

Run this cell in your notebook to search for the best `alpha` value for Naive Bayes:

```python
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score

# Define values to search over
alpha_values = [0.1, 0.5, 1.0, 2.0]

for alpha in alpha_values:
    # Start a nested child run for each parameter configuration
    with mlflow.start_run(run_name=f"NB_Alpha_{alpha}", nested=True):
        
        # Train model using Pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', MultinomialNB(alpha=alpha))
        ])
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        predictions = pipeline.predict(X_test)
        accuracy = (predictions == y_test).mean()
        precision = precision_score(y_test, predictions, pos_label='spam')
        recall = recall_score(y_test, predictions, pos_label='spam')
        f1 = f1_score(y_test, predictions, pos_label='spam')
        
        # Log to Azure ML run dashboard
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Spam_Precision", precision)
        mlflow.log_metric("Spam_Recall", recall)
        mlflow.log_metric("Spam_F1_Score", f1)
        
        print(f"Alpha: {alpha} | Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
```
Go back to the **Jobs** tab in Azure ML Studio to compare these 4 runs side-by-side to see which alpha value yielded the highest accuracy!

---

## Part C — Automated ML (AutoML via UI)

You can also run Automated ML (which automatically tries different models like Logistic Regression, Random Forest, etc.) using serverless compute without writing code:
1. In Azure ML Studio, click **Automated ML** on the left menu.
2. Click **+ New Automated ML job**.
3. Choose your dataset: Select `spam_raw_data`. Click **Next**.
4. Configure Job:
   - **Task type:** **Classification**.
   - **Target column:** Select `label` (the column containing 'ham' or 'spam').
   - **Compute type:** Select **Serverless** (this runs the job on temporary serverless VMs managed by Azure, meaning you don't have to create your own compute cluster!).
5. Click **Submit** to run.

---

## 📦 Required Packages

These are preinstalled in your Compute Instance environment:
```bash
pip install azure-ai-ml azure-identity "mlflow<2.8" azureml-mlflow pandas scikit-learn
```
