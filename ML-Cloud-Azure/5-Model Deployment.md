# 5. Model Deployment

In this stage, we will deploy our registered model to an **Azure ML Managed Online Endpoint** (a hosted web service API) directly from our **Jupyter Notebook** running on our **Compute Instance**. 

Once deployed, the endpoint will provide a public HTTP URL that we can consume locally using a FastAPI backend and web frontend to make spam predictions in real-time.

---

## Part A — Model Deployment in Notebook Cells

Run the following code blocks in the cells of your Jupyter Notebook:

### Cell 1: Write the Scoring Script
We use the Jupyter `%%writefile` magic command to write our prediction scoring script (`score.py`) directly from the notebook onto the Compute Instance storage.

```python
# Create a folder for deployment configurations
import os
os.makedirs("deploy_src", exist_ok=True)
```

```python
%%writefile deploy_src/score.py
import os
import logging
import json
import mlflow
import re

def clean_text(text):
    text = str(text).lower()                      # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)    # Keep only letters, numbers, and spaces
    text = re.sub(r'\s+', ' ', text).strip()      # Normalize whitespace
    return text

def init():
    """Runs once when the web service container starts. Loads the model."""
    global model
    
    # Get model folder path in container and load it
    model_path = os.getenv("AZUREML_MODEL_DIR")
    # If the files are nested under a subfolder, resolve that path
    if not os.path.exists(os.path.join(model_path, "MLmodel")):
        model_path = os.path.join(model_path, "spam_nb_model")
        
    logging.info(f"Loading model from: {model_path}")
    model = mlflow.sklearn.load_model(model_path)

def run(raw_data):
    """Runs on every incoming API request. Expects JSON input."""
    logging.info("Prediction request received.")
    try:
        data = json.loads(raw_data)
        messages = data["data"]  # Expects list of strings: {"data": ["msg1", "msg2"]}
        
        # Apply the exact same text cleaning logic as training
        cleaned_messages = [clean_text(msg) for msg in messages]
        
        # Predict spam vs ham
        predictions = model.predict(cleaned_messages)
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}
```

### Cell 2: Define and Create the Online Endpoint
This creates the HTTP URL endpoint (the API gateway/route) in the workspace.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint

# Connect to workspace
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Define endpoint configuration
endpoint_name = "spam-detector-endpoint-2026"
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Online endpoint for spam detection",
    auth_mode="key"
)

# Submit creation
print(f"Creating endpoint '{endpoint_name}'...")
ml_client.begin_create_or_update(endpoint).result()
print("Endpoint created successfully!")
```

> [!NOTE]
> **Troubleshooting — SubscriptionNotRegistered Error:**
> If running `begin_create_or_update(endpoint)` fails with a `SubscriptionNotRegistered` error, it means the required Azure Resource Providers are not registered in your subscription.
> Follow these steps in the Azure Portal to resolve it:
> 1. Open the [Azure Portal](https://portal.azure.com/).
> 2. In the top search bar, type **Subscriptions** and select it.
> 3. Click on your subscription name (e.g., *Azure subscription 1*).
> 4. In the left-hand navigation menu under **Settings**, click **Resource providers**.
> 5. Search for **`Microsoft.Cdn`**, click on its row, and click **Register** at the top of the table.
> 6. Search for **`Microsoft.PolicyInsights`**, click on its row, and click **Register** at the top of the table.
> 7. Wait 1–2 minutes until both status columns show **Registered**, then re-run this endpoint creation cell!
>
> **Troubleshooting — InferencingClientCallFailed / BadRequest Error:**
> If your previous execution failed due to registration issues, Azure ML might leave a corrupted, failed endpoint with that name in the workspace. Re-running the cell will yield an `InferencingClientCallFailed` error stating: *"Specified endpoint has not been created successfully. Please recreate the endpoint."*
> * **Fix:** Simply change the `endpoint_name` in your notebook cell to a new unique name (e.g., change `"spam-detector-endpoint"` to `"spam-detector-endpoint-2026"` or `"spam-detector-endpoint-v2"`) and run the cell again. This will create a fresh, healthy endpoint!

### Cell 3: Deploy the Model to the Endpoint
This provisions a virtual machine (e.g., `Standard_DS1_v2`), packages our model and scoring script in a Docker container, and hosts it.

```python
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration, Environment

# 1. Retrieve the latest registered model version dynamically to avoid hardcoding versions
latest_model = max(ml_client.models.list(name="spam_naive_bayes_model"), key=lambda x: int(x.version))
print(f"Deploying model '{latest_model.name}' version '{latest_model.version}'...")

# 2. Define a custom environment to avoid deprecated curated environment 404 errors
custom_env = Environment(
    name="spam-detector-env",
    version="1",
    description="Custom environment for spam detector",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",  # Standard active base image
    conda_file={
        "channels": ["conda-forge", "default"],
        "dependencies": [
            "python=3.10",
            "numpy",
            "pandas",
            "scikit-learn",
            "mlflow",
            {
                "pip": [
                    "azureml-inference-server-http",
                    "azureml-defaults"
                ]
            }
        ]
    }
)

# 3. Define deployment configuration
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=latest_model,
    environment=custom_env,  # Use custom environment
    code_configuration=CodeConfiguration(
        code="./deploy_src",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS1_v2",  # Hosting VM size (low-cost, fits trial quotas)
    instance_count=1                 # Scale to 1 instance
)

# 4. Deploy model onto ACI
print("Deploying model (takes ~5-10 minutes)...")
ml_client.begin_create_or_update(deployment).result()

# 5. Route 100% of endpoint traffic to this new deployment
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()
print("Deployment complete!")
```


> [!NOTE]
> **Troubleshooting — ImageBuildFailure / EmsClientError (404) Error:**
> If the deployment execution fails with an `ImageBuildFailure` and a `404` status code error referencing `AssetManagement` or `ImageAndModelPrepFlow`, it means the container builder is trying to deploy a corrupted or empty model version.
> This typically happens if the last registered version in the workspace (which the code retrieves automatically) was from a crashed run where the actual model files were never uploaded to storage.
> * **Fix:** Re-run your model training and registration steps to create a fresh, complete model version. Alternatively, override the dynamic loader by replacing `latest_model` with a specific known-healthy version, for example: `latest_model = ml_client.models.get(name="spam_naive_bayes_model", version="3")`.
>
> **Troubleshooting — Failed Deployment (blue) in Unrecoverable State:**
> If a previous deployment execution failed (like the one with version 1), the deployment resource named `blue` remains under your endpoint in a failed, unrecoverable state. Re-running the cell will throw an `InferencingClientCallFailed` error: *"Specified deployment [blue] failed during initial provisioning and is in an unrecoverable state. Delete and re-create."*
> * **Fix:** Change the deployment name in your notebook cell from `name="blue"` to `name="green"` (or `"blue-v2"`) and run the cell again. This triggers a fresh deployment container.
>
> **Troubleshooting — OutOfQuota Error:**
> If the deployment fails with an `OutOfQuota` error stating *"Not enough subscription CPU quota..."*, it means your Azure trial/student subscription has restricted VM allocations.
> * **Fix 1 (Cloud):** Try the smallest VM size, which is `instance_type="Standard_DS1_v2"` (1 vCPU). If your quota is 0, you must use Fix 2 (Local Deployment).
> * **Fix 2 (Local Deployment - Free & Bypasses Quota):** Run the deployment container directly inside your Compute Instance (which has Docker pre-configured). Change the deployment invocation to run locally:
>   ```python
>   ml_client.online_deployments.begin_create_or_update(deployment, local=True)
>   ```
>   To invoke and test a local deployment, you can check it using `ml_client.online_endpoints.invoke(...)` with local parameters.

### Cell 4: Test Endpoint Predictions in Notebook
Test the API endpoint to verify it works correctly. (Note: If you deployed locally in the previous step, add `local=True` to the `invoke` call):

```python
import json

# 1. Define the testing sample data
sample_payload = {
    "data": [
        "Congratulations! You won a free cruise. Call 1-800-spam to claim now.",
        "Hey, are we still meeting for lunch today?"
    ]
}

# 2. Write payload to a local JSON file (required by SDK v2 invoke method)
with open("sample_payload.json", "w") as f:
    json.dump(sample_payload, f)

# 3. Invoke the endpoint (add local=True if testing a local deployment)
response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    request_file="sample_payload.json"
)
print("Response from API:")
print(response)  # Expected: {"predictions": ["spam", "ham"]}
```

---

## Next Step

Now that your online model endpoint is deployed and verified, you can build a web application around it. Proceed to the next stage: **[6-Web App Deployment on Render](file:///Users/abakar/Desktop/AI%20Engineer%20Bible/AI%20Engineer%20Bible/ML-Cloud-Azure/6-Web%20App%20Deployment%20on%20Render.md)**.

---

## 📦 Required Packages

These are preinstalled in your Compute Instance environment:
```bash
pip install azure-ai-ml azure-identity
```
