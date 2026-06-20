# 2. Data Preparation & EDA

Before training a model, we must explore and clean the dataset. In Azure, you can manage and track datasets securely using **Datastores** (connections to storage services) and **Data Assets** (versioned data references) inside **Azure Machine Learning Studio**.

We will set up our workspace, spin up a single **Compute Instance** (which will host our entire end-to-end MLOps pipeline notebook), and prepare our data.

---

## Part A — Manual Setup (Azure ML Studio)

### Step 1: Create an Azure Machine Learning Workspace
If you don't have an Azure ML Workspace yet:
1. Search for **Azure Machine Learning** in the Azure Portal.
2. Click **+ Create** ➔ **New workspace**.
3. In the **Basics** tab, configure the following:
   - **Subscription:** Select your subscription.
   - **Resource Group:** Choose the one you created in Step 1 (e.g., `spam-detection-rg`).
   - **Name:** Enter `spam-detection-mlw`.
   - **Region:** Choose the same region as your storage account (e.g., `East US`).
   - **Storage account, Key vault, Application insights:** Leave these at their default **Create new** settings. Azure will automatically generate its own default storage account to save workspace system logs and internal run artifacts. *(Note: Keep the "Create new" default here; do not select the storage account you created in Guide 1. We will link your Guide 1 storage account as a Datastore in Step 2 below).*
   - **Container registry:** Keep it at **None** (Azure will automatically create one later when we build custom environments/containers).
4. **All Other Tabs (Inbound Access, Outbound Access, Encryption, Identity, Tags):** You can safely ignore these and **leave them at their default settings** for this tutorial.
5. Click **Review + create** at the bottom, then click **Create**.
6. Once deployed, click **Go to resource**, then click **Launch studio** (which takes you to `ml.azure.com`).

### Step 2: Register a Datastore
A Datastore securely saves connection details to your Azure Blob Storage so you don't have to hardcode connection strings in your scripts.
1. In Azure ML Studio, click **Data** in the left menu.
2. Select the **Datastores** tab at the top, then click **+ Create**.
3. Configure the Datastore:
   - **Datastore name:** `spam_storage_ds`
   - **Datastore type:** **Azure Blob Storage**
   - **Account selection method:** From Azure subscription.
   - **Storage account:** Choose the storage account you created in Step 1 (e.g., `spamdetectionsa2026` or `v2spamdetection2026`).
   - **Blob container:** Select `raw-data` (which has your `spam.csv`).
   - **Authentication type:** Select **Account key**.
   - **Account key:** Paste the access key of your Storage Account (copy this from your Storage Account page ➔ **Access keys** ➔ **key1 Key** in the Azure Portal).
     *(Note: Do not turn off authentication or choose "None", since the container is Private and Azure ML needs this key to access the files).*
4. Click **Create**.

### Step 3: Register a Data Asset
1. In Azure ML Studio, go to **Data** on the left menu, select the **Data assets** tab, and click **+ Create**.
2. Set Name: `spam_raw_data`, Type: **File (uri_file)**. Click **Next**.
3. Choose **From Azure storage** ➔ Select the Datastore `spam_storage_ds` ➔ Browse to `spam.csv`. Click **Next**.
4. Review the schema details and click **Create**.

### Step 4: Create a Compute Instance to Run Notebooks
We will create a single Compute Instance that serves as our development environment for the entire MLOps workflow.
1. Go to **Compute** in the left menu of Azure ML Studio.
2. Under the **Compute instances** tab, click **+ New**.
3. Configure the following settings:
   - **Compute name:** Enter a unique name (e.g., `v2spamdetection-instance`).
   - **Virtual machine type:** Select **CPU**.
   - **Virtual machine size:** Select **Standard_DS11_v2** (2 cores, 14GB RAM, 28GB storage).
4. Click **Review + create** to review your settings. Verify they match these recommended defaults:
   - **Scheduling:** Auto shutdown should be **Enabled** (set to shut down after 60 minutes of inactivity to prevent extra charges).
   - **Security:** SSH: `no`, Virtual network: `no`, Root access: `yes`, SSO: `yes`, Managed identity: `no`.
   - **Startup script, Creation script, Tags:** Leave empty/default.
5. Click **Create** (takes ~3 minutes).
6. Once running, go to **Notebooks** in the left menu, click **+** ➔ **Create new file**, select **Notebook (.ipynb)**, and attach it to your running compute instance. Choose the preinstalled kernel: **Python 3.10 - SDK v2**.

---

## Part B — Data Preparation Notebook Cells

Write this Python code in the cells of your Jupyter Notebook to clean the dataset:

```python
import os
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# 1. Connect to the Azure ML Workspace
# DefaultAzureCredential will automatically authenticate you inside the Compute Instance!
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# 2. Retrieve the registered Data Asset
data_asset = ml_client.data.get(name="spam_raw_data", version="1")
print(f"Loading data from: {data_asset.path}")

# 3. Read dataset into Pandas
df = pd.read_csv(data_asset.path)

# 4. Exploratory Data Analysis (EDA)
print(f"Total records: {len(df)}")
print(df['label'].value_counts())  # Class balance

# 5. Clean Dataset
# Select only relevant columns, drop duplicates, and clean text explicitly
df = df[['label', 'message']]
df = df.dropna()
df = df.drop_duplicates()

import re
def clean_text(text):
    text = str(text).lower()                      # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)    # Keep only letters, numbers, and spaces
    text = re.sub(r'\s+', ' ', text).strip()      # Normalize whitespace
    return text

df["message"] = df["message"].apply(clean_text)

# 6. Save the cleaned CSV file locally on the Compute Instance storage
cleaned_file = "spam_clean.csv"
df.to_csv(cleaned_file, index=False)
print("Data cleaned successfully!")

# 7. Upload cleaned dataset back to the 'processed-data' Blob container

CONNECTION_STRING = "YOUR_STORAGE_CONNECTION_STRING"
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
blob_client = blob_service_client.get_blob_client(container="processed-data", blob="spam_clean.csv")

print("Uploading cleaned dataset to 'processed-data' container...")
with open(cleaned_file, "rb") as file_data:
    blob_client.upload_blob(file_data, overwrite=True)
print("Data Preparation complete!")
```
 
 Where to get the Connection String: (YOUR_STORAGE_CONNECTION_STRING)
1. Open the Azure Portal and go to your Storage Account (e.g., v2spamdetection2026).
2. In the left navigation pane under 'Security + networking', click 'Access keys'.
3. Click 'Show' next to the **Connection string** under **key1**, and copy it.
---

## 📦 Required Packages

Note: Azure ML Compute Instances come pre-configured with all ML packages and SDKs. If you need to verify or run the scripts outside the default environment, install:
```bash
pip install azure-ai-ml azure-identity azure-storage-blob pandas
```
