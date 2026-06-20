# 1. Data Collection & Storage

In our cloud-native machine learning lifecycle, all files, models, and run outputs are managed on Azure. The first stage is **Data Collection & Storage**. We use **Azure Blob Storage** (similar to AWS S3) as our secure data repository.

For this guide, we will set up the storage account, create containers for our datasets, and upload our raw spam detection dataset (`spam.csv`).

---

## Part A — Manual Storage Setup (Azure Portal)

### Step 1: Create a Storage Account
1. Open the [Azure Portal](https://portal.azure.com/).
2. In the top search bar, type **Storage accounts** and select it.
3. Click the **+ Create** button in the top left.
4. Fill in the basics tab:
   - **Subscription:** Select your subscription.
   - **Resource Group:** Click **Create new** and name it `spam-detection-rg`.
   - **Storage account name:** Enter a unique lowercase name (e.g., `spamdetectionsa2026`).
   - **Region:** Select a region (e.g., `East US`).
   - **Performance:** **Standard**.
   - **Preferred storage type:** Select **Azure Blob Storage or Azure Data Lake Storage** (the recommended choice for machine learning datasets and models).
   - **Redundancy:** Select **Locally-redundant storage (LRS)** (recommended for lab, learning, and dev/test environments as it is the lowest-cost option).
   - **All Other Settings (Advanced, Networking, Data protection, Encryption, Tags):** You can safely **leave these at their default settings**!
5. Click **Review + create** at the bottom, then click **Create**.
6. Wait for deployment, and click **Go to resource**.

### Step 2: Create Containers for Raw & Processed Data
1. Inside your Storage Account page, scroll down the left menu to **Data storage** and click **Containers**.
2. Click **+ Container** at the top.
   - **Name:** `raw-data`
   - **Public access level:** **Private**
   - Click **Create**.
3. Click **+ Container** again.
   - **Name:** `processed-data`
   - **Public access level:** **Private**
   - Click **Create**.

### Step 3: Upload the Raw Dataset
1. Click on the `raw-data` container in the list.
2. Click **Upload** at the top.
3. Browse and select your local dataset file: `spam.csv`.
4. Click **Upload**.

---

## Part B — Programmatic Ingestion (Optional python upload)
If you want to upload the file programmatically, you can run this script:

```python
import os
from azure.storage.blob import BlobServiceClient

# Configure connection details (Get Connection String from Storage Account -> Access keys)
CONNECTION_STRING = "YOUR_STORAGE_CONNECTION_STRING"
CONTAINER_NAME = "raw-data"
LOCAL_FILE = "spam.csv"

def upload_to_azure():
    if not os.path.exists(LOCAL_FILE):
        print(f"Error: Local file '{LOCAL_FILE}' not found.")
        return

    # Connect and upload
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="spam.csv")
    
    print(f"Uploading {LOCAL_FILE} to Azure Blob Storage...")
    with open(LOCAL_FILE, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Upload successful! Blob URL: {blob_client.url}")

if __name__ == "__main__":
    upload_to_azure()
```

---

## 📦 Required Packages

To run the optional upload script:
```bash
pip install azure-storage-blob
```
