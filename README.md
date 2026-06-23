# Spam-Detection-Azure-v1

> **AI-powered spam detection** — A FastAPI web app that classifies messages as **Spam** or **Ham** using a Multinomial Naive Bayes model deployed on Azure ML.

## 🚀 Live Demo

**[https://spam-detection-svis.onrender.com/](https://spam-detection-svis.onrender.com/)**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Multinomial Naive Bayes (scikit-learn) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML / CSS / JavaScript (glassmorphism UI) |
| **ML Endpoint** | Azure Machine Learning Online Endpoint |
| **Deployment** | Render (Docker) |
| **Container** | Dockerfile |

## ✨ Features

- 🔍 Real-time spam detection via Azure ML inference
- 🎨 Premium dark-mode glassmorphism UI with animated background
- ⚡ Fast API response with loading states
- 📱 Fully responsive design
- 🐳 Dockerized for easy deployment

## 📂 Project Structure

```
spam-app/
├── main.py                # FastAPI app + /predict endpoint
├── templates/
│   └── index.html         # Frontend UI
├── ML-Cloud-Azure/        # Azure ML deployment docs & guides
├── spam.csv               # Training dataset
├── Dockerfile             # Container config
├── requirements.txt       # Python dependencies
└── .env                   # Azure ML endpoint URL & key (not committed)
```

## 🏃 Local Setup

```bash
# Clone the repo
git clone https://github.com/sidicksino/Spam-Detection-Azure-v1.git
cd Spam-Detection-Azure-v1

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (.env)
AZURE_ENDPOINT_URL=<your-azure-endpoint>
AZURE_API_KEY=<your-azure-key>

# Run
uvicorn main:app --reload
```

## 📜 License

MIT
