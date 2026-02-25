#  Hogwarts Sorting Hat Classifier

Web application that predicts your Hogwarts house using machine learning.

**Live Demo**: https://hogwarts-sorting-hat-358396978468.us-central1.run.app

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Development Process](#development-process)
- [Deployment](#deployment)
- [Local Setup](#local-setup)
- [Team](#team)

---

## Overview

This project implements a multi-class classification system to predict which Hogwarts house a person belongs to based on their personality traits and skills. The application uses a Logistic Regression model trained on the Harry Potter Sorting Dataset from Kaggle.

**Model Performance:**
- Accuracy: 89%
- F1-Score: 87%
- Balanced Accuracy: 88%

---

## Features

- **Interactive Prediction Interface**: Input personality traits and get instant house predictions
- **Probability Visualization**: See confidence scores for all four houses
- **Skills Radar Chart**: Visual representation of your magical abilities
- **Data Insights**: Explore dataset statistics and model performance
- **Model Information**: Detailed architecture and training metrics
- **Responsive Design**: Modern dark theme with smooth animations
- **Cloud Deployment**: Serverless hosting on Google Cloud Run

---

## Technology Stack

### Machine Learning
- **Scikit-learn 1.6.1**: Model training and pipeline
- **Pandas 2.0.3**: Data manipulation
- **NumPy 1.26+**: Numerical computing
- **Joblib 1.3.2**: Model serialization

### Web Application
- **Streamlit 1.28.0**: Web framework
- **Plotly 5.17.0**: Interactive visualizations
- **Custom CSS**: Modern UI styling

### Deployment
- **Docker**: Containerization
- **Google Cloud Run**: Serverless deployment
- **Artifact Registry**: Container storage
- **Python 3.9**: Runtime environment

---

## Project Structure

```
ProyectoHP/
├── app.py                          # Main Streamlit application
├── Dockerfile                      # Container configuration
├── requirements.txt                # Python dependencies
├── .dockerignore                   # Docker build exclusions
├── models/
│   ├── best_model_logistic_regression.joblib
│   └── label_encoder.joblib
├── Proyecto_P1_ICO.ipynb          # Training notebook
├── README.md                       # This file
├── DEPLOYMENT.md                   # Deployment documentation
└── QUICK_REFERENCE.txt            # Quick commands reference
```

---

## Development Process

### 1. Data Preparation & EDA

**Dataset**: Harry Potter Sorting Dataset from Kaggle
- 1,000+ student records
- 9 features (1 categorical, 8 numeric)
- 4 target classes (Gryffindor, Hufflepuff, Ravenclaw, Slytherin)

**Exploratory Analysis** (in `Proyecto_P1_ICO.ipynb`):
```python
import pandas as pd
import kagglehub

# Download dataset
path = kagglehub.dataset_download("sahityapalacharla/harry-potter-sorting-dataset")
df = pd.read_csv(f"{path}/sorting_hat.csv")

# Analyze distributions, correlations, missing values
df.describe()
df['House'].value_counts()
```

### 2. Model Training

**Preprocessing Pipeline**:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Numeric features: median imputation + scaling
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features: mode imputation + one-hot encoding
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

**Model Selection**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Create full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Hyperparameter tuning
param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=StratifiedKFold(5),
    scoring='f1_macro'
)

grid_search.fit(X_train, y_train)
```

**Model Export**:
```python
import joblib

# Save best model and label encoder
joblib.dump(grid_search.best_estimator_, 'models/best_model_logistic_regression.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')
```

### 3. Web Application Development

**Core Application** (`app.py`):

```python
import streamlit as st
import joblib
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model_logistic_regression.joblib")
    le = joblib.load("models/label_encoder.joblib")
    return model, le

model, le = load_model()

# Prediction interface
st.title(" THE SORTING HAT")

# Input features
bravery = st.slider("Bravery", 1, 10, 5)
intelligence = st.slider("Intelligence", 1, 10, 5)
# ... more features

# Make prediction
if st.button("THE SORTING HAT DECIDES"):
    input_data = pd.DataFrame([{
        "Blood Status": blood_status,
        "Bravery": bravery,
        # ... all features
    }])
    
    prediction = model.predict(input_data)[0]
    house = le.inverse_transform([prediction])[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.success(f"You belong in {house}!")
```

**UI Enhancements**:
- Custom CSS for dark theme with gradients
- Google Fonts (Cinzel for headers, Inter for body)
- House-specific color schemes
- Plotly interactive charts
- Smooth animations and transitions

### 4. Containerization

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y build-essential curl

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Configure Streamlit
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "port = 8080" >> ~/.streamlit/config.toml

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

**Requirements** (`requirements.txt`):
```
numpy>=1.26.0
pandas==2.0.3
scikit-learn>=1.5.0
joblib==1.3.2
streamlit==1.28.0
plotly==5.17.0
```

### 5. Cloud Deployment

**Step 1: Create GCP Project**
```bash
# Generate unique project ID
PROJECT_ID="hogwarts-sorting-$(date +%s | tail -c 7)"

# Create project
gcloud projects create $PROJECT_ID --name="Hogwarts Sorting Hat"

# Set active project
gcloud config set project $PROJECT_ID

# Link billing account
BILLING_ACCOUNT=$(gcloud billing accounts list --format="value(name)" | head -1)
gcloud billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT
```

**Step 2: Enable APIs**
```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com
```

**Step 3: Create Artifact Registry**
```bash
REGION="us-central1"

gcloud artifacts repositories create hogwarts-repo \
    --repository-format=docker \
    --location=$REGION \
    --description="Hogwarts Sorting Hat"

# Configure Docker authentication
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

**Step 4: Build Container**
```bash
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/hogwarts-repo/hogwarts-sorting-hat"

# Build for linux/amd64 (Cloud Run architecture)
docker build --platform linux/amd64 -t $IMAGE_NAME .

# Push to registry
docker push $IMAGE_NAME
```

**Step 5: Deploy to Cloud Run**
```bash
SERVICE_NAME="hogwarts-sorting-hat"

gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --port 8080
```

**Deployment Output**:
```
Service URL: https://hogwarts-sorting-hat-358396978468.us-central1.run.app
```

---

## Local Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

1. **Clone the repository**
```bash
cd ProyectoHP
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the notebook** (optional - models already included)
```bash
jupyter notebook Proyecto_P1_ICO.ipynb
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Deployment Commands

### View Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND \
    resource.labels.service_name=hogwarts-sorting-hat" \
    --limit 50 --project hogwarts-sorting-955744
```

### Update Deployment
```bash
# Rebuild
docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/hogwarts-sorting-955744/hogwarts-repo/hogwarts-sorting-hat .

# Push
docker push us-central1-docker.pkg.dev/hogwarts-sorting-955744/hogwarts-repo/hogwarts-sorting-hat

# Deploy
gcloud run deploy hogwarts-sorting-hat \
    --image us-central1-docker.pkg.dev/hogwarts-sorting-955744/hogwarts-repo/hogwarts-sorting-hat \
    --region us-central1 \
    --project hogwarts-sorting-955744
```

### Delete Resources
```bash
# Delete service
gcloud run services delete hogwarts-sorting-hat \
    --region us-central1 \
    --project hogwarts-sorting-955744

# Delete entire project
gcloud projects delete hogwarts-sorting-955744
```

---

## Cost Estimation

**Google Cloud Run Pricing**:
- Free tier: 2 million requests/month
- After free tier: ~$0.00002 per request
- Idle time: $0 (serverless)

**Expected monthly cost**: $0-5 for moderate usage

---

## Team

**Escuela de Ingeniería**  
**Ingeniería en Ciencias Computacionales**  
**Inteligencia Computacional**

**Proyecto 1 - Aplicación con funcionalidades de ML**

- David Alexandro García Morales (31624)
- Luis Bernardo Bremer Ortega (32366)
- Edgar Daniel De la Torre Reza (34887)
- Hugo German Manzano López (36231)

---

## License

This project was developed for educational purposes as part of the Computational Intelligence course.

---

## Acknowledgments

- Dataset: [Harry Potter Sorting Dataset](https://www.kaggle.com/datasets/sahityapalacharla/harry-potter-sorting-dataset) by Sahitya Palacharla on Kaggle
- Framework: Streamlit for rapid web app development
- Deployment: Google Cloud Run for serverless hosting

---

**Live Application**: https://hogwarts-sorting-hat-358396978468.us-central1.run.app
