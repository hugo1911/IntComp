#!/bin/bash

# Configuration
PROJECT_ID="hogwarts-sorting-$(date +%s)"
REGION="us-central1"
SERVICE_NAME="hogwarts-sorting-hat"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🔮 Deploying Hogwarts Sorting Hat to Google Cloud Run"
echo "=================================================="

# Create new GCP project
echo "📦 Creating new GCP project: ${PROJECT_ID}"
gcloud projects create ${PROJECT_ID} --name="Hogwarts Sorting Hat"

# Set project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build container
echo "🏗️  Building Docker container..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10

echo ""
echo "✅ Deployment complete!"
echo "=================================================="
echo "🌐 Your app is now live!"
echo ""
echo "Project ID: ${PROJECT_ID}"
echo "Service URL: $(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')"
echo ""
echo "💡 To view logs: gcloud run logs read --service ${SERVICE_NAME} --region ${REGION}"
echo "💡 To delete: gcloud projects delete ${PROJECT_ID}"
