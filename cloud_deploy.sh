#!/bin/bash

echo "☁️  NFL Predictor - Google Cloud Deployment"
echo "==========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No Google Cloud project set. Please run:"
    echo "   gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "📋 Using Google Cloud Project: $PROJECT_ID"

# Step 1: Train models locally
echo ""
echo "📚 Step 1: Training models locally..."
python3 train_and_save_model.py

if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi

# Step 2: Build and submit to Cloud Build
echo ""
echo "🏗️  Step 2: Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/nfl-predictor

if [ $? -ne 0 ]; then
    echo "❌ Container build failed!"
    exit 1
fi

# Step 3: Deploy to Cloud Run
echo ""
echo "🚀 Step 3: Deploying to Cloud Run..."
gcloud run deploy nfl-predictor \
    --image gcr.io/$PROJECT_ID/nfl-predictor \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 300

if [ $? -ne 0 ]; then
    echo "❌ Cloud Run deployment failed!"
    exit 1
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe nfl-predictor --region=us-central1 --format='value(status.url)')

echo ""
echo "✅ Deployment successful!"
echo "🌐 Your API is now live at: $SERVICE_URL"
echo ""
echo "🧪 Test your API:"
echo "   Health check: curl $SERVICE_URL/health"
echo "   Teams list:   curl $SERVICE_URL/teams"
echo "   Prediction:   curl -X POST $SERVICE_URL/predict_game -H 'Content-Type: application/json' -d '{\"home_team\": \"KC\", \"away_team\": \"BUF\"}'"
echo ""
echo "🎉 Your resume bullet point is now 100% accurate!"
echo "   Deployed a Flask-based REST API on Google Cloud Run to deliver win probabilities and score forecasts for live interaction."
