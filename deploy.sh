#!/bin/bash

echo "🏈 NFL Predictor - Deployment Script"
echo "===================================="

# Step 1: Train and save models
echo ""
echo "📚 Step 1: Training and saving models..."
python3 train_and_save_model.py
 
if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi

echo "✅ Models trained and saved successfully!"

# Step 2: Build Docker image
echo ""
echo "🐳 Step 2: Building Docker image..."
docker build -t nfl-predictor .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully!"

# Step 3: Run locally for testing
echo ""
echo "🚀 Step 3: Starting local container for testing..."
echo "Container will run on http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the container"
echo ""

docker run -p 8080:8080 nfl-predictor
