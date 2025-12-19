# NFL Match Predictor

A machine learning-powered web application that predicts NFL game outcomes using ensemble models trained on historical data.

## Overview

This project uses three ML models (Logistic Regression, Decision Tree, Random Forest) combined through ensemble voting to predict NFL game winners. The system pulls real-time schedule data and calculates team statistics to generate predictions with confidence scores.

**Key Feature:** Predictions for past weeks use only data that was available at that time—no future information leaks into historical predictions.

## Features

- **Weekly Predictions** — Get predictions for all games in any NFL week
- **Real-time Data** — Pulls current season schedules and scores via `nfl_data_py`
- **Ensemble Voting** — Three models vote on each prediction for improved accuracy
- **Confidence Scores** — See how confident the model is in each prediction
- **Accuracy Tracking** — For completed games, see which predictions were correct
- **Historical Analysis** — View predictions for past weeks using only pre-game data

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| ML Models | scikit-learn |
| Data | nfl_data_py, pandas, numpy |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Docker, Google Cloud Run |

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NFL_match_predictor.git
cd NFL_match_predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train Models

```bash
python train_and_save_model.py
```

This generates the model artifacts:
- `model_logistic_regression.pkl`
- `model_decision_tree.pkl`
- `model_random_forest.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

### Run the Application

```bash
python app.py
```

Open http://localhost:8080 in your browser.

## Usage

1. Select a **season** (2024 or 2025)
2. Select a **week** (1-18, or playoff rounds)
3. Click **Get Predictions**

The app displays:
- Matchup details for each game
- Predicted winner with confidence percentage
- Individual model votes (Decision Tree, Logistic Regression, Random Forest)
- For completed games: actual scores and prediction accuracy

## How It Works

### Data Pipeline
1. Fetches NFL schedule data for the selected season
2. Filters completed games from weeks *before* the prediction week
3. Calculates team averages (points scored, points allowed, win %)
4. Combines current season stats (90%) with previous season baseline (10%)

### Prediction Model
1. Creates feature vectors for each matchup (home/away stats + differentials)
2. Scales features using pre-fitted StandardScaler
3. Each model makes an independent prediction
4. Final prediction determined by majority vote
5. Confidence calculated as average probability across models

### Features Used (12 total)
- Points scored per game (home, away, differential)
- Points allowed per game (home, away, differential)
- Home field factor (home, away, differential)
- Win percentage (home, away, differential)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api` | GET | API status |
| `/health` | GET | Detailed health check |
| `/teams` | GET | List all NFL teams |
| `/predict_game` | POST | Predict single matchup |
| `/predict_week` | POST | Predict all games for a week |

See [README_API.md](README_API.md) for detailed API documentation.

## Docker Deployment

```bash
# Build image
docker build -t nfl-predictor .

# Run container
docker run -p 8080:8080 nfl-predictor
```

## Project Structure

```
NFL_match_predictor/
├── app.py                    # Flask application
├── train_and_save_model.py   # Model training script
├── templates/
│   └── index.html            # Web interface
├── model_*.pkl               # Trained model files
├── scaler.pkl                # Feature scaler
├── feature_columns.pkl       # Feature configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container configuration
├── deploy.sh                 # Local deployment script
├── README.md                 # This file
└── README_API.md             # API documentation
```

## Model Performance

- **Training Data:** 2022-2024 NFL seasons (~1,000+ games)
- **Typical Accuracy:** 55-65% on test data
- **Response Time:** <100ms per prediction

*Note: NFL game prediction is inherently difficult. Even Vegas odds typically achieve ~65-68% accuracy.*

## License

MIT

---

Built with Python and scikit-learn

