#!/usr/bin/env python3
"""
NFL Predictor Flask REST API
Serves NFL game predictions via HTTP endpoints
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import nfl_data_py as nfl
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for models and preprocessing
models = {}
scaler = None
feature_columns = []

def load_artifacts():
    """Load trained models and preprocessing artifacts"""
    global models, scaler, feature_columns
    
    print("Loading trained models and artifacts...")
    
    try:
        # Load models
        model_files = {
            'Logistic Regression': 'model_logistic_regression.pkl',
            'Decision Tree': 'model_decision_tree.pkl', 
            'Random Forest': 'model_random_forest.pkl'
        }
        
        for name, filename in model_files.items():
            if os.path.exists(filename):
                models[name] = joblib.load(filename)
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠️  {filename} not found")
        
        # Load scaler
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            print("✅ Loaded scaler")
        else:
            print("⚠️  scaler.pkl not found")
        
        # Load feature columns
        if os.path.exists('feature_columns.pkl'):
            feature_columns = joblib.load('feature_columns.pkl')
            print("✅ Loaded feature columns")
        else:
            print("⚠️  feature_columns.pkl not found")
        
        if not models or scaler is None or not feature_columns:
            print("❌ Missing required artifacts. Run train_and_save_model.py first!")
            return False
        
        print(f"🚀 Successfully loaded {len(models)} models with {len(feature_columns)} features")
        return True
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return False

def prepare_team_stats(schedules):
    """Prepare team statistics from schedule data"""
    team_stats = []
    
    for _, game in schedules.iterrows():
        if pd.notna(game['home_score']) and pd.notna(game['away_score']):
            # Home team stats
            team_stats.append({
                'team': game['home_team'],
                'season': game['season'],
                'game_id': game['game_id'],
                'points_scored': game['home_score'],
                'points_allowed': game['away_score'],
                'home_game': 1,
                'won': 1 if game['home_score'] > game['away_score'] else 0
            })
            
            # Away team stats  
            team_stats.append({
                'team': game['away_team'],
                'season': game['season'],
                'game_id': game['game_id'],
                'points_scored': game['away_score'],
                'points_allowed': game['home_score'],
                'home_game': 0,
                'won': 1 if game['away_score'] > game['home_score'] else 0
            })
    
    team_stats_df = pd.DataFrame(team_stats)
    
    # Calculate averages per team per season
    team_averages = team_stats_df.groupby(['team', 'season']).agg({
        'points_scored': 'mean',
        'points_allowed': 'mean',
        'home_game': 'mean',  # Home field advantage
        'won': 'mean'  # Win percentage
    }).reset_index()
    
    return team_averages

def combine_weighted_stats(stats_2024, stats_2025):
    """Combine 2024 and 2025 stats with weighted averages (90% current, 10% baseline)"""
    all_teams = set(stats_2024['team'].unique()) | set(stats_2025['team'].unique())
    
    combined_stats = []
    
    for team in all_teams:
        team_2024 = stats_2024[stats_2024['team'] == team]
        team_2025 = stats_2025[stats_2025['team'] == team]
        
        if len(team_2024) > 0 and len(team_2025) > 0:
            # Both seasons available - use weighted average
            stats_2024_row = team_2024.iloc[0]
            stats_2025_row = team_2025.iloc[0]
            
            combined_row = {
                'team': team,
                'season': 2025,
            }
            
            # Weight the numerical stats (90% current season, 10% last season)
            for col in ['points_scored', 'points_allowed', 'home_game', 'won']:
                if col in stats_2024_row and col in stats_2025_row:
                    weighted_value = (stats_2024_row[col] * 0.1) + (stats_2025_row[col] * 0.9)
                    combined_row[col] = weighted_value
            
            combined_stats.append(combined_row)
            
        elif len(team_2025) > 0:
            # Only 2025 data available
            stats_2025_row = team_2025.iloc[0]
            combined_row = {
                'team': team,
                'season': 2025,
                'points_scored': stats_2025_row['points_scored'],
                'points_allowed': stats_2025_row['points_allowed'], 
                'home_game': stats_2025_row['home_game'],
                'won': stats_2025_row['won']
            }
            combined_stats.append(combined_row)
            
        elif len(team_2024) > 0:
            # Only 2024 data available
            stats_2024_row = team_2024.iloc[0]
            combined_row = {
                'team': team,
                'season': 2025,
                'points_scored': stats_2024_row['points_scored'],
                'points_allowed': stats_2024_row['points_allowed'],
                'home_game': stats_2024_row['home_game'], 
                'won': stats_2024_row['won']
            }
            combined_stats.append(combined_row)
    
    return pd.DataFrame(combined_stats)

def ensemble_predict(X):
    """Make ensemble predictions using voting"""
    predictions = {}
    probabilities = {}
    
    # Get predictions from each model
    for name, model in models.items():
        pred = model.predict(X)
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[:, 1]  # Probability of home team winning
        else:
            prob = pred.astype(float)
        
        predictions[name] = pred
        probabilities[name] = prob
    
    # Voting ensemble
    pred_df = pd.DataFrame(predictions)
    ensemble_pred = pred_df.mode(axis=1)[0].astype(int)
    
    # Average probabilities for confidence
    prob_df = pd.DataFrame(probabilities)
    ensemble_prob = prob_df.mean(axis=1)
    
    return ensemble_pred, ensemble_prob, predictions

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'NFL Predictor API running',
        'models_loaded': len(models),
        'features': len(feature_columns)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict game outcome from features"""
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features array'}), 400
        
        # Convert features to numpy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Validate feature count
        if features.shape[1] != len(feature_columns):
            return jsonify({
                'error': f'Expected {len(feature_columns)} features, got {features.shape[1]}'
            }), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        ensemble_pred, ensemble_prob, individual_preds = ensemble_predict(features_scaled)
        
        # Format individual predictions
        individual_votes = {}
        for model_name, pred in individual_preds.items():
            individual_votes[model_name] = int(pred[0])
        
        return jsonify({
            'prediction': int(ensemble_pred[0]),  # 1 = home wins, 0 = away wins
            'home_win_probability': float(ensemble_prob[0]),
            'confidence': float(max(ensemble_prob[0], 1 - ensemble_prob[0])),
            'individual_votes': individual_votes
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_game', methods=['POST'])
def predict_game():
    """Predict game outcome from team names"""
    try:
        data = request.get_json()
        
        if 'home_team' not in data or 'away_team' not in data:
            return jsonify({'error': 'Missing home_team or away_team'}), 400
        
        home_team = data['home_team']
        away_team = data['away_team']
        
        # Get current team stats
        try:
            # Get 2024 data as baseline
            stats_2024 = nfl.import_schedules([2024])
            team_averages_2024 = prepare_team_stats(stats_2024)
            
            # Get 2025 data for current form
            stats_2025 = nfl.import_schedules([2025])
            completed_2025 = stats_2025.dropna(subset=['home_score', 'away_score'])
            
            if len(completed_2025) > 0:
                team_averages_2025 = prepare_team_stats(completed_2025)
                team_averages = combine_weighted_stats(team_averages_2024, team_averages_2025)
            else:
                team_averages = team_averages_2024
            
        except Exception as e:
            return jsonify({'error': f'Could not load team data: {str(e)}'}), 500
        
        # Get team stats
        home_stats_df = team_averages[team_averages['team'] == home_team]
        away_stats_df = team_averages[team_averages['team'] == away_team]
        
        if len(home_stats_df) == 0:
            return jsonify({'error': f'Team not found: {home_team}'}), 400
        if len(away_stats_df) == 0:
            return jsonify({'error': f'Team not found: {away_team}'}), 400
        
        home_stats = home_stats_df.iloc[0]
        away_stats = away_stats_df.iloc[0]
        
        # Create matchup features
        matchup = {}
        
        # Add all available stats
        for col in team_averages.columns:
            if col not in ['team', 'season']:
                matchup[f"{col}_home"] = home_stats[col]
                matchup[f"{col}_away"] = away_stats[col]
                matchup[f"{col}_diff"] = home_stats[col] - away_stats[col]
        
        # Prepare features in correct order
        matchup_df = pd.DataFrame([matchup])
        available_features = [col for col in feature_columns if col in matchup_df.columns]
        X = matchup_df[available_features]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        ensemble_pred, ensemble_prob, individual_preds = ensemble_predict(X_scaled)
        
        # Format individual predictions
        individual_votes = {}
        for model_name, pred in individual_preds.items():
            vote = home_team if pred[0] == 1 else away_team
            individual_votes[model_name] = vote
        
        predicted_winner = home_team if ensemble_pred[0] == 1 else away_team
        confidence = ensemble_prob[0] if ensemble_pred[0] == 1 else (1 - ensemble_prob[0])
        
        return jsonify({
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'home_win_probability': float(ensemble_prob[0]),
            'confidence': float(confidence),
            'confidence_percent': f"{confidence * 100:.1f}%",
            'individual_votes': individual_votes,
            'team_stats': {
                'home': {
                    'points_per_game': float(home_stats['points_scored']),
                    'points_allowed_per_game': float(home_stats['points_allowed']),
                    'win_percentage': float(home_stats['won'])
                },
                'away': {
                    'points_per_game': float(away_stats['points_scored']),
                    'points_allowed_per_game': float(away_stats['points_allowed']),
                    'win_percentage': float(away_stats['won'])
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/teams', methods=['GET'])
def get_teams():
    """Get list of available teams"""
    try:
        # Get current team data
        stats_2025 = nfl.import_schedules([2025])
        stats_2024 = nfl.import_schedules([2024])
        
        # Get all unique teams
        teams_2025 = set(stats_2025['home_team'].dropna()) | set(stats_2025['away_team'].dropna())
        teams_2024 = set(stats_2024['home_team'].dropna()) | set(stats_2024['away_team'].dropna())
        all_teams = sorted(teams_2025 | teams_2024)
        
        return jsonify({
            'teams': all_teams,
            'count': len(all_teams)
        }) 
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'models': list(models.keys()),
        'features_count': len(feature_columns),
        'scaler_loaded': scaler is not None,
        'api_version': '1.0'
    })

if __name__ == '__main__':
    # Load models on startup
    if load_artifacts():
        print("🚀 Starting NFL Predictor API...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load models. Run train_and_save_model.py first!")
