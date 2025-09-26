#!/usr/bin/env python3
"""
NFL Weekly Predictor - Predict ALL games for a specific week
Usage: python3 nfl_weekly_predictor.py [week_number]
Example: python3 nfl_weekly_predictor.py 4
"""

import sys
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class NFLWeeklyPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_nfl_data(self, years=None):
        """Load NFL data from nfl_data_py"""
        if years is None:
            years = [2022, 2023, 2024]  # Historical data for training (exclude 2025)
        
        print("Loading NFL data...")
        
        # Load historical data for training (don't include 2025 to avoid data leakage)
        print("Loading historical data for training...")
        schedules = nfl.import_schedules(years)
        
        print(f"Loaded training data for years: {years}")
        return schedules
    
    def prepare_team_stats(self, schedules):
        """Prepare team statistics from schedule data"""
        print("Preparing team statistics from schedule data...")
        
        # Calculate team stats from completed games
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
    
    def combine_weighted_stats(self, stats_2024, stats_2025):
        """Combine 2024 and 2025 stats with weighted averages"""
        print("Combining 2024 baseline with 2025 current form...")
        
        # Get all teams from both seasons
        all_teams = set(stats_2024['team'].unique()) | set(stats_2025['team'].unique())
        
        combined_stats = []
        
        for team in all_teams:
            # Get team data from both seasons
            team_2024 = stats_2024[stats_2024['team'] == team]
            team_2025 = stats_2025[stats_2025['team'] == team]
            
            if len(team_2024) > 0 and len(team_2025) > 0:
                # Both seasons available - use weighted average
                stats_2024_row = team_2024.iloc[0]
                stats_2025_row = team_2025.iloc[0]
                
                combined_row = {
                    'team': team,
                    'season': 2025,  # Current season
                }
                
                # Weight the numerical stats (90% current season, 10% last season)
                for col in ['points_scored', 'points_allowed', 'home_game', 'won']:
                    if col in stats_2024_row and col in stats_2025_row:
                        weighted_value = (stats_2024_row[col] * 0.1) + (stats_2025_row[col] * 0.9)
                        combined_row[col] = weighted_value
                
                combined_stats.append(combined_row)
                
            elif len(team_2025) > 0:
                # Only 2025 data available (new team or renamed)
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
                # Only 2024 data available (team didn't play in 2025 yet)
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
    
    def create_matchup_features(self, schedules):
        """Create features for each game matchup"""
        print("Creating matchup features...")
        
        # Prepare team stats
        team_averages = self.prepare_team_stats(schedules)
        
        print("Team averages columns:", team_averages.columns.tolist())
        
        # Merge home team stats
        matchups = schedules.merge(
            team_averages, 
            left_on=['home_team', 'season'], 
            right_on=['team', 'season'],
            suffixes=('', '_home')
        )
        
        # Merge away team stats
        matchups = matchups.merge(
            team_averages,
            left_on=['away_team', 'season'],
            right_on=['team', 'season'],
            suffixes=('_home', '_away')
        )
        
        # Create feature columns
        available_stats = [col for col in team_averages.columns if col not in ['team', 'season']]
        
        self.feature_columns = []
        
        # Add home and away stats plus differentials
        for stat in available_stats:
            home_col = f"{stat}_home"
            away_col = f"{stat}_away"
            if home_col in matchups.columns and away_col in matchups.columns:
                self.feature_columns.extend([home_col, away_col])
                
                # Create differential features
                diff_col = f"{stat}_diff"
                matchups[diff_col] = matchups[home_col] - matchups[away_col]
                self.feature_columns.append(diff_col)
        
        print(f"Feature columns: {self.feature_columns}")
        
        # Create target variable (1 if home team wins, 0 if away team wins)
        matchups['home_wins'] = (matchups['home_score'] > matchups['away_score']).astype(int)
        
        # Remove games without results
        matchups = matchups.dropna(subset=['home_score', 'away_score'])
        
        return matchups
    
    def train_models(self, X_train, y_train):
        """Train ensemble models"""
        print("Training ensemble models...")
        
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate individual model performance"""
        print("\nModel Performance:")
        print("=" * 50)
        
        model_accuracies = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_accuracies[name] = accuracy
            print(f"{name}: {accuracy:.3f}")
        
        return model_accuracies
    
    def ensemble_predict(self, X):
        """Make ensemble predictions using voting"""
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
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
    
    def predict_week_games(self, week_number):
        """Predict all games for a specific week"""
        print(f"Getting all games for Week {week_number} of 2025 season...")
        
        try:
            # Get 2025 schedule
            current_schedules = nfl.import_schedules([2025])
            
            # Get all games for the specified week
            week_games = current_schedules[current_schedules['week'] == week_number]
            
            if len(week_games) == 0:
                print(f"❌ No games found for Week {week_number}")
                return None, None, None, None
            
            print(f"Found {len(week_games)} games for Week {week_number}")
            
            # Get team stats including 2025 games played so far
            print("Loading team stats including 2025 season games played so far...")
            
            # Get 2024 data as baseline
            stats_2024 = nfl.import_schedules([2024])
            
            # Get 2025 data and filter to completed games only
            stats_2025 = nfl.import_schedules([2025])
            completed_2025 = stats_2025.dropna(subset=['home_score', 'away_score'])
            
            print(f"Using {len(completed_2025)} completed 2025 games for current team form")
            
            # Combine 2024 and completed 2025 games, but weight 2025 MUCH more heavily
            if len(completed_2025) > 0:
                # Calculate 2024 season averages
                team_averages_2024 = self.prepare_team_stats(stats_2024)
                team_averages_2024['weight'] = 0.1  # Only 10% weight for 2024 data (mostly for fallback)
                
                # Calculate 2025 season averages so far
                team_averages_2025 = self.prepare_team_stats(completed_2025)
                team_averages_2025['weight'] = 0.9  # 90% weight for current 2025 season!
                
                # Combine with weighted averages
                team_averages = self.combine_weighted_stats(team_averages_2024, team_averages_2025)
            else:
                print("No 2025 games completed yet, using 2024 data only")
                team_averages = self.prepare_team_stats(stats_2024)
            
            # Create matchup features for week games
            sample_matchups = []
            
            for _, game in week_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Get team stats
                home_stats_df = team_averages[team_averages['team'] == home_team]
                away_stats_df = team_averages[team_averages['team'] == away_team]
                
                if len(home_stats_df) > 0 and len(away_stats_df) > 0:
                    home_stats = home_stats_df.iloc[0]
                    away_stats = away_stats_df.iloc[0]
                    
                    matchup = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'week': game['week'],
                        'gameday': game.get('gameday', 'TBD'),
                        'gametime': game.get('gametime', 'TBD'),
                        'actual_home_score': game.get('home_score', None),
                        'actual_away_score': game.get('away_score', None)
                    }
                    
                    # Add all available stats
                    for col in team_averages.columns:
                        if col not in ['team', 'season']:
                            matchup[f"{col}_home"] = home_stats[col]
                            matchup[f"{col}_away"] = away_stats[col]
                            matchup[f"{col}_diff"] = home_stats[col] - away_stats[col]
                    
                    sample_matchups.append(matchup)
            
            if not sample_matchups:
                print("❌ No valid matchups could be created")
                return None, None, None, None
                
            upcoming_games = pd.DataFrame(sample_matchups)
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in upcoming_games.columns]
            X_upcoming = upcoming_games[available_features]
            X_upcoming_scaled = self.scaler.transform(X_upcoming)
            
            # Make predictions
            ensemble_pred, ensemble_prob, individual_preds = self.ensemble_predict(X_upcoming_scaled)
            
            return upcoming_games, ensemble_pred, ensemble_prob, individual_preds
            
        except Exception as e:
            print(f"❌ Error loading Week {week_number} data: {e}")
            return None, None, None, None
    
    def display_week_predictions(self, games, ensemble_pred, ensemble_prob, individual_preds, week_number):
        """Display prediction results for a week"""
        print("\n" + "="*80)
        print(f"🏈 NFL WEEK {week_number} PREDICTIONS - 2025 SEASON")
        print("="*80)
        
        correct_predictions = 0
        total_with_results = 0
        
        for i, (_, game) in enumerate(games.iterrows()):
            home_team = game['home_team']
            away_team = game['away_team']
            
            predicted_winner = home_team if ensemble_pred[i] == 1 else away_team
            confidence = ensemble_prob[i] if ensemble_pred[i] == 1 else (1 - ensemble_prob[i])
            confidence_pct = confidence * 100
            
            # Game info
            game_info = f"Game {i+1}: {away_team} @ {home_team}"
            if pd.notna(game['gameday']):
                game_info += f" - {game['gameday']}"
            if pd.notna(game['gametime']):
                game_info += f" {game['gametime']}"
                
            print(f"\n{game_info}")
            print(f"🎯 Predicted Winner: {predicted_winner}")
            print(f"📊 Confidence: {confidence_pct:.1f}%")
            
            # Show actual results if available (for completed games)
            if pd.notna(game['actual_home_score']) and pd.notna(game['actual_away_score']):
                actual_winner = home_team if game['actual_home_score'] > game['actual_away_score'] else away_team
                print(f"🏆 Actual Result: {home_team} {int(game['actual_home_score'])}-{int(game['actual_away_score'])} {away_team}")
                print(f"🏆 Actual Winner: {actual_winner}")
                
                if predicted_winner == actual_winner:
                    print("✅ CORRECT PREDICTION!")
                    correct_predictions += 1
                else:
                    print("❌ Incorrect prediction")
                total_with_results += 1
            else:
                print("⏳ Game not yet played")
            
            print("🗳️  Individual Model Votes:")
            for model_name, pred in individual_preds.items():
                vote = home_team if pred[i] == 1 else away_team
                print(f"   {model_name}: {vote}")
            print("-" * 50)
        
        # Show accuracy if we have actual results
        if total_with_results > 0:
            accuracy = (correct_predictions / total_with_results) * 100
            print(f"\n🎯 Week {week_number} Prediction Accuracy: {correct_predictions}/{total_with_results} ({accuracy:.1f}%)")

def main():
    # Get week number from command line argument
    week_number = 1  # Default to week 1 for 2025 season
    if len(sys.argv) > 1:
        try:
            week_number = int(sys.argv[1])
        except ValueError:
            print("Usage: python3 nfl_weekly_predictor.py [week_number]")
            print("Example: python3 nfl_weekly_predictor.py 1")
            return
    
    print("🏈 NFL Weekly Predictor - Ensemble Machine Learning")
    print("=" * 60)
    print(f"Predicting ALL games for Week {week_number} of 2025 NFL Season")
    
    # Initialize predictor
    predictor = NFLWeeklyPredictor()
    
    # Load and prepare training data (historical)
    schedules = predictor.load_nfl_data()
    
    # Create training dataset
    matchups = predictor.create_matchup_features(schedules)
    
    print(f"Total training games: {len(matchups)}")
    
    # Prepare features and target
    X = matchups[predictor.feature_columns]
    y = matchups['home_wins']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Train models
    predictor.train_models(X_train_scaled, y_train)
    
    # Evaluate models
    model_accuracies = predictor.evaluate_models(X_test_scaled, y_test)
    
    # Test ensemble performance
    ensemble_pred, ensemble_prob, _ = predictor.ensemble_predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.3f}")
    
    # Predict all games for the specified week
    upcoming_games, pred, prob, individual_preds = predictor.predict_week_games(week_number)
    
    if upcoming_games is not None:
        # Display results
        predictor.display_week_predictions(upcoming_games, pred, prob, individual_preds, week_number)
    else:
        print(f"❌ Could not generate predictions for Week {week_number}")

if __name__ == "__main__":
    main()
