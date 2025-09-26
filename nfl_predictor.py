"""
NFL Match Predictor using Ensemble Machine Learning Models
Combines Logistic Regression, XGBoost, Decision Trees, and Random Forest
to predict NFL game outcomes with confidence levels.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class NFLPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.team_stats = None
        
    def load_nfl_data(self, years=None, include_current_season=True):
        """Load NFL data from nfl_data_py"""
        if years is None:
            years = [2022, 2023]  # Recent historical data for training
        
        print("Loading NFL data...")
        
        # Load historical data for training
        print("Loading historical data for training...")
        pbp_data = nfl.import_pbp_data(years)
        schedules = nfl.import_schedules(years)
        
        # Also load current 2024 season data
        if include_current_season:
            print("Loading current 2024 season data...")
            try:
                current_pbp = nfl.import_pbp_data([2024])
                current_schedules = nfl.import_schedules([2024])
                
                # Combine historical and current data
                pbp_data = pd.concat([pbp_data, current_pbp], ignore_index=True)
                schedules = pd.concat([schedules, current_schedules], ignore_index=True)
                
                years.append(2024)
                print("✅ Successfully loaded 2024 season data!")
            except Exception as e:
                print(f"⚠️  Could not load 2024 data: {e}")
                print("Using historical data only for training")
        
        print(f"Loaded data for years: {years}")
        return pbp_data, schedules
    
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
                    'home_game': 1
                })
                
                # Away team stats  
                team_stats.append({
                    'team': game['away_team'],
                    'season': game['season'],
                    'game_id': game['game_id'],
                    'points_scored': game['away_score'],
                    'points_allowed': game['home_score'],
                    'home_game': 0
                })
        
        team_stats_df = pd.DataFrame(team_stats)
        
        # Calculate averages per team per season
        team_averages = team_stats_df.groupby(['team', 'season']).agg({
            'points_scored': 'mean',
            'points_allowed': 'mean',
            'home_game': 'mean'  # Home field advantage
        }).reset_index()
        
        return team_averages
    
    def create_matchup_features(self, schedules, pbp_data):
        """Create features for each game matchup"""
        print("Creating matchup features...")
        
        # First prepare basic team stats from schedules
        team_averages = self.prepare_team_stats(schedules)
        
        # Now enhance with play-by-play derived stats
        team_averages = self.enhance_with_pbp_stats(team_averages, pbp_data)
        
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
        
        # Dynamically create feature columns based on available data
        available_stats = [col for col in team_averages.columns if col not in ['team', 'season']]
        
        self.feature_columns = []
        
        # Add home and away stats
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
        
    def enhance_with_pbp_stats(self, team_averages, pbp_data):
        """Add play-by-play derived statistics to team averages"""
        print("Enhancing with play-by-play statistics...")
        
        # Calculate team stats from play-by-play data
        pbp_stats = []
        
        for team in team_averages['team'].unique():
            for season in team_averages['season'].unique():
                # Filter plays for this team and season
                team_plays = pbp_data[
                    (pbp_data['season'] == season) & 
                    ((pbp_data['posteam'] == team) | (pbp_data['defteam'] == team))
                ]
                
                if len(team_plays) > 0:
                    # Offensive stats (when team has possession)
                    off_plays = team_plays[team_plays['posteam'] == team]
                    
                    passing_yards = off_plays['passing_yards'].sum() if 'passing_yards' in off_plays.columns else 0
                    rushing_yards = off_plays['rushing_yards'].sum() if 'rushing_yards' in off_plays.columns else 0
                    turnovers = off_plays['interception'].sum() + off_plays['fumble_lost'].sum() if 'interception' in off_plays.columns and 'fumble_lost' in off_plays.columns else 0
                    
                    # Defensive stats (when opponent has possession)
                    def_plays = team_plays[team_plays['defteam'] == team]
                    opp_passing_yards = def_plays['passing_yards'].sum() if 'passing_yards' in def_plays.columns else 0
                    opp_rushing_yards = def_plays['rushing_yards'].sum() if 'rushing_yards' in def_plays.columns else 0
                    
                    # Count games played
                    games_played = len(team_plays['game_id'].unique())
                    
                    if games_played > 0:
                        pbp_stats.append({
                            'team': team,
                            'season': season,
                            'passing_yards_per_game': passing_yards / games_played,
                            'rushing_yards_per_game': rushing_yards / games_played,
                            'turnovers_per_game': turnovers / games_played,
                            'opp_passing_yards_per_game': opp_passing_yards / games_played,
                            'opp_rushing_yards_per_game': opp_rushing_yards / games_played
                        })
        
        if pbp_stats:
            pbp_df = pd.DataFrame(pbp_stats)
            # Merge with existing team averages
            team_averages = team_averages.merge(pbp_df, on=['team', 'season'], how='left')
        
        return team_averages
    
    def train_models(self, X_train, y_train):
        """Train all ensemble models"""
        print("Training ensemble models...")
        
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
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
    
    def predict_games(self, upcoming_games=None):
        """Predict upcoming NFL games"""
        if upcoming_games is None:
            # Use 2024 season data for predictions
            print("Creating predictions using 2024 season data...")
            
            try:
                # Get 2024 schedules and pbp data
                current_schedules = nfl.import_schedules([2024])
                current_pbp = nfl.import_pbp_data([2024])
                
                # Prepare current team stats the same way as training
                team_averages = self.prepare_team_stats(current_schedules)
                team_averages = self.enhance_with_pbp_stats(team_averages, current_pbp)
                
                print(f"Found {len(team_averages)} teams in 2024 data")
                
                # Get upcoming games from 2024 schedule (games without results)
                upcoming_schedule = current_schedules[
                    pd.isna(current_schedules['home_score']) | 
                    pd.isna(current_schedules['away_score'])
                ].head(4)  # Take first 4 upcoming games
                
                if len(upcoming_schedule) == 0:
                    print("No upcoming games found, creating sample matchups...")
                    # Create sample matchups if no upcoming games
                    teams = team_averages['team'].unique()[:8]
                    sample_matchups = []
                    
                    for i in range(0, min(len(teams), 8), 2):
                        if i+1 < len(teams):
                            home_team = teams[i]
                            away_team = teams[i+1]
                            
                            # Get team stats
                            home_stats = team_averages[team_averages['team'] == home_team].iloc[0]
                            away_stats = team_averages[team_averages['team'] == away_team].iloc[0]
                            
                            matchup = {
                                'home_team': home_team,
                                'away_team': away_team,
                            }
                            
                            # Add all available stats
                            for col in team_averages.columns:
                                if col not in ['team', 'season']:
                                    matchup[f"{col}_home"] = home_stats[col]
                                    matchup[f"{col}_away"] = away_stats[col]
                                    matchup[f"{col}_diff"] = home_stats[col] - away_stats[col]
                            
                            sample_matchups.append(matchup)
                    
                    upcoming_games = pd.DataFrame(sample_matchups)
                else:
                    print(f"Found {len(upcoming_schedule)} upcoming games in 2024 season")
                    # Process actual upcoming games
                    sample_matchups = []
                    
                    for _, game in upcoming_schedule.iterrows():
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
                                'week': game.get('week', 'TBD'),
                                'gameday': game.get('gameday', 'TBD')
                            }
                            
                            # Add all available stats
                            for col in team_averages.columns:
                                if col not in ['team', 'season']:
                                    matchup[f"{col}_home"] = home_stats[col]
                                    matchup[f"{col}_away"] = away_stats[col]
                                    matchup[f"{col}_diff"] = home_stats[col] - away_stats[col]
                            
                            sample_matchups.append(matchup)
                    
                    upcoming_games = pd.DataFrame(sample_matchups)
                
            except Exception as e:
                print(f"⚠️  Error loading 2024 data: {e}")
                print("Falling back to sample predictions with 2023 data...")
                
                # Fallback to 2023 data
                latest_schedules = nfl.import_schedules([2023])
                latest_pbp = nfl.import_pbp_data([2023])
                
                team_averages = self.prepare_team_stats(latest_schedules)
                team_averages = self.enhance_with_pbp_stats(team_averages, latest_pbp)
                
                teams = team_averages['team'].unique()[:8]
                sample_matchups = []
                
                for i in range(0, len(teams), 2):
                    if i+1 < len(teams):
                        home_team = teams[i]
                        away_team = teams[i+1]
                        
                        home_stats = team_averages[team_averages['team'] == home_team].iloc[0]
                        away_stats = team_averages[team_averages['team'] == away_team].iloc[0]
                        
                        matchup = {
                            'home_team': home_team,
                            'away_team': away_team,
                        }
                        
                        for col in team_averages.columns:
                            if col not in ['team', 'season']:
                                matchup[f"{col}_home"] = home_stats[col]
                                matchup[f"{col}_away"] = away_stats[col]
                                matchup[f"{col}_diff"] = home_stats[col] - away_stats[col]
                        
                        sample_matchups.append(matchup)
                
                upcoming_games = pd.DataFrame(sample_matchups)
        
        # Prepare features - only use features that exist in upcoming_games
        available_features = [col for col in self.feature_columns if col in upcoming_games.columns]
        X_upcoming = upcoming_games[available_features]
        X_upcoming_scaled = self.scaler.transform(X_upcoming)
        
        # Make predictions
        ensemble_pred, ensemble_prob, individual_preds = self.ensemble_predict(X_upcoming_scaled)
        
        return upcoming_games, ensemble_pred, ensemble_prob, individual_preds
    
    def display_predictions(self, games, ensemble_pred, ensemble_prob, individual_preds):
        """Display prediction results"""
        print("\n" + "="*80)
        print("🏈 NFL GAME PREDICTIONS - 2024 SEASON")
        print("="*80)
        
        for i, (_, game) in enumerate(games.iterrows()):
            home_team = game['home_team']
            away_team = game['away_team']
            
            predicted_winner = home_team if ensemble_pred[i] == 1 else away_team
            confidence = ensemble_prob[i] if ensemble_pred[i] == 1 else (1 - ensemble_prob[i])
            confidence_pct = confidence * 100
            
            # Add game details if available
            game_info = f"Game {i+1}: {away_team} @ {home_team}"
            if 'week' in game.columns and pd.notna(game['week']):
                game_info += f" (Week {game['week']})"
            if 'gameday' in game.columns and pd.notna(game['gameday']):
                game_info += f" - {game['gameday']}"
                
            print(f"\n{game_info}")
            print(f"🎯 Predicted Winner: {predicted_winner}")
            print(f"📊 Confidence: {confidence_pct:.1f}%")
            
            print("🗳️  Individual Model Votes:")
            for model_name, pred in individual_preds.items():
                vote = home_team if pred[i] == 1 else away_team
                print(f"   {model_name}: {vote}")
            print("-" * 50)

def main():
    print("NFL Match Predictor - Ensemble Machine Learning")
    print("=" * 60)
    
    # Initialize predictor
    predictor = NFLPredictor()
    
    # Load and prepare data
    pbp_data, schedules = predictor.load_nfl_data()
    
    # Create training dataset
    matchups = predictor.create_matchup_features(schedules, pbp_data)
    
    print(f"Total games in dataset: {len(matchups)}")
    
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
    
    # Make predictions on upcoming games
    upcoming_games, pred, prob, individual_preds = predictor.predict_games()
    
    # Display results
    predictor.display_predictions(upcoming_games, pred, prob, individual_preds)

if __name__ == "__main__":
    main()
