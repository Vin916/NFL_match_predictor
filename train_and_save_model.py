#!/usr/bin/env python3
"""
Train NFL predictor models and save them for Flask API deployment
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class NFLModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_nfl_data(self, years=None):
        """Load NFL data from nfl_data_py"""
        if years is None:
            years = [2022, 2023, 2024]  # Historical data for training
        
        print("Loading NFL data...")
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
    
    def create_matchup_features(self, schedules):
        """Create features for each game matchup"""
        print("Creating matchup features...")
        
        # Prepare team stats
        team_averages = self.prepare_team_stats(schedules)
        
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
    
    def save_models(self):
        """Save trained models and preprocessing artifacts"""
        print("\nSaving models and preprocessing artifacts...")
        
        # Save individual models
        for name, model in self.models.items():
            filename = f"model_{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, "scaler.pkl")
        print("Saved scaler.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_columns, "feature_columns.pkl")
        print("Saved feature_columns.pkl")
        
        print("\n✅ All models and artifacts saved successfully!")

def main():
    print("🏈 NFL Model Training and Packaging")
    print("=" * 50)
    
    # Initialize trainer
    trainer = NFLModelTrainer()
    
    # Load and prepare data
    schedules = trainer.load_nfl_data()
    
    # Create training dataset
    matchups = trainer.create_matchup_features(schedules)
    
    print(f"Total games in dataset: {len(matchups)}")
    
    # Prepare features and target
    X = matchups[trainer.feature_columns]
    y = matchups['home_wins']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    # Train models
    trainer.train_models(X_train_scaled, y_train)
    
    # Evaluate models
    model_accuracies = trainer.evaluate_models(X_test_scaled, y_test)
    
    # Save everything
    trainer.save_models()
    
    print(f"\n🚀 Ready for Flask API deployment!")
    print(f"   Run: python app.py")

if __name__ == "__main__":
    main()
