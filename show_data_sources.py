#!/usr/bin/env python3
"""
Show what data sources are being used for predictions
"""

import nfl_data_py as nfl
import pandas as pd

def show_data_sources():
    print("🏈 NFL Predictor Data Sources")
    print("=" * 50)
    
    # Training data
    print("📚 TRAINING DATA (for model learning):")
    training_years = [2022, 2023, 2024]
    total_training = 0
    for year in training_years:
        sched = nfl.import_schedules([year])
        completed = sched.dropna(subset=['home_score', 'away_score'])
        total_training += len(completed)
        print(f"  {year}: {len(completed)} games")
    print(f"  Total: {total_training} games")
    
    print("\n🎯 PREDICTION DATA (for team stats):")
    
    # 2024 baseline data
    sched_2024 = nfl.import_schedules([2024])
    completed_2024 = sched_2024.dropna(subset=['home_score', 'away_score'])
    print(f"  2024 baseline: {len(completed_2024)} games (10% weight - fallback only)")
    
    # 2025 current season data
    sched_2025 = nfl.import_schedules([2025])
    completed_2025 = sched_2025.dropna(subset=['home_score', 'away_score'])
    print(f"  2025 current: {len(completed_2025)} games (90% weight - PRIMARY)")
    
    if len(completed_2025) > 0:
        weeks_played = sorted(completed_2025['week'].unique())
        print(f"  2025 weeks completed: {weeks_played}")
        
        print(f"\n📊 Sample 2025 team performance so far:")
        # Show a few team examples
        team_stats = []
        for _, game in completed_2025.iterrows():
            team_stats.append({
                'team': game['home_team'],
                'points_scored': game['home_score'],
                'points_allowed': game['away_score'],
                'won': 1 if game['home_score'] > game['away_score'] else 0
            })
            team_stats.append({
                'team': game['away_team'], 
                'points_scored': game['away_score'],
                'points_allowed': game['home_score'],
                'won': 1 if game['away_score'] > game['home_score'] else 0
            })
        
        team_df = pd.DataFrame(team_stats)
        team_summary = team_df.groupby('team').agg({
            'points_scored': 'mean',
            'points_allowed': 'mean',
            'won': 'mean'
        }).round(1)
        
        print("  Top 5 teams by win rate in 2025:")
        top_teams = team_summary.sort_values('won', ascending=False).head(5)
        for team, stats in top_teams.iterrows():
            print(f"    {team}: {stats['won']:.1%} wins, {stats['points_scored']:.1f} pts/game")
    
    print(f"\n✅ RESULT: Predictions HEAVILY favor current 2025 team form!")
    print(f"   - 90% weight on 2025 season (current performance)")
    print(f"   - 10% weight on 2024 season (fallback only)")
    print(f"   - Prioritizes recent form over outdated historical data")
    print(f"   - Updates automatically as more 2025 games are played")

if __name__ == "__main__":
    show_data_sources()
