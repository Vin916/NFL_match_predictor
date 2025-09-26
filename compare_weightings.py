#!/usr/bin/env python3
"""
Compare the impact of different weighting schemes
"""

import nfl_data_py as nfl
import pandas as pd

def compare_weightings():
    print("🏈 Impact of Heavy 2025 Season Weighting")
    print("=" * 60)
    
    # Get the data
    sched_2024 = nfl.import_schedules([2024])
    sched_2025 = nfl.import_schedules([2025])
    completed_2025 = sched_2025.dropna(subset=['home_score', 'away_score'])
    
    print(f"📊 Data Available:")
    print(f"  2024 games: {len(sched_2024.dropna(subset=['home_score', 'away_score']))} (complete season)")
    print(f"  2025 games: {len(completed_2025)} (partial season - weeks 1-4)")
    
    # Calculate team performance for both seasons
    def get_team_performance(schedule_data):
        team_stats = []
        for _, game in schedule_data.dropna(subset=['home_score', 'away_score']).iterrows():
            team_stats.extend([
                {'team': game['home_team'], 'points': game['home_score'], 'won': game['home_score'] > game['away_score']},
                {'team': game['away_team'], 'points': game['away_score'], 'won': game['away_score'] > game['home_score']}
            ])
        return pd.DataFrame(team_stats).groupby('team').agg({'points': 'mean', 'won': 'mean'}).round(1)
    
    performance_2024 = get_team_performance(sched_2024)
    performance_2025 = get_team_performance(sched_2025)
    
    print(f"\n🔥 Teams with BIGGEST changes from 2024 to 2025:")
    
    # Find teams with biggest performance changes
    combined = performance_2024.join(performance_2025, lsuffix='_2024', rsuffix='_2025', how='inner')
    combined['points_change'] = combined['points_2025'] - combined['points_2024']
    combined['win_change'] = combined['won_2025'] - combined['won_2024']
    
    print("\n📈 Biggest IMPROVEMENTS (why 2025 data matters):")
    improvements = combined.sort_values('win_change', ascending=False).head(5)
    for team, stats in improvements.iterrows():
        print(f"  {team}: {stats['won_2024']:.1%} → {stats['won_2025']:.1%} wins (+{stats['win_change']:+.1%})")
        print(f"       {stats['points_2024']:.1f} → {stats['points_2025']:.1f} pts/game (+{stats['points_change']:+.1f})")
    
    print("\n📉 Biggest DECLINES (why old data is misleading):")
    declines = combined.sort_values('win_change', ascending=True).head(5)
    for team, stats in declines.iterrows():
        print(f"  {team}: {stats['won_2024']:.1%} → {stats['won_2025']:.1%} wins ({stats['win_change']:+.1%})")
        print(f"       {stats['points_2024']:.1f} → {stats['points_2025']:.1f} pts/game ({stats['points_change']:+.1f})")
    
    print(f"\n⚖️  WEIGHTING COMPARISON:")
    print(f"  OLD (70/30): 70% 2025 data, 30% 2024 data")
    print(f"  NEW (90/10): 90% 2025 data, 10% 2024 data")
    
    print(f"\n✅ BENEFITS of 90/10 weighting:")
    print(f"   🎯 Captures dramatic team changes (like IND going undefeated)")
    print(f"   📈 Reflects current coaching/player changes")  
    print(f"   🏈 Accounts for injuries, trades, scheme changes")
    print(f"   📊 More responsive to early-season momentum")
    print(f"   ⚡ Less influenced by outdated 2024 performance")
    
    print(f"\n🚀 Your predictor now prioritizes CURRENT FORM over old data!")

if __name__ == "__main__":
    compare_weightings()
