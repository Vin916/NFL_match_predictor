#!/usr/bin/env python3
"""
Show upcoming NFL weeks with games that haven't been played yet
"""

import nfl_data_py as nfl
import pandas as pd

def show_upcoming_weeks():
    print("🏈 2025 NFL Season - Upcoming Games")
    print("=" * 50)
    
    try:
        # Get 2025 schedule
        sched = nfl.import_schedules([2025])
        
        # Find games without results (upcoming games)
        upcoming = sched[pd.isna(sched['home_score']) | pd.isna(sched['away_score'])]
        
        if len(upcoming) == 0:
            print("❌ No upcoming games found - season may be complete")
            return
        
        print(f"📊 Total upcoming games: {len(upcoming)}")
        
        # Group by week
        upcoming_weeks = sorted(upcoming['week'].dropna().unique())
        print(f"📅 Upcoming weeks: {upcoming_weeks}")
        
        print("\n🎯 Games per upcoming week:")
        for week in upcoming_weeks[:5]:  # Show first 5 upcoming weeks
            week_games = upcoming[upcoming['week'] == week]
            print(f"  Week {int(week)}: {len(week_games)} games")
        
        print(f"\n🔥 Next upcoming games (Week {int(upcoming_weeks[0])}):")
        next_week_games = upcoming[upcoming['week'] == upcoming_weeks[0]]
        for _, game in next_week_games.head(5).iterrows():
            print(f"  {game['away_team']} @ {game['home_team']} - {game['gameday']}")
        
        print(f"\n💡 To predict all games for a week, run:")
        print(f"   python3 nfl_weekly_predictor.py {int(upcoming_weeks[0])}")
        
    except Exception as e:
        print(f"❌ Error loading schedule: {e}")

if __name__ == "__main__":
    show_upcoming_weeks()
