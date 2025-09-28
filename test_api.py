#!/usr/bin/env python3
"""
Test script for NFL Predictor API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8080"

def test_health(): 
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_teams():
    """Test teams endpoint"""
    print("\n🏈 Testing teams endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/teams")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Found {data['count']} teams")
        print(f"Sample teams: {data['teams'][:5]}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_game():
    """Test game prediction endpoint"""
    print("\n🎯 Testing game prediction...")
    try:
        # Test with popular teams
        payload = {
            "home_team": "KC",
            "away_team": "BUF"
        }
        
        response = requests.post(
            f"{BASE_URL}/predict_game",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Game: {data['away_team']} @ {data['home_team']}")
            print(f"Predicted Winner: {data['predicted_winner']}")
            print(f"Confidence: {data['confidence_percent']}")
            print(f"Individual Votes: {data['individual_votes']}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_features():
    """Test raw features prediction endpoint"""
    print("\n🔢 Testing features prediction...")
    try:
        # Sample features (12 features based on the model)
        # [points_scored_home, points_scored_away, points_scored_diff,
        #  points_allowed_home, points_allowed_away, points_allowed_diff,
        #  home_game_home, home_game_away, home_game_diff,
        #  won_home, won_away, won_diff]
        payload = {
            "features": [25.5, 22.1, 3.4, 18.2, 21.5, -3.3, 0.5, 0.5, 0.0, 0.65, 0.45, 0.2]
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {'Home wins' if data['prediction'] == 1 else 'Away wins'}")
            print(f"Home win probability: {data['home_win_probability']:.3f}")
            print(f"Confidence: {data['confidence']:.3f}")
            print(f"Individual votes: {data['individual_votes']}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🏈 NFL Predictor API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Teams List", test_teams),
        ("Game Prediction", test_predict_game),
        ("Features Prediction", test_predict_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API setup.")

if __name__ == "__main__":
    main()
