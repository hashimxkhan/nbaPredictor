import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pickle
import os
from datetime import datetime

def fetch_nba_data():
    """Fetch NBA game data for all teams"""
    nba_teams = teams.get_teams()
    team_abbr_to_id = {team['abbreviation']: team['id'] for team in nba_teams}
    all_games = pd.DataFrame()
    
    print("Fetching NBA game data...")
    for team in nba_teams:
        team_id = team['id']
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
            games = gamefinder.get_data_frames()[0]
            all_games = pd.concat([all_games, games], ignore_index=True)
        except Exception as e:
            print(f"Error fetching data for team {team['full_name']}: {e}")
    
    return all_games, team_abbr_to_id

def process_data(all_games, team_abbr_to_id):
    """Process and prepare the data for training"""
    # Convert date and create win indicator
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
    all_games['WIN'] = all_games['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # Calculate points per game (rolling 10-game average)
    all_games['Points_Per_Game'] = all_games.groupby('TEAM_ID')['PTS'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    
    # Calculate points allowed (rolling 10-game average)
    all_games['Points_Allowed'] = all_games.groupby('TEAM_ID')['PTS'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    
    # Create home game indicator
    all_games['HOME_GAME'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # Calculate win streak (last 5 games)
    all_games['Win_Streak'] = all_games.groupby('TEAM_ID')['WIN'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    
    # Add last game result
    all_games['LAST_GAME_RESULT'] = all_games.groupby('TEAM_ID')['WIN'].shift(1).fillna(0)
    
    # Calculate home/away record (rolling 10-game)
    home_games = all_games[all_games['HOME_GAME'] == 1]
    away_games = all_games[all_games['HOME_GAME'] == 0]
    
    home_games['Home_Record'] = home_games.groupby('TEAM_ID')['WIN'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    away_games['Away_Record'] = away_games.groupby('TEAM_ID')['WIN'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    
    all_games['Home_Record'] = home_games['Home_Record']
    all_games['Away_Record'] = away_games['Away_Record']
    
    # Fill NaN values with 0.5 (neutral starting point)
    all_games['Home_Record'] = all_games['Home_Record'].fillna(0.5)
    all_games['Away_Record'] = all_games['Away_Record'].fillna(0.5)
    all_games['Points_Allowed'] = all_games['Points_Allowed'].fillna(all_games['PTS'].mean())
    
    # Extract opponent team ID
    def get_opponent_team_id(matchup, team_id):
        if '@' in matchup:
            opponent_abbr = matchup.split(' @ ')[-1]
        else:
            opponent_abbr = matchup.split(' vs. ')[-1]
        return team_abbr_to_id.get(opponent_abbr, team_id)
    
    all_games['OPPONENT_TEAM_ID'] = all_games.apply(
        lambda row: get_opponent_team_id(row['MATCHUP'], row['TEAM_ID']), 
        axis=1
    )
    
    return all_games

def train_model(all_games):
    """Train the prediction model"""
    # Prepare features
    features = [
        'TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'Points_Allowed',
        'Win_Streak', 'HOME_GAME', 'LAST_GAME_RESULT', 'Home_Record', 'Away_Record'
    ]
    
    X = all_games[features]
    y = all_games['WIN']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with more estimators and balanced class weights
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, feature_importance

def save_model(model, team_abbr_to_id):
    """Save the trained model and team mapping"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_filename = f"models/nba_predictor_{datetime.now().strftime('%Y%m%d')}.pkl"
    mapping_filename = f"models/team_mapping_{datetime.now().strftime('%Y%m%d')}.pkl"
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    with open(mapping_filename, 'wb') as f:
        pickle.dump(team_abbr_to_id, f)
    
    print(f"\nModel saved to {model_filename}")
    print(f"Team mapping saved to {mapping_filename}")

def main():
    # Fetch and process data
    all_games, team_abbr_to_id = fetch_nba_data()
    processed_games = process_data(all_games, team_abbr_to_id)
    
    # Train and save model
    model, feature_importance = train_model(processed_games)
    save_model(model, team_abbr_to_id)

if __name__ == "__main__":
    main() 