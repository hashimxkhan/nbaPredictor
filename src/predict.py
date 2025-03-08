import pickle
import pandas as pd
import os
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import glob

def get_team_mapping():
    """Get mappings between team names and abbreviations"""
    nba_teams = teams.get_teams()
    name_to_abbr = {team['full_name'].upper(): team['abbreviation'] for team in nba_teams}
    # Add last names and common names for easier matching
    last_names = {team['full_name'].split()[-1].upper(): team['full_name'].upper() for team in nba_teams}
    return name_to_abbr, last_names

def find_team_name(input_name, name_to_abbr, last_names):
    """Find the full team name from partial input"""
    input_upper = input_name.upper()
    
    # Direct match with full name
    for full_name in name_to_abbr.keys():
        if input_upper == full_name:
            return full_name
    
    # Match with last name
    if input_upper in last_names:
        return last_names[input_upper]
    
    # Partial match with full name
    matches = [name for name in name_to_abbr.keys() if input_upper in name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Multiple matches found for '{input_name}':\n" + "\n".join(matches))
    
    return None

def load_latest_model():
    """Load the most recent trained model and team mapping"""
    model_files = glob.glob('models/nba_predictor_*.pkl')
    mapping_files = glob.glob('models/team_mapping_*.pkl')
    
    if not model_files or not mapping_files:
        raise FileNotFoundError("No trained model found. Please run train_model.py first.")
    
    latest_model = max(model_files)
    latest_mapping = max(mapping_files)
    
    with open(latest_model, 'rb') as f:
        model = pickle.load(f)
    
    with open(latest_mapping, 'rb') as f:
        team_mapping = pickle.load(f)
    
    return model, team_mapping

def get_team_stats(team_id):
    """Get recent team statistics"""
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    
    # Calculate average points per game from last 10 games
    recent_games = games.head(10)
    points_per_game = recent_games['PTS'].mean()
    
    # Get last game result
    last_game_result = 1 if games.iloc[0]['WL'] == 'W' else 0
    
    return points_per_game, last_game_result

def predict_game(team1_name, team2_name, is_team1_home=True):
    """Predict the outcome of a game between two teams"""
    try:
        # Get team name mappings
        name_to_abbr, last_names = get_team_mapping()
        
        # Find full team names
        team1_full = find_team_name(team1_name, name_to_abbr, last_names)
        team2_full = find_team_name(team2_name, name_to_abbr, last_names)
        
        if not team1_full or not team2_full:
            available_teams = "\n".join(sorted(name_to_abbr.keys()))
            raise ValueError(f"Invalid team name(s). Available teams:\n{available_teams}")
        
        team1_abbr = name_to_abbr[team1_full]
        team2_abbr = name_to_abbr[team2_full]
        
        # Load model and team mapping
        model, team_mapping = load_latest_model()
        
        # Get team IDs
        nba_teams = teams.get_teams()
        team_dict = {team['abbreviation']: team['id'] for team in nba_teams}
        
        team1_id = team_dict[team1_abbr]
        team2_id = team_dict[team2_abbr]
        
        # Get team stats
        team1_ppg, team1_last_result = get_team_stats(team1_id)
        
        # Create prediction data
        pred_data = pd.DataFrame({
            'TEAM_ID': [team1_id],
            'OPPONENT_TEAM_ID': [team2_id],
            'Points_Per_Game': [team1_ppg],
            'HOME_GAME': [1 if is_team1_home else 0],
            'LAST_GAME_RESULT': [team1_last_result]
        })
        
        # Make prediction
        win_prob = model.predict_proba(pred_data)[0]
        prediction = model.predict(pred_data)[0]
        
        return {
            'team1': team1_full,
            'team2': team2_full,
            'home_team': team1_full if is_team1_home else team2_full,
            'prediction': 'Win' if prediction == 1 else 'Loss',
            'win_probability': win_prob[1],
            'loss_probability': win_prob[0]
        }
        
    except Exception as e:
        return {'error': str(e)}

def main():
    print("NBA Game Predictor")
    print("-----------------")
    print("Type 'exit' at any prompt to quit the program")
    
    # Show available teams
    name_to_abbr, last_names = get_team_mapping()
    print("\nAvailable teams:")
    for team_name in sorted(name_to_abbr.keys()):
        print(f"- {team_name}")
    print("\nYou can enter full names or just the team nickname (e.g., 'Lakers', 'Celtics')")
    
    while True:
        print("\nEnter team names:")
        team1 = input("Home team: ")
        if team1.lower() == 'exit':
            break
            
        team2 = input("Away team: ")
        if team2.lower() == 'exit':
            break
        
        result = predict_game(team1, team2, is_team1_home=True)
        
        if 'error' in result:
            print(f"\nError: {result['error']}")
        else:
            print(f"\nPrediction for {result['team1']} vs {result['team2']}:")
            print(f"Home team: {result['home_team']}")
            print(f"Predicted outcome: {result['prediction']}")
            print(f"Win probability: {result['win_probability']:.2%}")
            print(f"Loss probability: {result['loss_probability']:.2%}")
        
        print("\n" + "="*50)
    
    print("\nThanks for using NBA Game Predictor!")

if __name__ == "__main__":
    main() 