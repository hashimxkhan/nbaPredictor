# NBA Game Predictor

A machine learning model that predicts NBA game outcomes using historical data and team statistics.

## Credits

This project was inspired by the article [Predicting NBA Game Results Using Machine Learning and Python](https://medium.com/@juliuscecilia33/predicting-nba-game-results-using-machine-learning-and-python-6be209d6d165) by Julius Cecilia.

### Data Source
The project uses real-time NBA data through the official [NBA API](https://github.com/swar/nba_api), which provides access to NBA.com's data including:
- Game results
- Team statistics
- Player performance
- Historical matchups

## Features

- Fetches real NBA game data using the NBA API
- Uses Random Forest Classifier for predictions
- Considers factors like:
  - Team performance
  - Home court advantage
  - Recent points per game
  - Last game results
- Provides win/loss probabilities for matchups

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nbaPredictor.git
cd nbaPredictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model:
```bash
python src/train_model.py
```
This will:
- Fetch historical NBA game data
- Process and prepare the data
- Train the prediction model
- Save the model and team mappings

2. Make predictions:
```bash
python src/predict.py
```
Follow the prompts to:
- Enter the home team abbreviation (e.g., LAL for Lakers)
- Enter the away team abbreviation (e.g., BOS for Celtics)
- Get the prediction results with win probabilities

## Model Details

The prediction model uses the following features:
- Team IDs (both teams)
- Points per game (based on recent performance)
- Home court advantage
- Last game result

The model is trained using a Random Forest Classifier with 100 estimators, which helps capture complex patterns in team performance and game outcomes.

## Limitations

- The model's predictions are based on historical data and may not account for recent team changes or injuries
- Performance can vary based on the quality and quantity of available training data
- The model assumes that historical patterns will continue to be relevant for future games
