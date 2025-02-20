import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Get opponent team ID
def get_opponent_team_id(matchup, team_abbr_to_id, team_id):
    if '@' in matchup:
        opponent_abbr = matchup.split(' @ ')[-1]
    else:
        opponent_abbr = matchup.split(' vs. ')[-1]
    return team_abbr_to_id.get(opponent_abbr, team_id)

def confidence_to_american_odds(confidence):
    # Handle 50% edge case (neutral odds)
    if confidence == 0.5:
        return 100  # Even money

    # Convert probability to American odds
    if confidence > 0.5:
        odds = -100 * (confidence / (1 - confidence))  # Favorite (negative odds)
    else:
        odds = 100 * ((1 - confidence) / confidence)  # Underdog (positive odds)
    return round(odds)

def weighted_team_stats(team_abbr, opponent_abbr):
    team_games = all_games[all_games['TEAM_ABBREVIATION'] == team_abbr].copy()

    # Sort by game date (most recent first)
    team_games = team_games.sort_values(by='GAME_DATE', ascending=False)

    # Convert to datetime if not already
    team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])

    # Calculate time-based weights
    max_days = (team_games['GAME_DATE'].max() - team_games['GAME_DATE']).dt.days + 1
    time_weights = np.exp(-max_days / 7)  # Decay factor (higher number is slower decay)

    # Weigh matchup data
    opponent_weight = 2  # Games against today's opponent get 2x weight
    team_games['weight'] = np.where(team_games['OPPONENT_TEAM_ID'] == team_abbr_to_id.get(opponent_abbr, -1),
                                    time_weights * opponent_weight, time_weights)

    # Convert all numeric columns to float before multiplying
    numeric_cols = ['Points_Per_Game', 'OFF_REB', 'DEF_REB', 'TURNOVERS', 'FG%', '3P%', 'DEF_EFFICIENCY']
    team_games[numeric_cols] = team_games[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Compute weighted averages
    weighted_stats = (team_games[numeric_cols]
                      .multiply(team_games['weight'], axis=0)
                      .sum() / team_games['weight'].sum())

    return weighted_stats

# Get NBA team list
nba_teams = teams.get_teams()
team_abbr_to_id = {team['abbreviation']: team['id'] for team in nba_teams}

# Fetch all games
all_games = pd.DataFrame()
for team in nba_teams:
    team_id = team['id']
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    all_games = pd.concat([all_games, games], ignore_index=True)

# Convert date and add win column
all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
all_games['WIN'] = all_games['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Add factors
all_games['OFF_REB'] = all_games['OREB'].astype(float)
all_games['DEF_REB'] = all_games['DREB'].astype(float)
all_games['TURNOVERS'] = all_games['TOV'].astype(float)
all_games['FG%'] = all_games['FG_PCT'].astype(float)
all_games['3P%'] = all_games['FG3_PCT'].astype(float)

# Calculate Points Allowed
all_games['PTS_ALLOWED'] = all_games.groupby('GAME_ID')['PTS'].transform('sum') - all_games['PTS']

# Calculate Possessions
all_games['POSS'] = 0.5 * (
    (all_games['FGA'] + 0.4 * all_games['FTA'] - all_games['OREB'] + all_games['TOV']) +
    (all_games['PTS_ALLOWED'] + 0.4 * all_games['FTA'] - all_games['OREB'] + all_games['TOV'])
)

# Calculate Defensive Efficiency (Points allowed per possession)
all_games['DEF_EFFICIENCY'] = all_games['PTS_ALLOWED'] / all_games['POSS']

# Calculate PPG
all_games['Points_Per_Game'] = all_games.groupby('TEAM_ID')['PTS'].transform('mean')

all_games['OPPONENT_TEAM_ID'] = all_games.apply(
    lambda row: get_opponent_team_id(row['MATCHUP'], team_abbr_to_id, row['TEAM_ID']), axis=1
)

# Add home/away indicator
all_games['HOME_GAME'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

# Add result of the last game
all_games['LAST_GAME_RESULT'] = all_games.groupby('TEAM_ID')['WIN'].shift(1).fillna(0)

# Encode categorical variables
le = LabelEncoder()
all_games['TEAM_ID'] = le.fit_transform(all_games['TEAM_ID'])
all_games['OPPONENT_TEAM_ID'] = le.fit_transform(all_games['OPPONENT_TEAM_ID'])

# Select features and target
X = all_games[[
    'TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME', 'LAST_GAME_RESULT',
    'OFF_REB', 'DEF_REB', 'TURNOVERS', 'FG%', '3P%', 'DEF_EFFICIENCY'
]]
y = all_games['WIN']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Display feature importances
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances['importance'].plot(kind='bar', color='skyblue')
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

################################################### LIVE PART ##########################################################
today = datetime.today().strftime('%Y-%m-%d')

# For future dates, use the below code to get date instead
# To get dates further in advance, just change days = x where x is how many days in advance you want
#td = datetime.today()
#today = (td + timedelta(days=1)).strftime('%Y-%m-%d')

# Get NBA team information
nba_teams = teams.get_teams()
team_id_to_abbr = {team['id']: team['abbreviation'] for team in nba_teams}

# Get matchup data for today's games
scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
games_today = scoreboard.game_header.get_data_frame() 
predictions = []

# Iterate through games today
for _, game in games_today.iterrows():
    home_team_id = game['HOME_TEAM_ID']
    away_team_id = game['VISITOR_TEAM_ID']
    home_team_name = team_id_to_abbr[home_team_id]
    away_team_name = team_id_to_abbr[away_team_id]

    home_team_games = all_games.loc[all_games['TEAM_ABBREVIATION'] == home_team_name]
    away_team_games = all_games.loc[all_games['TEAM_ABBREVIATION'] == home_team_name]

    print(f"{home_team_name} vs. {away_team_name}")

    # Make sure team id's are valid
    if home_team_id not in team_id_to_abbr or away_team_id not in team_id_to_abbr:
        print("Skipping: Invalid team ID mapping.")
        continue

    # Get weighted stats
    home_stats = weighted_team_stats(home_team_name, away_team_name)
    away_stats = weighted_team_stats(away_team_name, home_team_name)


    print(f"{home_team_name} Stats: {home_stats}")
    print(f"{away_team_name} Stats: {away_stats}")

    # Check if stats are missing
    if home_team_games.empty or away_team_games.empty:
        print(f"Skipping: No historical data for {home_team_id} or {away_team_id}")
        continue

    # Get game features
    game_features = pd.DataFrame([[
        home_team_id, 
        away_team_id, 
        home_stats['Points_Per_Game'], 
        1, 
        home_team_games['LAST_GAME_RESULT'].values[-1] if not home_team_games.empty else 0,  
        home_stats['OFF_REB'], 
        home_stats['DEF_REB'], 
        home_stats['TURNOVERS'], 
        home_stats['FG%'], 
        home_stats['3P%'], 
        home_stats['DEF_EFFICIENCY']
    ]], columns=X_train.columns)


    # Predict Winner
    probabilities = model.predict_proba(game_features)  # Get probabilities
    predicted_win = np.argmax(probabilities[0])  # Choose team with highest probability
    confidence = max(probabilities[0])  # Take the highest probability as confidence
    winner = home_team_name if predicted_win == 1 else away_team_name
    odds = confidence_to_american_odds(confidence) # Convert to American betting odds


    print(f"Predicted Winner: {winner}, confidence: {confidence}")

    predictions.append({'Matchup': f"{home_team_name} vs. {away_team_name}", 'Predicted Winner': winner, 'Confidence': confidence, 'Odds': odds})

# Print predictions
if predictions:
    predictions_df = pd.DataFrame(predictions)
    print("Final Predictions:\n", predictions_df)
else:
    print("No predictions made.")

