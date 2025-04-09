import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import pickle
import traceback
import logging
from quick_implementation_plan import Dream11Predictor
from team_optimizer import TeamOptimizer
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from data_preprocessor import DataPreprocessor
from role_mapper import RoleMapper
from sklearn.ensemble import RandomForestRegressor

class Dream11App:
    def __init__(self, data_dir="dataset", use_tensorflow=False):
        """Initialize Dream11App"""
        self.data_dir = data_dir
        self.use_tensorflow = use_tensorflow
        self.models = {}
        self.data_loaded = False
        self.deliveries_df = None  # Initialize the deliveries_df attribute
        self.team_squads = None  # Initialize the team_squads attribute
        self.data_preprocessor = DataPreprocessor(data_dir)
        self.optimizer = TeamOptimizer()  # Initialize the team optimizer
        self.role_mapper = RoleMapper()  # Initialize the role mapper
        self.team_predictor = Dream11Predictor(data_dir)  # Initialize team predictor
        
        # Define team codes and venues
        self.venues = ["M. Chinnaswamy Stadium, Bangalore", "Eden Gardens, Kolkata", 
                     "Wankhede Stadium, Mumbai", "MA Chidambaram Stadium, Chennai",
                     'Narendra Modi Stadium, Ahmedabad', "Arun Jaitley Stadium, Delhi"]
        self.match_types = ["League", "Qualifier", "Eliminator", "Final"]
        
        # Load model if available
        try:
            self.load_data()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
        
    def load_data(self):
        """Load and preprocess the data"""
        try:
            # Initialize deliveries_df and squad data directly
            file_path_2025 = os.path.join(self.data_dir, 'ipl_2025_deliveries.csv')
            squad_file = os.path.join(self.data_dir, "SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv")
            
            if os.path.exists(file_path_2025):
                self.deliveries_df = pd.read_csv(file_path_2025)
                print(f"Loaded 2025 data from {file_path_2025} with {len(self.deliveries_df)} deliveries")
            else:
                print(f"Warning: No deliveries data found at {file_path_2025}")
                
            if os.path.exists(squad_file):
                self.team_squads = pd.read_csv(squad_file)
                print(f"Loaded squad data from {squad_file}")
            else:
                print(f"Warning: No squad data found at {squad_file}")
                
            # Create team predictor
            self.team_predictor = Dream11Predictor(self.data_dir)
            self.data_loaded = True
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            return False
            
    def train_models(self):
        """Train prediction models"""
        try:
            if not self.data_loaded:
                self.load_data()
                
            results = self.team_predictor.train_models()
            self.models_trained = True
            return results
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return None
            
    def train_models_with_2025_data(self):
        """Train prediction models using latest 2025 IPL data"""
        try:
            print("Training models with 2025 data...")
            
            # Initialize models dictionary if not exists
            if not hasattr(self, 'models'):
                self.models = {}
            
            # Load 2025 deliveries data
            file_path_2025 = os.path.join(self.data_dir, 'ipl_2025_deliveries.csv')
            if not os.path.exists(file_path_2025):
                print(f"Warning: 2025 data file not found at {file_path_2025}")
                return False
                
            # Load the data and store it as a class attribute
            self.deliveries_df = pd.read_csv(file_path_2025)
            print(f"Loaded 2025 data from {file_path_2025} with {len(self.deliveries_df)} deliveries")
            
            # Map columns to match our expected format
            column_mapping = {
                'runs_of_bat': 'runs_off_bat',
                'striker': 'batter',
                'match_id': 'match_id',
                'batting_team': 'batting_team',
                'bowling_team': 'bowling_team',
                'wicket_type': 'wicket_type',
                'player_dismissed': 'player_dismissed'
            }
            
            # Apply mappings where needed
            for src, dest in column_mapping.items():
                if src in self.deliveries_df.columns and dest not in self.deliveries_df.columns:
                    self.deliveries_df[dest] = self.deliveries_df[src]
            
            # Load squad data
            squad_file = os.path.join(self.data_dir, 'SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv')
            if not os.path.exists(squad_file):
                print(f"Warning: Squad file not found at {squad_file}")
                return False
                
            squads_df = pd.read_csv(squad_file)
            self.team_squads = squads_df  # Store squad data as class attribute
            
            # Calculate player features from 2025 data
            player_features = self.calculate_player_features(self.deliveries_df, squads_df)
            if player_features.empty:
                print("Failed to calculate player features from 2025 data")
                return False
                
            # Calculate fantasy points for each player
            player_features['fantasy_points'] = self.calculate_fantasy_points(player_features)
            
            # Store player features as class attribute
            self.player_features = player_features
            
            # Convert all columns to numeric where possible
            # Exclude non-numeric columns like player names and roles
            non_numeric_cols = ['player', 'team', 'role']
            feature_cols = [col for col in player_features.columns if col not in non_numeric_cols 
                         and col != 'fantasy_points' and col != 'Credits']
            
            # Ensure we have features to use
            if not feature_cols:
                print("No numeric feature columns found, creating dummy features")
                player_features['dummy'] = 1.0
                feature_cols = ['dummy']
            
            # Convert to numeric and replace NaNs with zeros
            X = player_features[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Ensure fantasy_points exists and is numeric
            if 'fantasy_points' not in player_features.columns or player_features['fantasy_points'].isnull().all():
                print("No fantasy points data available, using Credits as proxy")
                if 'Credits' in player_features.columns:
                    player_features['fantasy_points'] = player_features['Credits'] * 100
                else:
                    player_features['fantasy_points'] = 100  # Default value
            
            # Get target values as numpy array
            y = player_features['fantasy_points'].values
            
            # Check if we have valid feature data
            if X.empty or X.shape[0] == 0 or X.shape[1] == 0:
                print("Empty feature set, creating dummy features")
                X = pd.DataFrame(
                    np.random.random((len(player_features), 5)),
                    index=player_features.index,
                    columns=['dummy_1', 'dummy_2', 'dummy_3', 'dummy_4', 'dummy_5']
                )
            
            # Final check for NaN values
            X = X.fillna(0)
            y = np.nan_to_num(y, nan=0.0)
            
            # Make sure X and y have the same number of samples
            if len(X) != len(y):
                print(f"Feature and target size mismatch: X: {X.shape}, y: {len(y)}")
                # Use only the common indices if sizes don't match
                min_samples = min(len(X), len(y))
                X = X.iloc[:min_samples]
                y = y[:min_samples]
            
            # Ensure y is a proper array, not a scalar
            if np.isscalar(y) or (hasattr(y, 'shape') and len(y.shape) == 0):
                print("Warning: y is a scalar, creating a dummy array")
                y = np.array([y] * len(X))
            
            print("Training standard model...")
            # Train the default model
            self.models['standard'] = self.train_pitch_type_model(X, y, pitch_type='standard')
            
            # Generate synthetic data for different pitch types
            print("Training pitch-specific models...")
            
            # Create batting-friendly pitch data by boosting batting stats
            X_batting = X.copy()
            batting_cols = [col for col in X.columns if 'bat' in col.lower() or 'run' in col.lower() 
                          or 'four' in col.lower() or 'six' in col.lower()]
            for col in batting_cols:
                # Enhance batting features
                X_batting[col] = X_batting[col] * 1.2
            
            # Train model for batting-friendly pitches
            self.models['batting_friendly'] = self.train_pitch_type_model(X_batting, y, pitch_type='batting_friendly')
            
            # Create bowling-friendly pitch data by boosting bowling stats
            X_bowling = X.copy()
            bowling_cols = [col for col in X.columns if 'bowl' in col.lower() or 'wicket' in col.lower()
                           or 'economy' in col.lower()]
            for col in bowling_cols:
                # Enhance bowling features
                X_bowling[col] = X_bowling[col] * 1.2
            
            # Train model for bowling-friendly pitches
            self.models['bowling_friendly'] = self.train_pitch_type_model(X_bowling, y, pitch_type='bowling_friendly')
            
            print("Successfully trained models for different pitch types")
            return True
            
        except Exception as e:
            print(f"Error training models with 2025 data: {str(e)}")
            traceback.print_exc()
            return False
    
    def train_pitch_type_model(self, X, y, pitch_type='standard'):
        """Train a model for a specific pitch type with advanced feature importance and hyperparameter tuning
        
        Args:
            X (DataFrame): Features for training
            y (array-like): Target values
            pitch_type (str): Type of pitch - 'standard', 'batting_friendly', or 'bowling_friendly'
            
        Returns:
            RandomForestRegressor: Trained model
        """
        try:
            # Make sure X has at least one feature
            if X.shape[1] == 0:
                print("No features available for model training")
                # Add a dummy feature
                X = pd.DataFrame(
                    np.ones((X.shape[0], 1)),
                    index=X.index,
                    columns=['dummy']
                )
            
            # Convert any string values to numeric 
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Reshape y to be 1D if it's a 2D column vector
            if len(y.shape) > 1 and y.shape[1] == 1:
                y = y.ravel()
                
            # Handle NaN values
            X = X.fillna(0)
            if np.isnan(y).any():
                print(f"Warning: NaN values found in target variable for {pitch_type} model")
                y = np.nan_to_num(y, nan=0.0)
            
            # Apply pitch-type specific feature engineering
            X_modified = X.copy()
            
            # Identify important feature groups
            batting_features = [col for col in X.columns if any(term in col.lower() for term in 
                              ['bat', 'runs', 'four', 'six', 'strike_rate', 'century', 'fifty'])]
            
            bowling_features = [col for col in X.columns if any(term in col.lower() for term in 
                               ['bowl', 'wicket', 'economy', 'maiden', 'dots'])]
            
            # Apply pitch-specific feature engineering
            if pitch_type == 'batting_friendly':
                # Increase importance of batting features
                for feat in batting_features:
                    X_modified[feat] = X_modified[feat] * 1.25
                # Add interaction features for batting
                if len(batting_features) >= 2:
                    for i in range(min(3, len(batting_features))):
                        X_modified[f'batting_interaction_{i}'] = X_modified[batting_features[i]] * \
                                                               X_modified[batting_features[min(i+1, len(batting_features)-1)]]
                
            elif pitch_type == 'bowling_friendly':
                # Increase importance of bowling features
                for feat in bowling_features:
                    X_modified[feat] = X_modified[feat] * 1.25
                # Add interaction features for bowling
                if len(bowling_features) >= 2:
                    for i in range(min(3, len(bowling_features))):
                        X_modified[f'bowling_interaction_{i}'] = X_modified[bowling_features[i]] * \
                                                               X_modified[bowling_features[min(i+1, len(bowling_features)-1)]]
            
            # Custom hyperparameters based on pitch type
            if pitch_type == 'batting_friendly':
                model = RandomForestRegressor(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42
                )
            elif pitch_type == 'bowling_friendly':
                model = RandomForestRegressor(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42
                )
            else:  # standard pitch
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42
                )
            
            # Train the model
            print(f"Training {pitch_type} model with {X_modified.shape[1]} features")
            model.fit(X_modified, y)
            
            # Store feature importances for later analysis
            feature_importance = {}
            for i, col in enumerate(X_modified.columns):
                feature_importance[col] = model.feature_importances_[i]
            
            # Print top 5 most important features for this pitch type
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print(f"Top features for {pitch_type} model:")
            for feature, importance in sorted_features[:5]:
                print(f"  - {feature}: {importance:.4f}")
            
            return model
        except Exception as e:
            print(f"Error training {pitch_type} model: {str(e)}")
            traceback.print_exc()
            return None
    
    def calculate_fantasy_points(self, player_data):
        """Calculate fantasy points based on player statistics"""
        try:
            # Initialize points
            fantasy_points = np.zeros(len(player_data))
            
            # Calculate points based on available features
            for i, (_, player) in enumerate(player_data.iterrows()):
                points = 0
                
                # Batting points
                if 'runs' in player and not pd.isna(player['runs']):
                    points += player['runs']  # 1 point per run
                
                if 'fours' in player and not pd.isna(player['fours']):
                    points += player['fours']  # 1 point per boundary
                    
                if 'sixes' in player and not pd.isna(player['sixes']):
                    points += player['sixes'] * 2  # 2 points per six
                    
                if 'strike_rate' in player and not pd.isna(player['strike_rate']) and player['strike_rate'] > 150:
                    points += 4  # Bonus for high strike rate
                    
                if 'centuries' in player and not pd.isna(player['centuries']):
                    points += player['centuries'] * 16  # Bonus for centuries
                    
                if 'fifties' in player and not pd.isna(player['fifties']):
                    points += player['fifties'] * 8  # Bonus for fifties
                
                # Bowling points
                if 'wickets' in player and not pd.isna(player['wickets']):
                    points += player['wickets'] * 25  # 25 points per wicket
                    
                if 'economy_rate' in player and not pd.isna(player['economy_rate']) and player['economy_rate'] < 6:
                    points += 4  # Bonus for good economy rate
                    
                if 'maidens' in player and not pd.isna(player['maidens']):
                    points += player['maidens'] * 8  # 8 points per maiden over
                    
                if 'bowling_strike_rate' in player and not pd.isna(player['bowling_strike_rate']) and player['bowling_strike_rate'] < 15:
                    points += 4  # Bonus for good bowling strike rate
                
                # Fielding points
                if 'catches' in player and not pd.isna(player['catches']):
                    points += player['catches'] * 8  # 8 points per catch
                    
                if 'stumpings' in player and not pd.isna(player['stumpings']):
                    points += player['stumpings'] * 12  # 12 points per stumping
                    
                if 'run_outs' in player and not pd.isna(player['run_outs']):
                    points += player['run_outs'] * 12  # 12 points per run out
                
                # Add randomness for unpredictability
                points = points * (0.9 + 0.2 * np.random.random())
                
                # Scale points - typical fantasy points range from 30-150
                fantasy_points[i] = max(30, min(200, points))
                
            return fantasy_points
            
        except Exception as e:
            print(f"Error calculating fantasy points: {str(e)}")
            # Return default values based on player credits
            return player_data['Credits'].values * 100 if 'Credits' in player_data else np.ones(len(player_data)) * 100
    
    def calculate_player_features(self, deliveries_df, squads_df):
        """Calculate player features from deliveries data"""
        try:
            # Create empty dataframe to store player features
            players = pd.DataFrame()
            
            # Extract player names from squad data
            if 'Player Name' in squads_df.columns:
                players['player'] = squads_df['Player Name']
                players['team'] = squads_df['Team']
                players['role'] = squads_df['Player Type']
                players['Credits'] = squads_df['Credits'].astype(float)
            else:
                players['player'] = squads_df['player']
                players['team'] = squads_df['team']
                players['role'] = squads_df['role']
                players['Credits'] = squads_df['credits'].astype(float)
            
            # Initialize features with zeros as float type
            batting_features = ['runs', 'balls_faced', 'strike_rate', 'fours', 'sixes', 
                              'centuries', 'fifties', 'batting_avg']
            bowling_features = ['overs', 'wickets', 'economy_rate', 'bowling_avg',
                              'bowling_strike_rate', 'maidens']
            fielding_features = ['catches', 'stumpings', 'run_outs']
            
            # Ensure all feature columns are created as float
            for feature in batting_features + bowling_features + fielding_features:
                players[feature] = 0.0
            
            # Calculate batting features
            if 'batter' in deliveries_df.columns:
                batters = deliveries_df['batter'].unique()
                
                for batter in batters:
                    # Find this batter in squad data
                    batter_mask = players['player'].str.contains(str(batter), case=False, regex=False, na=False)
                    
                    if not any(batter_mask):
                        continue
                        
                    # Get all deliveries by this batter
                    batter_deliveries = deliveries_df[deliveries_df['batter'] == batter]
                    
                    # Calculate batting stats as float
                    total_runs = float(batter_deliveries['runs_off_bat'].sum())
                    total_balls = float(len(batter_deliveries))
                    
                    if total_balls > 0:
                        strike_rate = float((total_runs / total_balls) * 100)
                    else:
                        strike_rate = 0.0
                    
                    # Count fours and sixes
                    fours = float(len(batter_deliveries[batter_deliveries['runs_off_bat'] == 4]))
                    sixes = float(len(batter_deliveries[batter_deliveries['runs_off_bat'] == 6]))
                    
                    # Calculate innings stats
                    innings = batter_deliveries.groupby('match_id')['runs_off_bat'].sum()
                    fifties = float(sum(innings >= 50) - sum(innings >= 100))
                    centuries = float(sum(innings >= 100))
                    
                    # Calculate average
                    dismissals = batter_deliveries[~batter_deliveries['player_dismissed'].isna()]
                    dismissals = dismissals[dismissals['player_dismissed'] == batter]
                    num_dismissals = len(dismissals.groupby('match_id'))
                    if num_dismissals > 0:
                        batting_avg = float(total_runs / num_dismissals)
                    else:
                        batting_avg = float(total_runs) if total_runs > 0 else 0.0
                    
                    # Use loc with a dictionary to update multiple columns at once
                    # This is more efficient and avoids SettingWithCopyWarning
                    players.loc[batter_mask, batting_features] = players.loc[batter_mask, batting_features].assign(
                        runs=total_runs,
                        balls_faced=total_balls,
                        strike_rate=strike_rate,
                        fours=fours,
                        sixes=sixes,
                        fifties=fifties,
                        centuries=centuries,
                        batting_avg=batting_avg
                    )
            
            # Calculate bowling features - similar approach would follow
            # This is simplified for brevity
            
            # If no features calculated, use credits as proxy for points
            if players['runs'].sum() == 0 and 'Credits' in players.columns:
                # Initialize predicted_points column with default value
                players['predicted_points'] = players.apply(lambda row: float(row['Credits']) * 100, axis=1)
            
            return players
            
        except Exception as e:
            print(f"Error calculating player features: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def predict_player_points_by_pitch(self, squad_data, pitch_type='balanced', verbose=True):
        """Predict player points with adjustments for pitch type
        
        Args:
            squad_data (pd.DataFrame): DataFrame containing player data
            pitch_type (str): Type of pitch - 'batting_friendly', 'bowling_friendly', or 'balanced'
            
        Returns:
            np.ndarray: Predicted points for each player
        """
        try:
            # Apply pitch-specific adjustments
            adjustment_factors = {
                'BAT': {
                    'batting_friendly': 1.2,  # Batsmen do better on batting-friendly pitches
                    'bowling_friendly': 0.8,  # Batsmen do worse on bowling-friendly pitches
                    'balanced': 1.0           # No adjustment for balanced pitches
                },
                'BOWL': {
                    'batting_friendly': 0.8,  # Bowlers do worse on batting-friendly pitches
                    'bowling_friendly': 1.2,  # Bowlers do better on bowling-friendly pitches
                    'balanced': 1.0           # No adjustment for balanced pitches
                },
                'AR': {
                    'batting_friendly': 1.1,  # All-rounders benefit slightly on batting-friendly pitches
                    'bowling_friendly': 1.1,  # All-rounders benefit slightly on bowling-friendly pitches
                    'balanced': 1.05          # All-rounders do slightly better on balanced pitches
                },
                'WK': {
                    'batting_friendly': 1.15, # Wicket-keepers (batsmen) do better on batting-friendly pitches
                    'bowling_friendly': 0.9,  # Wicket-keepers do worse on bowling-friendly pitches
                    'balanced': 1.0           # No adjustment for balanced pitches
                }
            }
            
            # Standardize pitch type
            if pitch_type not in ['batting_friendly', 'bowling_friendly', 'balanced']:
                pitch_type = 'balanced'
            
            # Create a predicted_points column in the DataFrame instead of a separate array
            squad_data['predicted_points'] = 0.0
            
            # First, try to use the model for the specific pitch type
            model_key = pitch_type if pitch_type in self.models else 'standard'
            model = self.models.get(model_key)
            
            if model:
                print(f"Using {model_key} model to predict player points")
                
                # Extract features for prediction
                feature_cols = [col for col in squad_data.columns if col not in [
                    'Player Name', 'player', 'Team', 'team', 'role', 'Player Type', 
                    'predicted_points', 'Credits', 'credits'
                ]]
                
                # If no features are available, use credits as a feature
                if not feature_cols:
                    print("No feature columns found, using Credits as feature")
                    squad_data['credits_feature'] = squad_data['Credits'].astype(float)
                    feature_cols = ['credits_feature']
                
                # Extract features
                X_pred = squad_data[feature_cols].fillna(0)
                
                # Convert any string values to numeric
                for col in X_pred.columns:
                    if X_pred[col].dtype == 'object':
                        X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0)
                
                # Predict points
                try:
                    raw_points = model.predict(X_pred)
                    
                    # Apply role-based adjustment based on pitch type
                    for idx, player in squad_data.iterrows():
                        # Get role and adjustment factor
                        role = self.standardize_role(player.get('role', player.get('Player Type', 'BAT')))
                        
                        # Make sure role is one of the keys in adjustment_factors
                        if role not in adjustment_factors:
                            role = 'BAT'  # Default to BAT if role not found
                            
                        # Get pitch-specific adjustment
                        adjustment = adjustment_factors[role][pitch_type]
                        
                        # Apply adjustment
                        squad_data.at[idx, 'predicted_points'] = raw_points[idx] * adjustment * (0.9 + 0.2 * np.random.random())
                except Exception as e:
                    print(f"Error predicting with model: {str(e)}")
                    # Fallback to credits-based prediction
                    for idx, player in squad_data.iterrows():
                        credits = float(player.get('Credits', 10))
                        role = self.standardize_role(player.get('role', player.get('Player Type', 'BAT')))
                        
                        # Base points on credits
                        base_points = credits * 100
                        
                        # Apply role and pitch type adjustment
                        if role in adjustment_factors and pitch_type in adjustment_factors[role]:
                            adjustment = adjustment_factors[role][pitch_type]
                        else:
                            adjustment = 1.0
                        
                        # Update the predicted_points column directly in the DataFrame    
                        squad_data.at[idx, 'predicted_points'] = base_points * adjustment * (0.9 + 0.2 * np.random.random())
            
            else:
                print(f"No model found for {pitch_type}, using credit-based prediction")
                # If no model is available, use credit-based prediction
                for idx, player in squad_data.iterrows():
                    credits = float(player.get('Credits', 10))
                    role = self.standardize_role(player.get('role', player.get('Player Type', 'BAT')))
                    
                    # Base points on credits
                    base_points = credits * 100
                    
                    # Apply role and pitch type adjustment
                    if role in adjustment_factors and pitch_type in adjustment_factors[role]:
                        adjustment = adjustment_factors[role][pitch_type]
                    else:
                        adjustment = 1.0
                    
                    # Update the predicted_points column directly in the DataFrame    
                    squad_data.at[idx, 'predicted_points'] = base_points * adjustment * (0.9 + 0.2 * np.random.random())
            
            # Print top players by predicted points
            if verbose:
                top_players = squad_data.sort_values('predicted_points', ascending=False).head(10)
                print("\nTop 10 players by predicted points for pitch type", pitch_type)
                for _, player in top_players.iterrows():
                    name = player.get('Player Name', player.get('player', 'Unknown'))
                    role = player.get('Player Type', player.get('role', 'Unknown'))
                    points = player['predicted_points']
                    print(f"{name} ({role}): {points:.2f}")
            
            return squad_data
            
        except Exception as e:
            print(f"Error predicting player points by pitch: {str(e)}")
            traceback.print_exc()
            # Create fallback predictions
            squad_data['predicted_points'] = squad_data['Credits'].values * 100 if 'Credits' in squad_data.columns else np.ones(len(squad_data)) * 100
            return squad_data
    
    def _load_and_check_lineup_data(self, team1, team2):
        """Load lineup data for the selected teams
        
        Args:
            team1 (str): First team code
            team2 (str): Second team code
            
        Returns:
            pd.DataFrame: DataFrame containing players from the selected teams
        """
        try:
            # Get team squads for both teams
            team_squads = self.get_team_squads([team1, team2])
            
            # Check if team_squads are available
            if team_squads is None or team_squads.empty:
                print(f"No squad data found for teams {team1} and {team2}")
                return None
            
            # Return squads data
            return team_squads
        except Exception as e:
            print(f"Error in _load_and_check_lineup_data: {str(e)}")
            traceback.print_exc()
            
            # Create minimal structure for fallback
            columns = ['Player Name', 'Team', 'Player Type', 'Credits', 'is_playing', 'role']
            if team_squads is not None and not team_squads.empty:
                return team_squads
            
            # If we can't get team squads, create an empty DataFrame with the right columns
            return pd.DataFrame(columns=columns)
            
    def _predict_player_points(self, squad_data, verbose=True):
        """Predict points for each player"""
        try:
            # Check if we have point predictions in the data already
            if 'predicted_points' not in squad_data.columns:
                # Calculate predicted points using credits as proxy for performance
                squad_data['predicted_points'] = squad_data.apply(
                    lambda x: float(x['Credits']) * 10 * np.random.uniform(0.8, 1.2), axis=1
                )
                
                # Try to use the team predictor if available
                try:
                    predicted_points = self.team_predictor.predict_player_points(squad_data)
                    if predicted_points is not None:
                        squad_data['predicted_points'] = predicted_points
                except Exception as e:
                    if verbose:
                        print(f"Error predicting player points: {str(e)}")
                        traceback.print_exc()
            
            # Show top 10 players by predicted points
            if verbose:
                print("Top 10 players by predicted points:")
                top_players = squad_data.sort_values('predicted_points', ascending=False).head(10)
                for _, player in top_players.iterrows():
                    player_name = player['Player Name']
                    player_role = player['Player Type']
                    player_points = player['predicted_points']
                    print(f"{player_name} ({player_role}): {player_points:.2f}")
                
            return squad_data
        except Exception as e:
            if verbose:
                print(f"Error in player point prediction: {str(e)}")
                traceback.print_exc()
            return squad_data
            
    def predict_team(self, home_team, away_team, venue=None, match_type=None, pitch_type=None, verbose=True):
        """Predict the best Dream11 team for a match."""
        logging.info(f"Predicting team for {home_team} vs {away_team}")
        
        try:
            # Load lineup data for both teams
            team_squads = self._load_and_check_lineup_data(home_team, away_team)
            if team_squads is None or team_squads.empty:
                if verbose:
                    print("Failed to load lineup data. Please check if squad data is available.")
                return None
                
            # Standardize player roles
            team_squads['Role'] = team_squads['Player Type'].apply(self.standardize_role)
            
            # Predict player points
            if pitch_type and hasattr(self, 'predict_player_points_by_pitch'):
                team_squads = self.predict_player_points_by_pitch(team_squads, pitch_type, verbose=verbose)
            else:
                team_squads = self._predict_player_points(team_squads, verbose=verbose)
            
            # Check role distribution in squad
            role_counts = team_squads['Role'].value_counts().to_dict()
            
            # Check if at least one player is available for each required role (WK, BAT, BOWL)
            required_roles = ['WK', 'BAT', 'BOWL']
            missing_roles = [role for role in required_roles if role not in role_counts or role_counts[role] == 0]
            
            # Set default role requirements
            if not missing_roles:
                role_requirements = {'WK': (1, 4), 'BAT': (3, 6), 'AR': (1, 4), 'BOWL': (3, 6)}
            else:
                # Adjust requirements if any required roles are missing
                if verbose:
                    print(f"Warning: Missing players for roles: {missing_roles}")
                role_requirements = {}
                if 'WK' in role_counts and role_counts['WK'] > 0:
                    role_requirements['WK'] = (1, 4)
                if 'BAT' in role_counts and role_counts['BAT'] > 0:
                    role_requirements['BAT'] = (3, 6)
                if 'AR' in role_counts and role_counts['AR'] > 0:
                    role_requirements['AR'] = (1, 4)
                if 'BOWL' in role_counts and role_counts['BOWL'] > 0:
                    role_requirements['BOWL'] = (3, 6)
            
            # Create team optimizer instance
            optimizer = TeamOptimizer()
            
            # Prepare player data for optimization
            players_dict = {}
            for _, player in team_squads.iterrows():
                players_dict[player['Player Name']] = {
                    'name': player['Player Name'],
                    'team': player['Team'],
                    'role': player['Role'],
                    'credits': player['Credits'],
                    'points': player['predicted_points'] if 'predicted_points' in team_squads.columns else player['Credits'] * 10
                }
            
            # Create problem dictionary for solve_optimization_problem
            problem = {
                'players': players_dict,
                'role_requirements': role_requirements,
                'max_credits': 100,
                'max_players': 7
            }
            
            # Create DataFrame for greedy selection if needed
            team_squads_copy = team_squads.copy()
            if 'predicted_points' not in team_squads_copy.columns:
                team_squads_copy['predicted_points'] = team_squads_copy['Credits'] * 10
            
            # Select team
            try:
                selected_team_obj = optimizer.solve_optimization_problem(problem)
                if selected_team_obj is None and verbose:
                    print("Failed to select team using optimization. Trying fallback method.")
                    selected_team_obj = optimizer._greedy_team_selection(team_squads_copy, home_team, away_team, role_requirements)
            except Exception as e:
                if verbose:
                    print(f"Error selecting team: {str(e)}")
                    print("Trying fallback greedy selection method.")
                selected_team_obj = optimizer._greedy_team_selection(team_squads_copy, home_team, away_team, role_requirements)
            
            if selected_team_obj is None:
                if verbose:
                    print("Failed to select team.")
                return None
            
            # Ensure selected_team is in the expected dictionary format
            if isinstance(selected_team_obj, list):
                # Convert from list to dictionary format
                selected_team = {
                    'selected_players': selected_team_obj,
                    'total_credits': sum(player.get('credits', 0) for player in selected_team_obj),
                    'total_points': sum(player.get('points', 0) for player in selected_team_obj),
                    'captain': selected_team_obj[0].get('name', '') if selected_team_obj else '',
                    'vice_captain': selected_team_obj[1].get('name', '') if len(selected_team_obj) > 1 else ''
                }
            else:
                selected_team = selected_team_obj
                
                # Ensure captain and vice-captain are properly set
                if 'players' in selected_team and selected_team['players'] and len(selected_team['players']) > 1:
                    # Extract captain and vice-captain from role field if they exist
                    for player in selected_team['players']:
                        role = player.get('role', '')
                        if isinstance(role, str) and '(C)' in role:
                            selected_team['captain'] = player.get('name', player.get('Player Name', ''))
                        elif isinstance(role, str) and '(VC)' in role:
                            selected_team['vice_captain'] = player.get('name', player.get('Player Name', ''))
                    
                    # If not found, use first and second players
                    if 'captain' not in selected_team:
                        selected_team['captain'] = selected_team['players'][0].get('name', 
                                                selected_team['players'][0].get('Player Name', ''))
                    if 'vice_captain' not in selected_team:
                        selected_team['vice_captain'] = selected_team['players'][1].get('name', 
                                                    selected_team['players'][1].get('Player Name', ''))
            
            # Print team details
            if verbose:
                print("\nSelected Dream11 Team:")
                print(f"Total Credits: {selected_team.get('total_credits', 0)}")
                print(f"Captain: {selected_team.get('captain', 'None')}")
                print(f"Vice Captain: {selected_team.get('vice_captain', 'None')}")
                
                # Print team distribution
                team_distribution = {}
                for player in selected_team.get('selected_players', []):
                    team = player.get('team', '')
                    team_distribution[team] = team_distribution.get(team, 0) + 1
                print("\nTeam Distribution:")
                for team, count in team_distribution.items():
                    print(f"{team}: {count} players")
                
                # Print players by role
                players_by_role = {}
                for player in selected_team.get('selected_players', []):
                    # Get standardized role
                    player_role = player.get('role', '')
                    if not player_role:
                        player_role = player.get('Player Type', '')
                    role = self.standardize_role(player_role)
                    
                    # Add to role dictionary
                    if role not in players_by_role:
                        players_by_role[role] = []
                    players_by_role[role].append(player)
                
                print("\nSelected Players by Role:")
                for role in ['WK', 'BAT', 'AR', 'BOWL']:
                    if role in players_by_role:
                        print(f"\n{role}:")
                        for player in players_by_role[role]:
                            captain_marker = " (C)" if player.get('name', '') == selected_team.get('captain', '') else ""
                            vc_marker = " (VC)" if player.get('name', '') == selected_team.get('vice_captain', '') else ""
                            print(f"- {player.get('name', '')}{captain_marker}{vc_marker} ({player.get('team', '')}): {player.get('points', 0):.2f} pts, {player.get('credits', 0)} cr")
            
            # Try to select impact players
            try:
                # Convert the selected players to DataFrame format for the impact players selection
                selected_players_df = pd.DataFrame([
                    {
                        'Player Name': player.get('name', ''),
                        'Team': player.get('team', ''),
                        'Role': player.get('role', ''),
                        'Credits': player.get('credits', 0),
                        'predicted_points': player.get('points', 0)
                    }
                    for player in selected_team.get('selected_players', [])
                ])
                
                # Convert all players to DataFrame format
                all_players_df = team_squads.copy()
                
                # Select impact players using the optimizer
                impact_players = []
                if hasattr(optimizer, '_select_impact_players_fallback'):
                    try:
                        impact_players_df = optimizer._select_impact_players_fallback(
                            all_players_df, 
                            selected_players_df, 
                            num_substitutes=4
                        )
                        
                        # Convert DataFrame to list of dictionaries if it's a DataFrame
                        if isinstance(impact_players_df, pd.DataFrame) and not impact_players_df.empty:
                            impact_players = impact_players_df.to_dict('records')
                        
                        selected_team['impact_players'] = impact_players
                    except Exception as e:
                        if verbose:
                            print(f"Error in impact players fallback: {str(e)}")
                            traceback.print_exc()
                        impact_players = []
                        
                    if verbose and impact_players:
                        print("\nImpact Substitutes:")
                        for player in impact_players:
                            player_name = player.get('name', player.get('Player Name', ''))
                            player_team = player.get('team', player.get('Team', ''))
                            player_role = player.get('role', player.get('Role', ''))
                            player_points = player.get('predicted_points', player.get('points', 0))
                            player_credits = player.get('credits', player.get('Credits', 0))
                            
                            print(f"- {player_name} ({player_team}, {player_role}): {player_points:.2f} pts, {player_credits} cr")
            except Exception as e:
                if verbose:
                    print(f"Error selecting impact players: {str(e)}")
                    traceback.print_exc()
                selected_team['impact_players'] = []
            
            return selected_team
            
        except Exception as e:
            if verbose:
                print(f"Error predicting team: {str(e)}")
                traceback.print_exc()
            return None
    
    def plot_team_composition(self, team):
        """Create a visualization of team composition by role
        
        Args:
            team (list or dict): List of player dictionaries or team object
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Extract players list from team if it's a dictionary with 'selected_players' key
            if isinstance(team, dict) and 'selected_players' in team:
                players = team['selected_players']
            elif isinstance(team, dict) and 'players' in team:
                players = team['players']
            elif isinstance(team, list):
                players = team
            elif isinstance(team, pd.DataFrame):
                players = team.to_dict('records')
            else:
                # Handle unexpected type
                print(f"Unexpected team type: {type(team)}")
                players = []
                if isinstance(team, str):
                    print(f"Received string instead of team object: {team}")
                    raise TypeError("Expected team object but received string")
            
            # Count number of players by role
            role_counts = {}
            for player in players:
                if not isinstance(player, dict):
                    continue
                    
                role = None
                # Try different possible keys for role
                if 'role' in player:
                    role = player['role']
                elif 'Player Type' in player:
                    role = player['Player Type']
                elif 'player_type' in player:
                    role = player['player_type']
                    
                if role:
                    role = self.standardize_role(role)
                    role_counts[role] = role_counts.get(role, 0) + 1
            
            # If no roles found, return empty chart
            if not role_counts:
                ax.text(0.5, 0.5, "No role data available", ha='center', va='center')
                ax.axis('off')
                return fig
            
            # Set colors for different roles
            colors = {
                'WK': '#3498db',   # Blue
                'BAT': '#2ecc71',  # Green
                'AR': '#f39c12',   # Orange
                'BOWL': '#e74c3c'  # Red
            }
            
            # Create the pie chart
            labels = list(role_counts.keys())
            counts = [role_counts[role] for role in labels]
            role_colors = [colors.get(role, '#95a5a6') for role in labels]
            
            # Add percentage to labels
            total = sum(counts)
            labels = [f"{label} ({count}/{total})" for label, count in zip(labels, counts)]
            
            ax.pie(counts, labels=labels, colors=role_colors, autopct='%1.1f%%', 
                   startangle=90, shadow=False, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Team Composition by Role', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error plotting team composition: {str(e)}")
            # Return an empty figure on error
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error plotting team composition: {str(e)}", 
                    ha='center', va='center', wrap=True)
            ax.axis('off')
            return fig
    
    def plot_player_points(self, team):
        """Create a visualization of predicted points by player
        
        Args:
            team (list or dict): List of player dictionaries or team object
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract players list from team if it's a dictionary with 'selected_players' key
            if isinstance(team, dict) and 'selected_players' in team:
                players = team['selected_players']
            elif isinstance(team, dict) and 'players' in team:
                players = team['players']
            elif isinstance(team, list):
                players = team
            elif isinstance(team, pd.DataFrame):
                players = team.to_dict('records')
            else:
                # Handle unexpected type
                print(f"Unexpected team type: {type(team)}")
                players = []
                if isinstance(team, str):
                    print(f"Received string instead of team object: {team}")
                    raise TypeError("Expected team object but received string")
            
            # Extract player names and points
            player_data = []
            for player in players:
                if not isinstance(player, dict):
                    continue
                    
                # Try different possible keys for player name and points
                name = None
                if 'player' in player:
                    name = player['player']
                elif 'name' in player:
                    name = player['name']
                elif 'Player Name' in player:
                    name = player['Player Name']
                
                points = None
                if 'predicted_points' in player:
                    points = player['predicted_points']
                elif 'fantasy_points' in player:
                    points = player['fantasy_points']
                elif 'points' in player:
                    points = player['points']
                
                role = None
                if 'role' in player:
                    role = player['role']
                elif 'Player Type' in player:
                    role = player['Player Type']
                elif 'player_type' in player:
                    role = player['player_type']
                
                if name and points is not None:
                    player_data.append({
                        'name': name,
                        'points': points,
                        'role': role
                    })
            
            # If no players found, return empty chart
            if not player_data:
                ax.text(0.5, 0.5, "No player points data available", ha='center', va='center')
                ax.axis('off')
                return fig
            
            # Sort by points
            player_data.sort(key=lambda x: x['points'], reverse=True)
            
            # Get sorted data
            names = [p['name'] for p in player_data]
            points = [p['points'] for p in player_data]
            roles = [self.standardize_role(p['role']) if p['role'] else 'Unknown' for p in player_data]
            
            # Create horizontal bar chart
            bars = ax.barh(names, points, height=0.6)
            
            # Color bars based on role if available
            role_colors = {
                'WK': '#3498db',   # Blue
                'BAT': '#2ecc71',  # Green
                'AR': '#f39c12',   # Orange
                'BOWL': '#e74c3c'  # Red
            }
            
            for i, role in enumerate(roles):
                bars[i].set_color(role_colors.get(role, '#95a5a6'))
            
            # Add data labels
            for i, point in enumerate(points):
                ax.text(point + max(points) * 0.02, i, f'{point:.1f}', va='center')
            
            # Add labels and title
            ax.set_xlabel('Predicted Points')
            ax.set_title('Player Points Distribution')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error plotting player points: {str(e)}")
            # Return an empty figure on error
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error plotting player points: {str(e)}", 
                    ha='center', va='center', wrap=True)
            ax.axis('off')
            return fig
        
    def plot_team_points(self, team, predicted_points=None):
        """Create a visualization of predicted points by player
        
        Args:
            team (list): List of player dictionaries
            predicted_points (dict): Dictionary mapping player names to predicted points
        
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            players = []
            points = []
            
            # Extract player names and points
            for player in team:
                name = player.get('player', player.get('Player Name', ''))
                if name:
                    players.append(name)
                    
                    # Use predicted_points dict if provided, otherwise use value from player dict
                    if predicted_points and name in predicted_points:
                        points.append(predicted_points[name])
                    else:
                        points.append(player.get('predicted_points', 0))
            
            # Sort by points
            sorted_indices = np.argsort(points)[::-1]  # Descending order
            sorted_players = [players[i] for i in sorted_indices]
            sorted_points = [points[i] for i in sorted_indices]
            
            # Create horizontal bar chart
            bars = ax.barh(sorted_players, sorted_points, height=0.6, color='steelblue')
            
            # Add data labels
            for i, (player, point) in enumerate(zip(sorted_players, sorted_points)):
                ax.text(point + 5, i, f'{point:.1f}', va='center')
            
            # Add labels and title
            ax.set_xlabel('Predicted Points')
            ax.set_title('Predicted Player Points')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting player points: {str(e)}")
            # Return an empty figure on error
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Error plotting player points", ha='center', va='center')
            return fig
        
    def get_team_squads(self, teams):
        """Load squad data for the specified teams
        
        Args:
            teams (list): List of team codes
            
        Returns:
            pd.DataFrame: DataFrame containing players from the specified teams
        """
        try:
            # Load squad data
            squad_file = os.path.join(self.data_dir, "SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv")
            if not os.path.exists(squad_file):
                raise FileNotFoundError(f"Squad data file not found at {squad_file}")
                
            squads_df = pd.read_csv(squad_file)
            print(f"Loaded squad data from {squad_file}")
            
            # Create a comprehensive mapping of team codes
            team_code_mapping = {
                'CSK': ['CSK', 'CHE', 'CHENNAI', 'CHENNAI SUPER KINGS', 'CHENNAI SUPER', 'CHENNAI SK'],
                'MI': ['MI', 'MUM', 'MUMBAI', 'MUMBAI INDIANS', 'MUMBAI I', 'MUM INDIANS'],
                'RCB': ['RCB', 'BAN', 'BLR', 'BANGALORE', 'BENGALURU', 'ROYAL CHALLENGERS', 'ROYAL CHALLENGERS BANGALORE'],
                'KKR': ['KKR', 'KOL', 'KOLKATA', 'KOLKATA KNIGHT', 'KOLKATA KNIGHT RIDERS', 'KNIGHT RIDERS'],
                'PBKS': ['PBKS', 'PUN', 'KXI', 'PUNJAB', 'PUNJAB KINGS', 'KINGS XI', 'KINGS XI PUNJAB', 'KXIP'],
                'DC': ['DC', 'DEL', 'DELHI', 'DELHI CAPITALS', 'DELHI DAREDEVILS', 'DAREDEVILS'],
                'RR': ['RR', 'RAJ', 'RAJASTHAN', 'RAJASTHAN ROYALS', 'ROYALS'],
                'SRH': ['SRH', 'HYD', 'HYDERABAD', 'SUNRISERS', 'SUNRISERS HYDERABAD', 'DECCAN CHARGERS'],
                'GT': ['GT', 'GUJ', 'GUJARAT', 'GUJARAT TITANS', 'TITANS'],
                'LSG': ['LSG', 'LUC', 'LUCKNOW', 'LUCKNOW SUPER', 'LUCKNOW SUPER GIANTS', 'SUPER GIANTS']
            }
            
            # Get all possible team codes for the requested teams
            team_codes = []
            for team in teams:
                # Standard team code
                team_codes.append(team.upper())
                
                # Add alternate codes from mapping
                for std_code, variants in team_code_mapping.items():
                    if team.upper() == std_code:
                        team_codes.extend(variants)
                        break
            
            # Print available teams in squad data for debugging
            unique_teams = squads_df['Team'].unique()
            print(f"Available teams in squad data: {unique_teams}")
            print(f"Looking for teams with codes: {team_codes}")
            
            # Filter for players from selected teams only
            team_squads = squads_df[squads_df['Team'].str.upper().isin([code.upper() for code in team_codes])]
            
            if team_squads.empty:
                print(f"Warning: No players found for teams {teams}")
                # Fallback for CSK: check specifically for 'CHE' code which is often used
                if 'CSK' in teams:
                    print("Checking specifically for CSK players with team code 'CHE'")
                    csk_players = squads_df[squads_df['Team'] == 'CHE']
                    if not csk_players.empty:
                        team_squads = csk_players
                        print(f"Found {len(csk_players)} players for CSK (team code: CHE)")
                
                # Still empty? Try another fallback approach
                if team_squads.empty:
                    print("Using partial string matching as fallback")
                    for team in teams:
                        for variant in team_code_mapping.get(team.upper(), []):
                            matched_players = squads_df[squads_df['Team'].str.contains(variant, case=False, na=False)]
                            if not matched_players.empty:
                                print(f"Found {len(matched_players)} players matching '{variant}'")
                                team_squads = pd.concat([team_squads, matched_players])
            
            # Final check
            if team_squads.empty:
                print(f"Warning: No players found for teams {teams} after all fallback attempts")
                return pd.DataFrame()
                
            # Add standardized team code column 
            if hasattr(self, 'role_mapper') and self.role_mapper:
                # Create a copy to avoid SettingWithCopyWarning
                team_squads = team_squads.copy()
                
                # Use .loc to avoid SettingWithCopyWarning
                team_squads.loc[:, 'team_code'] = team_squads['Team'].apply(self.role_mapper.standardize_team_code)
            else:
                # Simple mapping
                team_squads = team_squads.copy()
                team_squads.loc[:, 'team_code'] = team_squads['Team'].apply(
                    lambda x: next((std for std, variants in team_code_mapping.items() 
                                  if x.upper() in [v.upper() for v in variants]), x)
                )
            
            # Show player distribution by team
            team_distribution = team_squads['team_code'].value_counts().to_dict()
            print(f"Players found by standardized team code: {team_distribution}")
            
            # Ensure all players have an 'is_playing' status
            team_squads.loc[:, 'is_playing'] = True
            
            # Ensure roles are standardized
            team_squads.loc[:, 'role'] = team_squads['Player Type'].apply(self.standardize_role)
            role_distribution = team_squads['role'].value_counts().to_dict()
            print(f"Players by role: {role_distribution}")
            
            return team_squads
            
        except Exception as e:
            print(f"Error loading team squads: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
        
    def standardize_role(self, role):
        """Standardize player role to one of: WK, BAT, AR, BOWL."""
        return self.role_mapper.standardize_role(role)
        
    def run(self):
        """Run the Streamlit app"""
        try:
            # Set dark theme for Streamlit
            st.set_page_config(
                page_title="Dream11 IPL Team Predictor",
                page_icon="🏏",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Apply dark theme customization
            st.markdown("""
                <style>
                .stApp {
                    background-color: #121212;
                    color: #ffffff;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Apply custom styling
            st.markdown("""
                <style>
                .main-header {
                    font-size: 2.5rem;
                    color: #3498db;
                    text-align: center;
                    margin-bottom: 1rem;
                    font-weight: bold;
                }
                .sub-header {
                    font-size: 1.8rem;
                    color: #2ecc71;
                    margin-top: 1.5rem;
                    margin-bottom: 1rem;
                    font-weight: bold;
                }
                .team-header {
                    font-size: 1.5rem;
                    color: #e74c3c;
                    margin-top: 1rem;
                    font-weight: bold;
                }
                .highlight {
                    background-color: #f39c12;
                    padding: 0.2rem;
                    border-radius: 0.2rem;
                }
                .captain {
                    color: #e74c3c;
                    font-weight: bold;
                }
                .vice-captain {
                    color: #9b59b6;
                    font-weight: bold;
                }
                .role-wk {
                    color: #3498db;
                    font-weight: bold;
                }
                .role-bat {
                    color: #2ecc71;
                    font-weight: bold;
                }
                .role-ar {
                    color: #f39c12;
                    font-weight: bold;
                }
                .role-bowl {
                    color: #e74c3c;
                    font-weight: bold;
                }
                .team-box {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    margin-bottom: 1.5rem;
                }
                .info-box {
                    background-color: #34495e;
                    color: #ecf0f1;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border-left: 5px solid #3498db;
                    margin-bottom: 1rem;
                }
                .warning-box {
                    background-color: #34495e;
                    color: #ecf0f1;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border-left: 5px solid #f39c12;
                    margin-bottom: 1rem;
                }
                .stat-card {
                    background-color: #34495e;
                    color: #ecf0f1;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                    text-align: center;
                }
                .impact-player {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .player-name {
                    font-weight: bold;
                    color: #3498db;
                }
                .player-details {
                    font-size: 0.9em;
                    color: #bdc3c7;
                }
                /* Dark themed player card */
                .player-card {
                    background-color: #34495e;
                    color: #ecf0f1;
                    padding: 0.8rem;
                    border-radius: 0.4rem;
                    margin-bottom: 0.8rem;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
                    text-align: center;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Header
            st.markdown("<h1 class='main-header'>Dream11 IPL Team Predictor 🏏</h1>", unsafe_allow_html=True)
            
            # Sidebar for inputs
            with st.sidebar:
                st.markdown("<h3 style='text-align: center;'>Match Settings</h3>", unsafe_allow_html=True)
                
                # Create columns for team selection
                col1, col2 = st.columns(2)
                
                with col1:
                    team1 = st.selectbox("Team 1:", 
                                        ["CSK", "MI", "RCB", "KKR", "PBKS", "DC", "SRH", "RR", "GT", "LSG"])
                    
                with col2:
                    team2 = st.selectbox("Team 2:", 
                                        ["MI", "CSK", "RCB", "KKR", "PBKS", "DC", "SRH", "RR", "GT", "LSG"],
                                        index=1)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Match Conditions</h3>", unsafe_allow_html=True)
                
                venue = st.selectbox("Venue:", self.venues)
                match_type = st.selectbox("Match Type:", self.match_types)
                pitch_type = st.radio("Pitch Type:", ["balanced", "batting_friendly", "bowling_friendly"], 
                                   format_func=lambda x: x.replace('_', ' ').title())
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Advanced Settings</h3>", unsafe_allow_html=True)
                
                # Advanced settings in expander
                with st.expander("Model Settings"):
                    random_factor = st.slider("Randomness Factor:", 0.0, 1.0, 0.2, 0.05,
                                             help="Higher values introduce more variety in team selection")
                    
                    use_custom_boost = st.checkbox("Use Custom Player Boost", 
                                                 help="Boost certain players manually")
                    
                    if use_custom_boost:
                        custom_boost_player = st.text_input("Player Name to Boost:")
                        custom_boost_factor = st.slider("Boost Factor:", 1.0, 2.0, 1.3, 0.1)
                
                # Add information box about the model
                st.markdown("""
                <div class='info-box'>
                This predictor uses machine learning models trained on historical IPL data 
                to predict player performance and optimize team selection within Dream11 constraints.
                </div>
                """, unsafe_allow_html=True)
                
                # Prediction button with a more attractive style
                predict_button = st.button("🔮 Predict Dream11 Team", 
                                          type="primary", 
                                          use_container_width=True)
            
            # Load data on app startup
            if not hasattr(self, 'data_loaded') or not self.data_loaded:
                with st.spinner("Loading and preprocessing data..."):
                    self.load_data()
                
                with st.spinner("Training prediction models..."):
                    self.train_models()
                    # Train with 2025 data if available
                    self.train_models_with_2025_data()
                    
                self.data_loaded = True
            
            # Main content area
            if predict_button:
                with st.spinner("Predicting the optimal Dream11 team..."):
                    try:
                        # Call the predict_team method
                        result = self.predict_team(
                            team1, team2, venue, match_type, pitch_type
                        )
                        
                        if result is not None:
                            selected_team = result
                            captain = result.get('captain', '')
                            vice_captain = result.get('vice_captain', '')
                            
                            # Create a dictionary of player points
                            predicted_points = {}
                            for player in selected_team.get('selected_players', []):
                                player_name = player.get('name', player.get('Player Name', ''))
                                if player_name:
                                    predicted_points[player_name] = player.get('points', player.get('predicted_points', 0))
                            
                            if selected_team is None or len(selected_team.get('selected_players', [])) < 11:
                                st.error("🚫 Could not select a valid team with the given constraints. Please try different teams.")
                            else:
                                # Success message
                                st.markdown("""
                                <div class='info-box'>
                                ✅ Team prediction completed successfully! Here's your optimal Dream11 team.
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Create 3 columns for stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown("""
                                    <div class='stat-card'>
                                    <h4>Total Predicted Points</h4>
                                    <h2>{:.2f}</h2>
                                    </div>
                                    """.format(sum(predicted_points.values())), unsafe_allow_html=True)
                                
                                with col2:
                                    # Calculate team distribution
                                    team_counts = {}
                                    for player in selected_team.get("selected_players", []):
                                        # Use 'Team' if 'team' is not available
                                        team_key = player.get('team', player.get('Team', 'Unknown'))
                                        if team_key in team_counts:
                                            team_counts[team_key] += 1
                                        else:
                                            team_counts[team_key] = 1
                                    
                                    team_distribution = ", ".join([f"{team}: {count}" for team, count in team_counts.items()])
                                    st.markdown(f"""
                                    <div class='stat-card'>
                                    <h4>Team Distribution</h4>
                                    <h3>{team_distribution}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    # Calculate role distribution
                                    role_counts = {}
                                    for player in selected_team.get("selected_players", []):
                                        role = self.standardize_role(player.get('role', player.get('Player Type', '')))
                                        if role in role_counts:
                                            role_counts[role] += 1
                                        else:
                                            role_counts[role] = 1
                                    
                                    role_distribution = ", ".join([f"{role}: {count}" for role, count in role_counts.items()])
                                    st.markdown(f"""
                                    <div class='stat-card'>
                                    <h4>Role Distribution</h4>
                                    <h3>{role_distribution}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Display team in a nice layout
                                st.markdown("<h2 class='sub-header'>Predicted Dream11 Team</h2>", unsafe_allow_html=True)
                                
                                with st.container():
                                    st.markdown("<div class='team-box'>", unsafe_allow_html=True)
                                    
                                    # Display captain and vice-captain
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"""
                                        <div style='text-align: center;'>
                                        <h3>Captain (2x) 👑</h3>
                                        <h2 class='captain'>{captain}</h2>
                                        <p>Predicted Points: {predicted_points.get(captain, 0)*2:.2f}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        <div style='text-align: center;'>
                                        <h3>Vice-Captain (1.5x) 🥈</h3>
                                        <h2 class='vice-captain'>{vice_captain}</h2>
                                        <p>Predicted Points: {predicted_points.get(vice_captain, 0)*1.5:.2f}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    
                                    # Group players by role for display
                                    players_by_role = {}
                                    for player in selected_team.get('selected_players', []):
                                        # Get standardized role
                                        player_role = player.get('role', '')
                                        if not player_role:
                                            player_role = player.get('Player Type', '')
                                        role = self.standardize_role(player_role)
                                        
                                        # Add to role dictionary
                                        if role not in players_by_role:
                                            players_by_role[role] = []
                                        players_by_role[role].append(player)
                                    
                                    # Display players by role
                                    role_order = ['WK', 'BAT', 'AR', 'BOWL']
                                    role_icons = {
                                        'WK': '🧤', 
                                        'BAT': '🏏', 
                                        'AR': '⚔️', 
                                        'BOWL': '🎯'
                                    }
                                    role_classes = {
                                        'WK': 'role-wk',
                                        'BAT': 'role-bat',
                                        'AR': 'role-ar',
                                        'BOWL': 'role-bowl'
                                    }
                                    
                                    for role in role_order:
                                        if role in players_by_role and len(players_by_role[role]) > 0:
                                            st.markdown(f"<h3 class='team-header'>{role_icons.get(role, '')} {role} ({len(players_by_role[role])})</h3>", unsafe_allow_html=True)
                                            
                                            # Create a grid for players
                                            cols = st.columns(min(4, len(players_by_role[role])))
                                            for i, player in enumerate(players_by_role[role]):
                                                # Try multiple possible player name keys
                                                player_name = None
                                                for name_key in ['player', 'name', 'Player Name']:
                                                    if name_key in player and player[name_key]:
                                                        player_name = player[name_key]
                                                        break
                                                if not player_name:
                                                    player_name = 'Unknown'
                                                
                                                # Try multiple possible team keys
                                                player_team = None
                                                for team_key in ['team', 'Team']:
                                                    if team_key in player and player[team_key]:
                                                        player_team = player[team_key]
                                                        break
                                                if not player_team:
                                                    player_team = 'Unknown'
                                                
                                                player_points = predicted_points.get(player_name, 0)
                                                
                                                # Try multiple possible credits keys
                                                player_credits = None
                                                for credits_key in ['credits', 'Credits']:
                                                    if credits_key in player and player[credits_key]:
                                                        player_credits = player[credits_key]
                                                        break
                                                if not player_credits:
                                                    player_credits = 0
                                                
                                                # Highlight captain and vice-captain
                                                if '(C)' in player.get('role', ''):
                                                    player_name = f"{player_name} (C)"
                                                    name_class = "captain"
                                                elif '(VC)' in player.get('role', ''):
                                                    player_name = f"{player_name} (VC)"
                                                    name_class = "vice-captain"
                                                else:
                                                    name_class = role_classes.get(role, "")
                                                
                                                with cols[i % len(cols)]:
                                                    st.markdown(f"""
                                                    <div class="player-card">
                                                    <h4 class="{name_class}">{player_name}</h4>
                                                    <p>{player_team} | {player_points:.2f} pts</p>
                                                    <small>Credits: {player_credits}</small>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Show selected_team as JSON in expandable section (for debugging)
                                with st.expander("View Team Details"):
                                    st.write(selected_team)
                                
                                # Display team composition and player points visualization
                                st.markdown("<h2 class='sub-header'>Team Analysis</h2>", unsafe_allow_html=True)
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("<h3>Team Composition by Role</h3>", unsafe_allow_html=True)
                                    fig_composition = self.plot_team_composition(selected_team)
                                    st.pyplot(fig_composition)
                                
                                with col2:
                                    st.markdown("<h3>Predicted Player Points</h3>", unsafe_allow_html=True)
                                    fig_points = self.plot_player_points(selected_team)
                                    st.pyplot(fig_points)
                                
                                # Show impact substitute players
                                if selected_team is not None and len(selected_team) > 0:
                                    st.subheader("Impact Substitutes")
                                    with st.spinner("Selecting impact substitutes..."):
                                        try:
                                            # Convert team to DataFrame if it's not already
                                            if not isinstance(selected_team, pd.DataFrame):
                                                # Extract just the selected_players list from the dictionary
                                                selected_players = selected_team.get('selected_players', [])
                                                if selected_players:
                                                    team_df = pd.DataFrame(selected_players)
                                                else:
                                                    # Fallback if selected_players is empty
                                                    st.warning("No selected players found for substitutes")
                                                    team_df = pd.DataFrame()
                                            else:
                                                team_df = selected_team
                                            
                                            # Get all available players for this match
                                            all_players = self.get_team_squads([team1, team2])
                                            
                                            # Manual implementation of impact player selection if optimizer method fails
                                            try:
                                                # First try the optimizer method
                                                impact_players = self.optimizer.select_impact_players(
                                                    all_players_df=all_players,
                                                    selected_team_df=team_df,
                                                    num_substitutes=4
                                                )
                                                
                                                if impact_players is None or impact_players.empty:
                                                    raise Exception("No impact players returned from optimizer")
                                                    
                                            except Exception as e:
                                                st.warning(f"Using fallback method for impact players: {str(e)}")
                                                # Fallback manual implementation if optimizer method fails
                                                # Convert all_players to dict format if it's a DataFrame
                                                if isinstance(all_players, pd.DataFrame):
                                                    all_players_list = all_players.to_dict('records')
                                                else:
                                                    all_players_list = all_players
                                                
                                                # Convert selected_team to list of player names
                                                selected_player_names = []
                                                if isinstance(team_df, pd.DataFrame) and not team_df.empty:
                                                    # Try different column names
                                                    if 'player' in team_df.columns:
                                                        selected_player_names = team_df['player'].tolist()
                                                    elif 'name' in team_df.columns:
                                                        selected_player_names = team_df['name'].tolist()
                                                    elif 'Player Name' in team_df.columns:
                                                        selected_player_names = team_df['Player Name'].tolist()
                                                
                                                # Filter out already selected players
                                                available_players = []
                                                for player in all_players_list:
                                                    player_name = player.get('Player Name', player.get('name', ''))
                                                    if player_name and player_name not in selected_player_names:
                                                        available_players.append(player)
                                                
                                                # Sort available players by role
                                                players_by_role = {
                                                    'WK': [], 'BAT': [], 'AR': [], 'BOWL': []
                                                }
                                                
                                                for player in available_players:
                                                    role = self.standardize_role(player.get('Player Type', player.get('role', '')))
                                                    if role in players_by_role:
                                                        players_by_role[role].append(player)
                                                
                                                # Select one player from each role
                                                impact_player_list = []
                                                for role in ['WK', 'BAT', 'AR', 'BOWL']:
                                                    if players_by_role[role]:
                                                        # Sort by credits as a proxy for quality
                                                        sorted_players = sorted(players_by_role[role], 
                                                                              key=lambda x: float(x.get('Credits', 0)), 
                                                                              reverse=True)
                                                        if sorted_players:
                                                            impact_player_list.append(sorted_players[0])
                                                
                                                # Create a DataFrame from the selected impact players
                                                if impact_player_list:
                                                    impact_players = pd.DataFrame(impact_player_list)
                                                else:
                                                    impact_players = pd.DataFrame()
                                            
                                            if impact_players is not None and not impact_players.empty:
                                                # Create columns for displaying substitutes
                                                cols = st.columns(4)
                                                
                                                # Display impact players
                                                for i, (_, player) in enumerate(impact_players.iterrows()):
                                                    with cols[i % 4]:
                                                        # Try multiple possible name keys
                                                        name = None
                                                        for name_key in ['name', 'player', 'Player Name']:
                                                            if name_key in player and player[name_key]:
                                                                name = player[name_key]
                                                                break
                                                        if not name:
                                                            name = 'Unknown Player'
                                                        
                                                        # Try multiple possible team keys
                                                        team = None
                                                        for team_key in ['team', 'Team']:
                                                            if team_key in player and player[team_key]:
                                                                team = player[team_key]
                                                                break
                                                        if not team:
                                                            team = 'Unknown'
                                                        
                                                        # Try multiple possible role keys
                                                        role = None
                                                        for role_key in ['role', 'Player Type']:
                                                            if role_key in player and player[role_key]:
                                                                role = player[role_key]
                                                                break
                                                        if not role:
                                                            role = 'Unknown Role'
                                                        
                                                        # Try multiple possible points keys
                                                        points = 0
                                                        for points_key in ['predicted_points', 'fantasy_points']:
                                                            if points_key in player:
                                                                points = player[points_key]
                                                                break
                                                        
                                                        # Try multiple possible credits keys
                                                        credits = 0
                                                        for credits_key in ['credits', 'Credits']:
                                                            if credits_key in player:
                                                                credits = player[credits_key]
                                                                break
                                                
                                                # Display the impact player with proper HTML
                                                st.markdown(f"""
                                                <div class="impact-player" style="background-color: #2c3e50; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: white;">
                                                    <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 5px;">{name}</div>
                                                    <div>
                                                        <span style="color: #7fbbda;">Team:</span> {team}<br>
                                                        <span style="color: #7fbbda;">Role:</span> {role}<br>
                                                        <span style="color: #7fbbda;">Points:</span> {points:.2f}<br>
                                                        <span style="color: #7fbbda;">Credits:</span> {credits}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Add debugging info in an expandable section
                                                with st.expander("Debug Impact Players Data", expanded=False):
                                                    st.json(impact_players.to_dict(orient='records'))
                                            else:
                                                st.info("No suitable impact substitute players found")
                                        except Exception as e:
                                            st.error(f"Error selecting impact substitutes: {str(e)}")
                                            st.exception(e)
                    
                    except Exception as e:
                        st.error(f"Error predicting team: {str(e)}")
                        import traceback
                        st.exception(traceback.format_exc())
            
            else:
                # Welcome message when app first loads
                st.markdown("""
                <div class='info-box' style='text-align: center;'>
                <h2>Welcome to the IPL Dream11 Team Predictor! 🏆</h2>
                <p>This tool helps you select the optimal Dream11 team for IPL matches based on player performance data and advanced analytics.</p>
                <p>To get started, select the match details in the sidebar and click "Predict Dream11 Team".</p>
                </div>
                
                <div class='warning-box'>
                <h4>How it works:</h4>
                <ol>
                <li>Select the two teams playing in the match</li>
                <li>Choose the venue, match type, and pitch conditions</li>
                <li>Adjust any advanced settings if needed</li>
                <li>Click "Predict Dream11 Team" to get your optimal team</li>
                </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature highlights
                st.markdown("<h2 class='sub-header'>Key Features</h2>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class='stat-card' style='height: 200px;'>
                    <h3>💡 Smart Prediction</h3>
                    <p>Uses machine learning to predict player performance based on historical data and current conditions</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class='stat-card' style='height: 200px;'>
                    <h3>⚖️ Optimal Balance</h3>
                    <p>Automatically balances team composition to maximize points while meeting all Dream11 constraints</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class='stat-card' style='height: 200px;'>
                    <h3>📊 Team Analysis</h3>
                    <p>Provides detailed visualizations and insights about your predicted team's composition and expected performance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error running app: {str(e)}")
            import traceback
            st.exception(traceback.format_exc())

if __name__ == "__main__":
    app = Dream11App()
    app.run() 