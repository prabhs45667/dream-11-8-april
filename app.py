import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import pickle
import traceback
import logging
from team_optimizer import TeamOptimizer
from model_integration import Dream11ModelIntegrator
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from data_preprocessor import DataPreprocessor
from role_mapper import RoleMapper
from sklearn.ensemble import RandomForestRegressor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dream11App:
    def __init__(self, data_dir="dataset", use_tensorflow=False):
        """Initialize Dream11App"""
        logger.info("Initializing Dream11App...")
        self.data_dir = data_dir
        self.use_tensorflow = use_tensorflow
        self.data_loaded = False
        self.deliveries_df = None
        self.team_squads = None
        self.data_preprocessor = DataPreprocessor(data_dir)
        self.optimizer = TeamOptimizer()
        self.role_mapper = RoleMapper()
        self.pitch_types = ['balanced', 'batting_friendly', 'bowling_friendly']
        try:
            self.model_integrator = Dream11ModelIntegrator()
            # Get actual pitch types supported by loaded models
            loaded_pitch_types = self.model_integrator.get_available_pitch_types()
            if loaded_pitch_types: # Check if list is not empty
                self.pitch_types = loaded_pitch_types
            logger.info(f"ModelIntegrator initialized successfully. Supported pitch types: {self.pitch_types}")
        except Exception as e:
            logger.error(f"Failed to initialize ModelIntegrator: {e}", exc_info=True)
            # Keep self.model_integrator as None - app will show error later

        # Define team codes and venues
        self.available_teams = self._get_available_teams()
        self.venues = ["M. Chinnaswamy Stadium, Bangalore", "Eden Gardens, Kolkata", 
                     "Wankhede Stadium, Mumbai", "MA Chidambaram Stadium, Chennai",
                     'Narendra Modi Stadium, Ahmedabad', "Arun Jaitley Stadium, Delhi"]
        self.match_types = ["League", "Qualifier", "Eliminator", "Final"]
        
        # Load initial data (like squad file)
        try:
            self.load_data()
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}", exc_info=True)
            st.warning("Could not load initial squad data.")
        logger.info("Dream11App initialization complete.")
        
    def _get_available_teams(self):
        """Helper to get team codes from the squad file."""
        squad_file = os.path.join(self.data_dir, "SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv")
        default_teams = ['CHE', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR', 'SRH']
        try:
            if os.path.exists(squad_file):
                squad_df = pd.read_csv(squad_file)
                # Try multiple common column names for team codes
                for col_name in ['Team Code', 'TeamCode', 'team_code', 'Team', 'team']:
                     if col_name in squad_df.columns:
                          # Standardize team codes to uppercase and remove duplicates
                          teams = sorted(list(squad_df[col_name].astype(str).str.upper().unique()))
                          if teams:
                              logger.info(f"Found teams {teams} in column '{col_name}' of {squad_file}")
                              return teams
                logger.warning(f"Could not find a suitable team column in {squad_file}. Using default teams.")
            else:
                 logger.warning(f"Squad file not found at {squad_file}. Using default teams.")
            return default_teams # Default if file missing/no teams/no suitable column
        except Exception as e:
            logger.error(f"Error reading team codes from squad file {squad_file}: {e}")
            return default_teams # Default on error
    
    def load_data(self):
        """Load and preprocess the data (mainly ensures squad data is available)"""
        logger.info("Loading data...")
        try:
            # Load squad data if not already loaded
            if self.team_squads is None:
                squad_file = os.path.join(self.data_dir, "SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv")
                if os.path.exists(squad_file):
                    self.team_squads = pd.read_csv(squad_file)
                    logger.info(f"Loaded squad data from {squad_file}: {len(self.team_squads)} records")
                    self.data_loaded = True # Consider data loaded if squad is present
                else:
                    logger.error(f"Squad data file not found: {squad_file}")
                    self.data_loaded = False
                    return False
            else:
                 logger.info("Squad data already loaded.")
                 self.data_loaded = True

            # Load deliveries data if needed (optional for prediction, but useful)
            if self.deliveries_df is None:
                 file_path_2025 = os.path.join(self.data_dir, 'ipl_2025_deliveries.csv')
                 if os.path.exists(file_path_2025):
                     self.deliveries_df = pd.read_csv(file_path_2025)
                     logger.info(f"Loaded 2025 deliveries data from {file_path_2025} with {len(self.deliveries_df)} deliveries")
                 else:
                     logger.warning(f"No deliveries data found at {file_path_2025}")

            return self.data_loaded
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            self.data_loaded = False
            return False
            
    def train_models(self):
        """Train prediction models"""
        try:
            if not self.data_loaded:
                self.load_data()
                
            results = self.model_integrator.train_models()
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
    
    def _load_and_check_lineup_data(self, home_team, away_team):
        """Loads squad data for the two teams and performs basic checks."""
        try:
            # Get team squads for both teams
            team_squads = self.get_team_squads([home_team, away_team])
            
            # Check if team_squads are available
            if team_squads is None or team_squads.empty:
                print(f"No squad data found for teams {home_team} and {away_team}")
                return None
            
            # Return squads data
            logger.info(f"Loaded and checked lineup data for {home_team} vs {away_team}")
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
                    predicted_points = self.model_integrator.predict_player_points(squad_data)
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
            
    def predict_team(self, home_team, away_team, venue=None, match_type=None, pitch_type='balanced', verbose=True):
        """Predict the best Dream11 team for a match using ModelIntegrator."""
        logger.info(f"Predicting team for {home_team} vs {away_team} | Pitch: {pitch_type}")

        try:
            # 1. Load lineup data for both teams
            team_squads = self._load_and_check_lineup_data(home_team, away_team)
            if team_squads is None or team_squads.empty:
                logger.error("Failed to load lineup data.")
                st.error("Could not load player/squad data for the selected teams. Please check the dataset folder.")
                return None

            # 2. Ensure necessary columns exist
            required_cols = ['Player Name', 'Team', 'Player Type', 'Credits']
            missing_cols = [col for col in required_cols if col not in team_squads.columns]
            if missing_cols:
                logger.error(f"Squad data missing required columns: {missing_cols}")
                st.error(f"Squad data is missing essential columns: {', '.join(missing_cols)}. Cannot proceed.")
                return None

            # 3. Standardize player roles *before* prediction
            team_squads['Player Type'] = team_squads['Player Type'].astype(str)
            team_squads['Role'] = team_squads['Player Type'].apply(self.standardize_role)
            logger.info(f"Standardized roles for {len(team_squads)} players.")

            # 4. Predict player points using ModelIntegrator
            predicted_points_df = None
            if self.model_integrator:
                try:
                    logger.info(f"Attempting prediction using ModelIntegrator for pitch: {pitch_type}")
                    # Pass the original squad df, integrator handles feature eng.
                    predicted_points_df = self.model_integrator.predict_player_points(
                        player_data=team_squads.copy(), # Pass a copy
                        pitch_type=pitch_type
                    )
                    # Validate the output
                    if predicted_points_df is not None and not predicted_points_df.empty:
                         if 'Player Name' not in predicted_points_df.columns or 'predicted_points' not in predicted_points_df.columns:
                              logger.error("ModelIntegrator output missing required columns (\'Player Name\', \'predicted_points\').")
                              predicted_points_df = None # Invalidate bad output
                         else:
                              logger.info(f"ModelIntegrator returned {len(predicted_points_df)} predictions.")
                    else:
                         logger.warning("ModelIntegrator returned None or empty DataFrame.")
                         predicted_points_df = None # Ensure it's None if empty
                except Exception as e:
                    logger.error(f"Error predicting points with ModelIntegrator: {e}", exc_info=True)
                    st.warning(f"Model prediction failed ({e}). Falling back to credit-based points.")
                    predicted_points_df = None # Ensure fallback on error
            else:
                logger.warning("ModelIntegrator not available. Using credit-based points.")
                st.warning("Model Integrator not loaded. Using basic credit-based points.")
                predicted_points_df = None # Explicitly set to None

            # 5. Merge predicted points or use fallback
            if predicted_points_df is not None:
                 logger.info("Merging model predictions with squad data.")
                 team_squads = pd.merge(
                      team_squads,
                      predicted_points_df[['Player Name', 'predicted_points']],
                      on='Player Name',
                      how='left'
                 )
                 missing_predictions = team_squads['predicted_points'].isnull().sum()
                 if missing_predictions > 0:
                      logger.warning(f"{missing_predictions} players had no predicted points after merge. Applying credit-based fallback.")
                      team_squads['predicted_points'] = team_squads['predicted_points'].fillna(team_squads['Credits'] * 5)
                 team_squads['predicted_points'] = pd.to_numeric(team_squads['predicted_points'], errors='coerce').fillna(0)
            else:
                logger.warning("Using credit-based points as primary source for all players.")
                team_squads['predicted_points'] = team_squads['Credits'] * 10
                team_squads['predicted_points'] = pd.to_numeric(team_squads['predicted_points'], errors='coerce').fillna(0)

            team_squads['Credits'] = pd.to_numeric(team_squads['Credits'], errors='coerce').fillna(8.0)

            if verbose:
                 st.write("Top 10 Players by Predicted Points (after merge/fallback):")
                 st.dataframe(team_squads[['Player Name', 'Team', 'Role', 'Credits', 'predicted_points']].sort_values('predicted_points', ascending=False).head(10))

            # 6. Optimization Step
            role_counts = team_squads['Role'].value_counts().to_dict()
            required_roles_check = ['WK', 'BAT', 'AR', 'BOWL']
            missing_roles = [role for role in required_roles_check if role not in role_counts or role_counts[role] == 0]

            if missing_roles:
                logger.warning(f"Squad data missing players for required roles: {missing_roles}")
                st.warning(f"Warning: Not enough players found for roles: {', '.join(missing_roles)}. Team selection might be suboptimal.")

            role_requirements = {'WK': (1, 4), 'BAT': (3, 6), 'AR': (1, 4), 'BOWL': (3, 6)}

            players_dict = {}
            for _, player in team_squads.iterrows():
                 if pd.isna(player['Credits']) or pd.isna(player['predicted_points']):
                      logger.warning(f"Skipping player {player['Player Name']} due to NaN credits/points.")
                      continue
                 players_dict[player['Player Name']] = {
                    'name': player['Player Name'],
                    'team': player['Team'],
                    'role': player['Role'],
                    'credits': float(player['Credits']),
                    'points': float(player['predicted_points'])
                 }

            if not players_dict:
                logger.error("No valid players available for optimization after processing.")
                st.error("No valid players found for team selection. Check data quality.")
                return None

            problem = {
                'players': players_dict,
                'role_requirements': role_requirements,
                'max_credits': 100,
                'team_ratio_limit': 7
            }

            selected_team_result = None
            logger.info("Attempting team optimization...")
            try:
                if not self.optimizer:
                     logger.error("TeamOptimizer not initialized.")
                     raise ValueError("Optimizer not available.")
                selected_team_result = self.optimizer.solve_optimization_problem(problem)
                if selected_team_result is None or not selected_team_result.get('selected_players'):
                     logger.warning("Optimization failed or returned empty team. Trying greedy fallback.")
                     selected_team_result = self._fallback_greedy_selection(team_squads, role_requirements)
                     if selected_team_result is None:
                          logger.error("Greedy fallback also failed.")
                          st.error("Both optimization and fallback methods failed to select a team.")
                          return None
                     else:
                          logger.info("Greedy fallback selection successful.")
                else:
                     logger.info("Optimization successful.")
            except Exception as e:
                logger.error(f"Error during team optimization: {e}", exc_info=True)
                st.error(f"Optimization error: {e}. Trying greedy fallback.")
                selected_team_result = self._fallback_greedy_selection(team_squads, role_requirements)
                if selected_team_result is None:
                     logger.error("Greedy fallback failed after optimization error.")
                     st.error("Both optimization and fallback methods failed to select a team.")
                     return None
                else:
                     logger.info("Greedy fallback successful after optimization error.")

            if selected_team_result is None:
                 logger.critical("Selected team result is None after all attempts.")
                 return None

            # 7. Post-processing: Finalize Captain/Vice-Captain
            captain, vice_captain = None, None
            if self.model_integrator and hasattr(self.model_integrator, 'get_captain_vice_captain'):
                try:
                    selected_players_list_for_cvc = selected_team_result.get('selected_players', [])
                    if selected_players_list_for_cvc:
                         captain, vice_captain = self.model_integrator.get_captain_vice_captain(selected_players_list_for_cvc, pitch_type)
                         if captain and vice_captain:
                              logger.info(f"Integrator suggested C: {captain}, VC: {vice_captain}")
                         else:
                              logger.warning("Integrator did not return valid C/VC suggestions.")
                    else:
                        logger.warning("No selected players to pass for C/VC suggestion.")
                except Exception as e:
                     logger.warning(f"Could not get C/VC suggestions from integrator: {e}")
                     captain, vice_captain = None, None

            if 'selected_players' in selected_team_result and selected_team_result['selected_players']:
                 player_list = sorted(selected_team_result['selected_players'], key=lambda p: p.get('points', 0), reverse=True)
                 final_captain = captain if captain else (player_list[0].get('name', '') if player_list else '')
                 fallback_vc = player_list[1].get('name', '') if len(player_list) > 1 else final_captain
                 final_vice_captain = vice_captain if vice_captain and vice_captain != final_captain else fallback_vc
                 if final_vice_captain == final_captain and len(player_list) > 1:
                      final_vice_captain = player_list[1].get('name', '')
                 elif final_vice_captain == final_captain:
                      final_vice_captain = 'N/A'
                 selected_team_result['captain'] = final_captain
                 selected_team_result['vice_captain'] = final_vice_captain
                 logger.info(f"Final Captain: {selected_team_result['captain']}, Vice-Captain: {selected_team_result['vice_captain']}")
            else:
                 selected_team_result['captain'] = 'N/A'
                 selected_team_result['vice_captain'] = 'N/A'

            # 8. Select Impact Players
            selected_player_names = {p['name'] for p in selected_team_result.get('selected_players', [])}
            available_for_impact = team_squads[~team_squads['Player Name'].isin(selected_player_names)].copy()

            impact_players = []
            if not available_for_impact.empty:
                logger.info(f"Selecting impact players from {len(available_for_impact)} available candidates.")
                try:
                    if hasattr(self.optimizer, 'select_impact_players'):
                         impact_players = self.optimizer.select_impact_players(available_for_impact)
                         selected_team_result['impact_players'] = impact_players if impact_players is not None else []
                         logger.info(f"Selected {len(selected_team_result['impact_players'])} impact players.")
                    else:
                         logger.warning("Optimizer object missing 'select_impact_players' method.")
                         selected_team_result['impact_players'] = []
                except Exception as e:
                    logger.error(f"Error selecting impact players: {e}", exc_info=True)
                    st.warning(f"Could not select impact players due to an error: {e}")
                    selected_team_result['impact_players'] = []
            else:
                 logger.warning("No players available for impact player selection.")
                 selected_team_result['impact_players'] = []

            # 9. Return the final result dictionary
            logger.info("predict_team function completed successfully.")
            return selected_team_result

        except Exception as e:
            logger.error(f"Critical error in predict_team: {str(e)}", exc_info=True)
            st.error(f"An unexpected error occurred during team prediction: {str(e)}")
            return None

    def _fallback_greedy_selection(self, player_df, role_requirements):
        """Simple greedy selection based on predicted points as a fallback."""
        logger.info("Using fallback greedy team selection.")
        try:
            # Ensure points are numeric and sort
            player_df['predicted_points'] = pd.to_numeric(player_df['predicted_points'], errors='coerce').fillna(0)
            player_df['Credits'] = pd.to_numeric(player_df['Credits'], errors='coerce').fillna(8.0) # Ensure credits are numeric

            sorted_players = player_df.sort_values('predicted_points', ascending=False).to_dict('records')

            selected_players = []
            current_credits = 0
            team_counts = {}
            role_counts = {role: 0 for role in ['WK', 'BAT', 'AR', 'BOWL']}
            max_credits = 100
            max_players_from_team = 7

            for player in sorted_players:
                if len(selected_players) == 11:
                    break

                player_role = player.get('Role')
                player_team = player.get('Team')
                player_credits = player.get('Credits')
                player_name = player.get('Player Name')

                # Basic checks
                if not all([player_name, player_role, player_team, isinstance(player_credits, (int, float))]):
                    logger.warning(f"Skipping player in greedy fallback due to missing data: {player_name}")
                    continue
                player_credits = float(player_credits)

                # Check credits
                if current_credits + player_credits > max_credits:
                    continue

                # Check team limits
                if team_counts.get(player_team, 0) >= max_players_from_team:
                    continue

                # Check role counts (consider max limits)
                role_max = role_requirements.get(player_role, (0, 11))[1]
                if role_counts.get(player_role, 0) >= role_max:
                    continue

                # Add player
                selected_players.append({
                    'name': player_name,
                    'team': player_team,
                    'role': player_role,
                    'credits': player_credits,
                    'points': player['predicted_points']
                })
                current_credits += player_credits
                team_counts[player_team] = team_counts.get(player_team, 0) + 1
                role_counts[player_role] = role_counts.get(player_role, 0) + 1

            # Final check for minimum role requirements
            final_role_counts = {role: 0 for role in ['WK', 'BAT', 'AR', 'BOWL']}
            for p in selected_players:
                final_role_counts[p['role']] = final_role_counts.get(p['role'], 0) + 1

            valid_team = True
            if len(selected_players) != 11:
                logger.error(f"Greedy fallback only selected {len(selected_players)} players.")
                valid_team = False

            for role, (min_req, _) in role_requirements.items():
                if final_role_counts.get(role, 0) < min_req:
                    logger.warning(f"Greedy fallback failed minimum role requirement for {role}: Needed {min_req}, Got {final_role_counts.get(role, 0)}")
                    valid_team = False
                    # Don't break, report all unmet requirements

            if not valid_team:
                logger.error("Greedy fallback failed to form a valid 11-player team meeting all requirements.")
                st.warning("Greedy fallback could not meet all team constraints. Results may be incomplete.")
                # Return the partial team anyway, maybe better than nothing?
                # return None # Or return partial result

            # Construct result dictionary
            total_points = sum(p['points'] for p in selected_players)
            captain = selected_players[0]['name'] if selected_players else 'N/A'
            vice_captain = selected_players[1]['name'] if len(selected_players) > 1 else captain
            if captain == vice_captain and len(selected_players) > 1:
                vice_captain = selected_players[1]['name'] # Ensure different if possible
            elif captain == vice_captain:
                vice_captain = 'N/A'

            logger.info(f"Greedy fallback selection completed. Valid: {valid_team}")
            return {
                'selected_players': selected_players,
                'total_credits': current_credits,
                'total_points': total_points,
                'captain': captain,
                'vice_captain': vice_captain,
                'team_distribution': team_counts,
                'role_distribution': final_role_counts,
                'impact_players': [] # Greedy doesn't select impact players
            }
        except Exception as e:
            logger.error(f"Error in fallback greedy selection: {e}", exc_info=True)
            return None

    def plot_team_composition(self, team):
        """Create a visualization of team composition by role
        
        Args:
            team (list): List of player dictionaries
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Count number of players by role
            role_counts = {}
            for player in team:
                role = self.standardize_role(player.get('role', player.get('Player Type', '')))
                if role in role_counts:
                    role_counts[role] += 1
                else:
                    role_counts[role] = 1
            
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
            return fig
        
    def plot_player_points(self, team):
        """Create a visualization of predicted points by player
        
        Args:
            team (list): List of player dictionaries
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract player names and points
            players = []
            points = []
            for player in team:
                name = player.get('player', player.get('Player Name', ''))
                point = player.get('predicted_points', 0)
                if name:
                    players.append(name)
                    points.append(point)
            
            # Sort by points
            sorted_indices = np.argsort(points)[::-1]  # Descending order
            sorted_players = [players[i] for i in sorted_indices]
            sorted_points = [points[i] for i in sorted_indices]
            
            # Create horizontal bar chart
            bars = ax.barh(sorted_players, sorted_points, height=0.6)
            
            # Color bars based on role if available
            role_colors = {
                'WK': '#3498db',   # Blue
                'BAT': '#2ecc71',  # Green
                'AR': '#f39c12',   # Orange
                'BOWL': '#e74c3c'  # Red
            }
            
            for i, player in enumerate(sorted_indices):
                role = self.standardize_role(team[player].get('role', team[player].get('Player Type', '')))
                bars[i].set_color(role_colors.get(role, '#95a5a6'))
            
            # Add data labels
            for i, point in enumerate(sorted_points):
                ax.text(point + max(sorted_points) * 0.02, i, f'{point:.1f}', va='center')
            
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
        
    def run_app(self):
        st.set_page_config(layout="wide")
        st.title("🏏 Dream11 Team Predictor App")
        logger.info("Running Streamlit app UI...")

        # Check if Model Integrator loaded
        if not self.model_integrator:
            st.error("🚨 Critical Error: Model Integrator failed to initialize. Model-based predictions are unavailable. Please check the logs.")
            # Optionally provide more guidance or stop the app run here
            # return # Uncomment to stop if models are essential

        # --- Sidebar ---
        st.sidebar.header("Match Details")

        # Use dynamically loaded teams if available, else use default
        team_list = self.available_teams if self.available_teams else ['CHE', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR', 'SRH']
        if not team_list:
             st.sidebar.error("No teams available. Check squad data.")
             return

        # Ensure unique teams can be selected
        home_team = st.sidebar.selectbox("Select Home Team", team_list, index=0)

        # Filter away team list to exclude home team
        away_team_list = [t for t in team_list if t != home_team]
        if not away_team_list and len(team_list) > 1: # Handle case where only one team might be left
             away_team_list = [team_list[1]] # Just pick the second one
        elif not away_team_list:
             away_team_list = team_list # Fallback if only one team total

        # Try to find a different default index for away team
        away_team_default_index = 0
        if away_team_list and len(away_team_list) > away_team_default_index:
            pass # Index 0 is fine if list has elements
        elif away_team_list: # If index 0 is out of bounds but list not empty
             away_team_default_index = 0

        if away_team_list: # Check if list is not empty
             away_team = st.sidebar.selectbox("Select Away Team", away_team_list, index=away_team_default_index)
        else:
             st.sidebar.warning("Only one team available to select.")
             away_team = home_team # Or handle differently

        # Add Pitch Type Selector using types from ModelIntegrator
        selected_pitch_type = st.sidebar.selectbox(
            "Select Pitch Type",
            options=self.pitch_types if hasattr(self, 'pitch_types') and self.pitch_types else ['balanced'], # Use options parameter and provide default
            index=0, # Default to the first available pitch type
            help="Select the expected pitch condition. This affects player point predictions."
        )

        venue = st.sidebar.selectbox("Select Venue (Optional)", [""] + self.venues)
        match_type = st.sidebar.selectbox("Select Match Type (Optional)", [""] + self.match_types)

        st.sidebar.header("Actions")
        predict_button = st.sidebar.button("Predict Dream11 Team")
        # train_button = st.sidebar.button("Train Models (Optional)") # Keep or remove as needed

        # --- Main Area ---
        st.header(f"Predicting Team for {home_team} vs {away_team}")
        st.write(f"Pitch Type Selected: **{selected_pitch_type}**")
        if venue:
            st.write(f"Venue: {venue}")
        if match_type:
            st.write(f"Match Type: {match_type}")

        if predict_button:
            # Double-check integrator availability here before calling predict_team
            if not self.model_integrator:
                 st.error("Model Integrator is not available. Cannot generate predictions.")
                 return # Stop prediction if models didn't load

            if home_team == away_team:
                 st.error("Home and Away teams must be different.")
                 return

            with st.spinner(f"Predicting team for {home_team} vs {away_team} on a {selected_pitch_type} pitch..."):
                start_time = time.time()
                # Pass the selected pitch type to predict_team
                # Use verbose=False for cleaner UI output
                result = self.predict_team(home_team, away_team, venue, match_type, selected_pitch_type, verbose=False)
                end_time = time.time()
                logger.info(f"Prediction process took {end_time - start_time:.2f} seconds.")
                st.info(f"Prediction took {end_time - start_time:.2f} seconds.")

                if result:
                    # Call the new display method (to be implemented next)
                    self.display_prediction_results(result)
                else:
                    st.error("Failed to predict team. Check logs or ensure squad data is available and valid for the selected teams.")

        # --- Footer/Other Sections ---
        st.sidebar.markdown("---")
        st.sidebar.info("Dream11 Prediction Engine v2") # Updated footer

    def _standardize_columns(self, df):
        """Standardize column names to a consistent format."""
        # Implement your standardization logic here
        pass

    def display_prediction_results(self, result):
        """Displays the predicted team and impact players in Streamlit."""
        st.subheader("🏆 Predicted Dream11 Team")

        if not result or 'selected_players' not in result or not result['selected_players']:
            st.warning("No team could be selected or prediction failed.")
            return

        # --- Team Summary ---
        col1, col2, col3 = st.columns(3)
        # Recalculate points/credits/distribution directly from selected_players for consistency
        selected_players = result['selected_players']
        total_points = sum(p.get('points', 0) for p in selected_players)
        total_credits = sum(p.get('credits', 0) for p in selected_players)
        team_counts = {}
        for p in selected_players:
            # Handle potential missing team info gracefully
            team = p.get('team', 'Unknown')
            team_counts[team] = team_counts.get(team, 0) + 1
        team_dist_str = ", ".join([f"{t}: {c}" for t, c in sorted(team_counts.items())])

        with col1:
            st.metric("Total Predicted Points", f"{total_points:.2f}")
        with col2:
            st.metric("Total Credits Used", f"{total_credits:.1f} / 100")
        with col3:
            # Use markdown for better control over spacing/layout if needed
             st.markdown(f"**Team Distribution:**\
{team_dist_str}")

        # Display Captain and Vice-Captain clearly
        captain_name = result.get('captain', 'N/A')
        vice_captain_name = result.get('vice_captain', 'N/A')
        st.markdown(f"**Captain (C):** {captain_name}")
        st.markdown(f"**Vice-Captain (VC):** {vice_captain_name}")
        st.markdown("---")

        # --- Player Display --- #
        st.markdown("#### Selected Team Players")
        players_by_role = {'WK': [], 'BAT': [], 'AR': [], 'BOWL': []}
        role_mapping_display = {'WK':'Wicket Keepers', 'BAT':'Batters', 'AR':'All Rounders', 'BOWL':'Bowlers'}

        for player in selected_players:
            role = player.get('role', 'Unknown')
            if role not in players_by_role:
                 # Handle unexpected roles found during processing
                 logger.warning(f"Unexpected player role '{role}' encountered in display for player {player.get('name')}")
                 if 'Unknown' not in players_by_role: players_by_role['Unknown'] = []
                 players_by_role['Unknown'].append(player)
            else:
                 players_by_role[role].append(player)

        # Use columns for better layout - Adjust number based on expected roles with players
        active_roles = [r for r in role_mapping_display if r in players_by_role and players_by_role[r]]
        if 'Unknown' in players_by_role and players_by_role['Unknown']:
             active_roles.append('Unknown')

        if not active_roles:
            st.warning("No players found in standard roles to display.")
            return

        cols = st.columns(len(active_roles))
        role_order = ['WK', 'BAT', 'AR', 'BOWL', 'Unknown'] # Desired display order including Unknown

        col_idx = 0
        for role in role_order:
            if role in players_by_role and players_by_role[role]:
                role_display_name = role_mapping_display.get(role, role) # Use friendly name or role code
                with cols[col_idx]:
                    st.markdown(f"**{role_display_name} ({len(players_by_role[role])})**")
                    # Sort players within role by points (descending)
                    for player in sorted(players_by_role[role], key=lambda p: p.get('points', 0), reverse=True):
                        name = player.get('name', 'Unknown Player')
                        team = player.get('team', 'N/A')
                        points = player.get('points', 0)
                        credits_ = player.get('credits', 0)
                        marker = ""
                        if name == captain_name:
                            # Use HTML with distinct styling for C/VC
                            marker = " <span style='color: #FFD700; background-color: #4a4a4a; padding: 1px 4px; border-radius: 3px; font-weight: bold; font-size: 0.8em;'>C</span>"
                        elif name == vice_captain_name:
                            marker = " <span style='color: #C0C0C0; background-color: #4a4a4a; padding: 1px 4px; border-radius: 3px; font-weight: bold; font-size: 0.8em;'>VC</span>"

                        # Display player info with points and credits in a card-like format
                        st.markdown(
                            f"<div style='margin-bottom: 6px; padding: 6px 8px; border-radius: 5px; background-color: #f0f2f6; border-left: 3px solid #6c757d;'>"
                            f"<strong style='font-size: 0.95em; display: inline-block; margin-right: 5px;'>{name}</strong>{marker}<br>"
                            f"<small style='color: #555;'>Team: {team} | Pts: <strong>{points:.1f}</strong> | Cr: {credits_:.1f}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                col_idx += 1 # Move to the next column only if the current one was used

        # --- Impact Players --- #
        st.markdown("---")
        st.markdown("#### 🔄 Impact Player Suggestions")
        impact_players = result.get('impact_players', [])
        if impact_players:
             # Determine number of columns based on number of players (max 4)
             num_impact_players = len(impact_players)
             num_cols = min(num_impact_players, 4)
             if num_cols <= 0 : num_cols = 1 # Ensure at least one column if players exist

             impact_cols = st.columns(num_cols)
             # Sort impact players by predicted points
             sorted_impact_players = sorted(impact_players, key=lambda p: p.get('predicted_points', p.get('points', 0)), reverse=True)

             for i, player in enumerate(sorted_impact_players):
                 col_index = i % num_cols
                 with impact_cols[col_index]:
                     # Extract details safely
                     name = player.get('name', player.get('Player Name', 'Unknown'))
                     role = self.standardize_role(player.get('role', player.get('Role', 'N/A'))) # Standardize role for display consistency
                     team = player.get('team', player.get('Team', 'N/A'))
                     points = player.get('predicted_points', player.get('points', 0))
                     credits_ = player.get('credits', player.get('Credits', 0))

                     # Display using similar card styling
                     st.markdown(
                         f"<div style='margin-bottom: 6px; padding: 6px 8px; border-radius: 5px; background-color: #e8f0fe; border-left: 3px solid #007bff;'>"
                         f"<strong style='font-size: 0.9em;'>{name}</strong> <span style='font-size: 0.8em;'>({role}, {team})</span><br>"
                         f"<small style='color: #555;'>Pts: <strong>{points:.1f}</strong> | Cr: {credits_:.1f}</small>"
                         f"</div>",
                         unsafe_allow_html=True
                     )
        else:
             st.info("No specific impact players were selected or recommended based on current criteria.")


if __name__ == "__main__":
    app = Dream11App()
    app.run_app() 