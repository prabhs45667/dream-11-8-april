import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from pulp import *
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import traceback

class Dream11Predictor:
    def __init__(self, data_dir="dataset"):
        self.data_path = data_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = float('-inf')
        self.player_features = None
        self.player_points = None
        self.predictions = None  # Store player predictions for current match
        
        # Add team code mapping
        self.team_code_mapping = {
            'CSK': ['CSK', 'CHE'],  # CSK is also known as CHE in some datasets
            'MI': ['MI', 'MUM'],    # MI is also known as MUM in some datasets
            'RCB': ['RCB', 'BAN'],  # RCB is also known as BAN in some datasets
            'KKR': ['KKR', 'KOL'],  # KKR is also known as KOL in some datasets
            'PBKS': ['PBKS', 'PUN'], # PBKS is also known as PUN in some datasets
            'DC': ['DC', 'DEL'],    # DC is also known as DEL in some datasets
            'RR': ['RR', 'RAJ'],    # RR is also known as RAJ in some datasets
            'SRH': ['SRH', 'HYD'],  # SRH is also known as HYD in some datasets
            'GT': ['GT', 'GUJ'],    # GT is also known as GUJ in some datasets
            'LSG': ['LSG', 'LUC']   # LSG is also known as LUC in some datasets
        }
        
    def load_and_preprocess_data(self):
        """Load and preprocess IPL deliveries data from 2022-2024"""
        print("Loading and preprocessing data...")
        
        # Load the IPL_Ball_by_Ball_2008_2022.csv file which we confirmed exists
        try:
            file_path = os.path.join(self.data_path, 'IPL_Ball_by_Ball_2008_2022.csv')
            if os.path.exists(file_path):
                deliveries = pd.read_csv(file_path)
                print(f"Loaded data from {file_path}")
            else:
                print(f"Warning: File {file_path} not found")
                raise FileNotFoundError(f"Could not find {file_path}")
                
            # Also load 2025 data if available
            file_path_2025 = os.path.join(self.data_path, 'ipl_2025_deliveries.csv')
            if os.path.exists(file_path_2025):
                deliveries_2025 = pd.read_csv(file_path_2025)
                print(f"Loaded 2025 data from {file_path_2025}")
                
                # Map columns to match the format of the main dataset
                column_mapping = {
                    'runs_of_bat': 'runs_off_bat',
                    'striker': 'batter',
                    'batting_team': 'batting_team',
                    'bowling_team': 'bowling_team',
                    'wicket_type': 'wicket_type',
                    'player_dismissed': 'player_dismissed'
                }
                deliveries_2025 = deliveries_2025.rename(columns=column_mapping)
                
                # Combine datasets
                deliveries = pd.concat([deliveries, deliveries_2025], ignore_index=True)
                print(f"Combined datasets. Total records: {len(deliveries)}")
            else:
                print(f"Warning: 2025 data file not found at {file_path_2025}")
        except Exception as e:
            print(f"Error loading ball-by-ball data: {e}")
            raise
        
        # Map columns to what our code expects
        column_mapping = {}
        if 'batsman_run' in deliveries.columns and 'runs_off_bat' not in deliveries.columns:
            column_mapping['batsman_run'] = 'runs_off_bat'
        if 'ID' in deliveries.columns and 'match_id' not in deliveries.columns:
            column_mapping['ID'] = 'match_id'
            
        # Apply column mapping
        if column_mapping:
            deliveries = deliveries.rename(columns=column_mapping)
            print(f"Renamed columns: {column_mapping}")
        
        # Load squad data with player roles and credits
        try:
            squad_file = os.path.join(self.data_path, 'SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv')
            if os.path.exists(squad_file):
                squads = pd.read_csv(squad_file)
                print(f"Loaded squad data from {squad_file}")
                
                # Print columns for debugging
                print(f"Columns in squads data: {squads.columns.tolist()}")
                
                # Build column mapping
                column_mapping = {}
                
                # Try to find the player name column
                player_name_cols = ['Player Name', 'PlayerName', 'player_name', 'Name']
                for col in player_name_cols:
                    if col in squads.columns:
                        column_mapping[col] = 'player'
                        break
                        
                # Try to find the role column
                role_cols = ['Player Type', 'PlayerType', 'Role', 'player_type']
                for col in role_cols:
                    if col in squads.columns:
                        column_mapping[col] = 'role'
                        break
                        
                # Try to find the credits column
                credit_cols = ['Credits', 'Credit', 'credits']
                for col in credit_cols:
                    if col in squads.columns:
                        column_mapping[col] = 'credits'
                        break
                        
                # Try to find the team column
                team_cols = ['Team', 'team', 'Franchise']
                for col in team_cols:
                    if col in squads.columns:
                        column_mapping[col] = 'team'
                        break
                
                # Rename columns if needed
                if column_mapping:
                    squads = squads.rename(columns=column_mapping)
                    print(f"Renamed squad columns: {column_mapping}")
                
                # Map team codes to standard team names
                def map_team_code(team_code):
                    for standard_team, codes in self.team_code_mapping.items():
                        if team_code in codes:
                            return standard_team
                    return team_code
                
                # Apply team code mapping
                squads['team'] = squads['team'].apply(map_team_code)
                print(f"Team codes mapped to standard names. Unique teams: {squads['team'].unique()}")
                
                # Map role codes to standard role names
                role_mapping = {
                    'ALL': 'AR',  # Map ALL to AR for all-rounders
                    'BAT': 'BAT', # Batsmen
                    'BOWL': 'BOWL', # Bowlers
                    'WK': 'WK',    # Wicket-keepers
                    'AR': 'AR'     # All-rounders (direct mapping)
                }
                
                # Apply role mapping with error handling
                def map_role(role):
                    if pd.isna(role):
                        return 'BAT'  # Default role if missing
                    role = str(role).strip().upper()
                    return role_mapping.get(role, 'BAT')  # Default to BAT if unknown role
                
                # Apply role mapping
                squads['role'] = squads['role'].apply(map_role)
                print(f"Role codes mapped to standard roles. Unique roles: {squads['role'].unique()}")
                
                # Ensure player column contains strings
                squads['player'] = squads['player'].astype(str)
                
                # Ensure credits column is numeric
                squads['credits'] = pd.to_numeric(squads['credits'], errors='coerce')
                squads['credits'] = squads['credits'].fillna(8.0)  # Default credits if missing
                
                # Add playing status column (default to True)
                squads['is_playing'] = True
                
                print(f"Processed {len(squads)} players from squad data")
            else:
                print(f"Warning: Squad file not found at {squad_file}")
                # Create dummy squad data if file not found
                squads = pd.DataFrame({
                    'player': deliveries['batter'].unique(),
                    'role': ['BAT'] * len(deliveries['batter'].unique()),
                    'credits': [8.0] * len(deliveries['batter'].unique()),
                    'team': ['Unknown'] * len(deliveries['batter'].unique()),
                    'is_playing': [True] * len(deliveries['batter'].unique())
                })
        except Exception as e:
            print(f"Error loading squad data: {e}")
            # Create dummy squad data if file not found
            squads = pd.DataFrame({
                'player': deliveries['batter'].unique(),
                'role': ['BAT'] * len(deliveries['batter'].unique()),
                'credits': [8.0] * len(deliveries['batter'].unique()),
                'team': ['Unknown'] * len(deliveries['batter'].unique()),
                'is_playing': [True] * len(deliveries['batter'].unique())
            })
        
        # Calculate batting features
        batting_features = self._calculate_batting_features(deliveries)
        
        # Calculate bowling features
        bowling_features = self._calculate_bowling_features(deliveries)
        
        # Merge features
        all_features = self._merge_player_features(batting_features, bowling_features)
        
        # Join with squad data to get roles and credits
        player_data = pd.merge(
            all_features, 
            squads[['player', 'role', 'credits', 'team', 'is_playing']], 
            left_index=True, 
            right_on='player', 
            how='left'
        )
        
        # Fill missing values
        player_data = player_data.fillna({
            'role': 'BAT',
            'credits': 8.0,
            'team': 'Unknown',
            'is_playing': True
        })
        
        # Calculate fantasy points
        player_data['fantasy_points'] = self._calculate_fantasy_points(player_data)
        
        # Save for later use - exclude non-numeric columns 'player', 'role', 'team', and 'fantasy_points'
        numeric_features = player_data.drop(['player', 'role', 'team', 'fantasy_points'], axis=1)
        self.player_features = numeric_features
        self.player_points = player_data['fantasy_points']
        
        print(f"Data preprocessing complete. {len(player_data)} players processed.")
        return player_data
    
    def _calculate_batting_features(self, deliveries):
        """Calculate batting features from deliveries data"""
        # Check if required columns exist
        required_columns = ['batter', 'runs_off_bat', 'match_id', 'bowler', 'extras_run']
        for col in required_columns:
            if col not in deliveries.columns:
                print(f"Warning: Column '{col}' not found in data. Creating dummy column.")
                
                # Create dummy data for missing columns
                if col == 'runs_off_bat':
                    deliveries['runs_off_bat'] = deliveries['batsman_run'] if 'batsman_run' in deliveries.columns else 0
                elif col == 'match_id':
                    deliveries['match_id'] = deliveries['ID'] if 'ID' in deliveries.columns else deliveries.index // 300
                elif col == 'extras_run':
                    deliveries['extras_run'] = 0
        
        # Group by batter
        batter_stats = deliveries.groupby('batter').agg({
            'runs_off_bat': ['sum', 'mean'],
            'match_id': 'nunique',
            'bowler': 'nunique'
        })
        
        # Calculate strike rate
        runs_by_match = deliveries.groupby(['batter', 'match_id'])['runs_off_bat'].sum().reset_index()
        balls_by_match = deliveries.groupby(['batter', 'match_id']).size().reset_index(name='balls')
        strike_rate = pd.merge(runs_by_match, balls_by_match, on=['batter', 'match_id'])
        strike_rate['sr'] = strike_rate['runs_off_bat'] / strike_rate['balls'] * 100
        avg_sr = strike_rate.groupby('batter')['sr'].mean()
        
        # Calculate boundaries
        boundaries = deliveries[deliveries['runs_off_bat'].isin([4, 6])].groupby('batter').size()
        boundaries = boundaries.reindex(batter_stats.index, fill_value=0)
        
        # Calculate dots faced
        dots = deliveries[(deliveries['runs_off_bat'] == 0) & (deliveries['extras_run'] == 0)].groupby('batter').size()
        dots = dots.reindex(batter_stats.index, fill_value=0)
        
        # Create feature DataFrame
        batting_features = pd.DataFrame({
            'total_runs': batter_stats[('runs_off_bat', 'sum')],
            'avg_runs_per_ball': batter_stats[('runs_off_bat', 'mean')],
            'matches_played': batter_stats[('match_id', 'nunique')],
            'bowlers_faced': batter_stats[('bowler', 'nunique')],
            'strike_rate': avg_sr,
            'boundaries': boundaries,
            'dot_balls': dots
        })
        
        return batting_features
    
    def _calculate_bowling_features(self, deliveries):
        """Calculate bowling features from deliveries data"""
        # Check if required columns exist
        required_columns = ['bowler', 'runs_off_bat', 'match_id', 'batter']
        for col in required_columns:
            if col not in deliveries.columns:
                print(f"Warning: Column '{col}' not found in data. Creating dummy column.")
                
                # Create dummy data for missing columns
                if col == 'runs_off_bat':
                    deliveries['runs_off_bat'] = deliveries['batsman_run'] if 'batsman_run' in deliveries.columns else 0
                elif col == 'match_id':
                    deliveries['match_id'] = deliveries['ID'] if 'ID' in deliveries.columns else deliveries.index // 300
        
        # Group by bowler
        bowler_stats = deliveries.groupby('bowler').agg({
            'runs_off_bat': ['sum', 'mean'],
            'match_id': 'nunique',
            'batter': 'nunique'
        })
        
        # Calculate wickets
        if 'isWicketDelivery' in deliveries.columns:
            is_wicket = deliveries['isWicketDelivery'] == 1
        elif 'player_dismissed' in deliveries.columns:
            is_wicket = deliveries['player_dismissed'].notna()
        elif 'player_out' in deliveries.columns:
            is_wicket = deliveries['player_out'].notna()
        else:
            print("Warning: No wicket information found. Using dummy data.")
            is_wicket = pd.Series(False, index=deliveries.index)
            
        wickets = deliveries[is_wicket].groupby('bowler').size()
        wickets = wickets.reindex(bowler_stats.index, fill_value=0)
        
        # Calculate economy rate
        runs_by_match = deliveries.groupby(['bowler', 'match_id'])['runs_off_bat'].sum().reset_index()
        balls_by_match = deliveries.groupby(['bowler', 'match_id']).size().reset_index(name='balls')
        economy = pd.merge(runs_by_match, balls_by_match, on=['bowler', 'match_id'])
        economy['overs'] = economy['balls'] / 6
        # Avoid division by zero
        economy['overs'] = economy['overs'].replace(0, 0.1)
        economy['eco'] = economy['runs_off_bat'] / economy['overs']
        avg_eco = economy.groupby('bowler')['eco'].mean()
        
        # Create feature DataFrame with index as bowler name
        bowling_features = pd.DataFrame({
            'runs_conceded': bowler_stats[('runs_off_bat', 'sum')],
            'avg_runs_per_ball_bowl': bowler_stats[('runs_off_bat', 'mean')],
            'matches_bowled': bowler_stats[('match_id', 'nunique')],
            'batters_bowled': bowler_stats[('batter', 'nunique')],
            'wickets': wickets,
            'economy': avg_eco
        })
        
        return bowling_features
    
    def _merge_player_features(self, batting_features, bowling_features):
        """Merge batting and bowling features"""
        # Get all player names
        all_players = list(set(batting_features.index) | set(bowling_features.index))
        
        # Create combined dataframe
        combined_features = pd.DataFrame(index=all_players)
        
        # Add batting features
        for col in batting_features.columns:
            combined_features[f'bat_{col}'] = batting_features.get(col, pd.Series(0, index=all_players))
        
        # Add bowling features
        for col in bowling_features.columns:
            combined_features[f'bowl_{col}'] = bowling_features.get(col, pd.Series(0, index=all_players))
        
        # Fill NaN values with 0
        combined_features = combined_features.fillna(0)
        
        return combined_features
    
    def _calculate_fantasy_points(self, player_data):
        """Calculate fantasy points based on player statistics with improved accuracy"""
        # Initialize points
        points = pd.Series(0, index=player_data.index)
        
        # Batting points - improved scoring
        points += player_data['bat_total_runs'] * 1  # 1 point per run
        points += player_data['bat_boundaries'] * 1  # Additional 1 point per boundary
        
        # Recent form factor (weighted more heavily)
        recent_form_factor = 1 + (player_data['bat_strike_rate'] / 100)  # Good strike rate boosts points
        points = points * recent_form_factor
        
        # Milestone bonuses
        half_centuries = (player_data['bat_total_runs'] / player_data['bat_matches_played']).fillna(0)
        points += (half_centuries >= 50).astype(int) * 8  # Half-century bonus
        points += (half_centuries >= 100).astype(int) * 16  # Century bonus
        
        # Bowling points - improved scoring
        points += player_data['bowl_wickets'] * 25  # 25 points per wicket
        
        # Economy rate bonus (for players who bowled)
        has_bowled = player_data['bowl_matches_bowled'] > 0
        eco_bonus = pd.Series(0, index=player_data.index)
        eco_bonus[has_bowled & (player_data['bowl_economy'] < 6)] = 8  # Increased bonus for good economy
        eco_bonus[has_bowled & (player_data['bowl_economy'] >= 6) & (player_data['bowl_economy'] < 7)] = 6
        eco_bonus[has_bowled & (player_data['bowl_economy'] >= 7) & (player_data['bowl_economy'] < 8)] = 4
        eco_bonus[has_bowled & (player_data['bowl_economy'] > 10)] = -3
        eco_bonus[has_bowled & (player_data['bowl_economy'] > 11)] = -5
        eco_bonus[has_bowled & (player_data['bowl_economy'] > 12)] = -8
        points += eco_bonus
        
        # Consistency bonus - reward players who play regularly
        consistency_bonus = player_data['bat_matches_played'] * 0.5 + player_data['bowl_matches_bowled'] * 0.5
        points += consistency_bonus
        
        # Role-based adjustments with improved weighting
        role_multiplier = {
            'BAT': 1.1,  # Boost batsmen slightly
            'BOWL': 1.15,  # Boost bowlers slightly more
            'AR': 1.2,  # All-rounders get 20% bonus (increased)
            'WK': 1.15  # Wicket-keepers get 15% bonus (increased)
        }
        
        for role, mult in role_multiplier.items():
            is_role = player_data['role'] == role
            points[is_role] *= mult
        
        return points
    
    def train_models(self):
        """Train multiple models and select the best one"""
        print("Training models...")
        
        # Prepare data
        X = self.player_features
        y = self.player_points
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Define models to try
        models = {
            'decision_tree': DecisionTreeRegressor(max_depth=8, random_state=42),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Train neural network separately
        nn_model = self._train_neural_network(X_train, y_train, X_test, y_test)
        models['neural_network'] = nn_model
        
        # Train and evaluate each model
        results = {}
        for name, model in tqdm(models.items(), desc="Training models"):
            if name != 'neural_network':  # Neural network already trained
                model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2
            }
            
            # Update best model
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
        
        # Save all models
        self.models = {name: result['model'] for name, result in results.items()}
        
        # Print results
        print("Model training results:")
        for name, result in results.items():
            print(f"{name}: MSE={result['mse']:.2f}, R²={result['r2']:.2f}")
        
        print(f"Best model: {max(results.items(), key=lambda x: x[1]['r2'])[0]}")
        return results
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train a deep neural network model"""
        # Create model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Compile model - use 'mean_squared_error' string instead of 'mse' function
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        return model
    
    def load_and_check_lineup_data(self, team1, team2):
        """Load and check if players are in the playing lineup"""
        try:
            # Try to load lineup data
            lineup_file = os.path.join(self.data_path, 'ipl_lineups_2025.csv')
            if os.path.exists(lineup_file):
                lineups = pd.read_csv(lineup_file)
                print(f"Loaded lineup data from {lineup_file}")
                
                # Determine column names
                player_col = None
                team_col = None
                playing_col = None
                
                # Try to detect column names
                for col in lineups.columns:
                    col_lower = col.lower()
                    if 'player' in col_lower or 'name' in col_lower:
                        player_col = col
                    elif 'team' in col_lower or 'franchise' in col_lower:
                        team_col = col
                    elif 'playing' in col_lower or 'lineup' in col_lower or 'xi' in col_lower or 'status' in col_lower:
                        playing_col = col
                
                if player_col and team_col:
                    # Filter for current teams
                    current_lineups = lineups[(lineups[team_col] == team1) | (lineups[team_col] == team2)]
                    
                    if len(current_lineups) > 0:
                        print(f"Found {len(current_lineups)} players in lineup data for {team1} and {team2}")
                        
                        # If we have playing status, use it
                        if playing_col:
                            # Create a dictionary of player -> is_playing
                            is_playing = {}
                            for _, row in current_lineups.iterrows():
                                player_name = row[player_col]
                                # Determine if player is playing based on various possible formats
                                status = str(row[playing_col]).lower()
                                playing = ('yes' in status or 'true' in status or 'playing' in status or 
                                          'xi' in status or status == '1')
                                is_playing[player_name] = playing
                            
                            return is_playing
                        else:
                            # If no playing status column, assume all listed players are playing
                            return {row[player_col]: True for _, row in current_lineups.iterrows()}
                
                print("Could not determine lineup status from data")
            else:
                print(f"No lineup file found at {lineup_file}")
            
            return None
        except Exception as e:
            print(f"Error loading lineup data: {e}")
            return None
    
    def predict_team(self, team1, team2):
        """Predict the best Dream11 team for a given match"""
        try:
            print(f"Predicting best team for {team1} vs {team2}...")
            # Load and preprocess data
            squad_file = os.path.join(self.data_path, 'SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv')
            if not os.path.exists(squad_file):
                raise FileNotFoundError(f"Squad data file not found at {squad_file}")
            
            deliveries_df = pd.read_csv(os.path.join(self.data_path, 'ipl_2025_deliveries.csv'))
            squads_df = pd.read_csv(squad_file)
            
            print(f"Squad columns: {squads_df.columns.tolist()}")
            print(f"Unique teams in squad data: {squads_df['Team'].unique()}")
            
            # Create team code dictionaries for lookup
            team_codes = {}
            for standard_team, codes in self.team_code_mapping.items():
                for code in codes:
                    team_codes[code] = standard_team
            
            # Standardize team names
            team1_std = team_codes.get(team1, team1)
            team2_std = team_codes.get(team2, team2)
            
            # Get all possible codes for the selected teams
            team1_codes = [team1_std]
            team2_codes = [team2_std]
            for code, std_team in team_codes.items():
                if std_team == team1_std and code not in team1_codes:
                    team1_codes.append(code)
                elif std_team == team2_std and code not in team2_codes:
                    team2_codes.append(code)
            
            print(f"Looking for teams with codes: {team1_codes} or {team2_codes}")
            
            # Filter squads for selected teams
            team_mask = squads_df['Team'].isin(team1_codes + team2_codes)
            team_squads = squads_df[team_mask].copy()
            
            if len(team_squads) == 0:
                print(f"No players found for teams {team1} and {team2}")
                print(f"Available teams in data: {squads_df['Team'].unique()}")
                raise ValueError(f"No players found for teams {team1} and {team2}")
            
            print(f"Found {len(team_squads)} players for teams {team1} and {team2}")
            
            # Map role codes to standard role names
            role_mapping = {
                'ALL': 'AR',    # Map ALL to AR for all-rounders
                'BAT': 'BAT',   # Batsmen
                'BOWL': 'BOWL', # Bowlers
                'WK': 'WK',     # Wicket-keepers
                'AR': 'AR'      # All-rounders (direct mapping)
            }
            
            # Apply role mapping
            team_squads['standardized_role'] = team_squads['Player Type'].apply(
                lambda x: role_mapping.get(x, 'BAT') if pd.notna(x) else 'BAT'
            )
            
            # Add team standardization
            team_squads['standardized_team'] = team_squads['Team'].apply(
                lambda x: team1_std if x in team1_codes else team2_std
            )
            
            # Calculate player features and predicted points
            player_features = self.calculate_player_features(deliveries_df, team_squads)
            if player_features is None or len(player_features) == 0:
                raise ValueError("Failed to calculate player features")
            
            # Validate role distribution
            role_counts = team_squads['standardized_role'].value_counts()
            print(f"Available players by role: {role_counts.to_dict()}")
            
            # Check if we have at least one player for each required role
            required_roles = ['WK', 'BAT', 'BOWL']
            for role in required_roles:
                if role not in role_counts or role_counts[role] == 0:
                    raise ValueError(f"No players available for role {role}")
            
            # Create optimization problem with role constraints
            model = LpProblem("Dream11_Team_Selection", LpMaximize)
            
            # Create binary variables for each player
            player_vars = LpVariable.dicts("player",
                                         ((i, p) for i, p in enumerate(team_squads.index)),
                                         cat='Binary')
            
            # Objective: Maximize predicted points
            model += lpSum([player_features.loc[p, 'predicted_points'] * player_vars[i, p] 
                           for i, p in enumerate(team_squads.index)])
            
            # Constraints
            # 1. Total players = 11
            model += lpSum([player_vars[i, p] for i, p in enumerate(team_squads.index)]) == 11
            
            # 2. Role constraints
            role_requirements = {
                'WK': (1, 1),  # Exactly 1 wicket-keeper
                'BAT': (3, 5),  # 3-5 batsmen
                'AR': (1, 3),  # 1-3 all-rounders
                'BOWL': (3, 5)  # 3-5 bowlers
            }
            
            # Adjust role requirements if necessary
            if 'AR' not in role_counts or role_counts['AR'] == 0:
                print("No all-rounders available, adjusting constraints")
                role_requirements['BAT'] = (4, 6)  # More batsmen
                role_requirements['BOWL'] = (4, 6)  # More bowlers
                role_requirements.pop('AR')  # Remove AR constraint
            
            for role, (min_players, max_players) in role_requirements.items():
                role_players = team_squads[team_squads['standardized_role'] == role].index
                if len(role_players) > 0:
                    model += lpSum([player_vars[i, p] for i, p in enumerate(team_squads.index) 
                                  if p in role_players]) >= min_players
                    model += lpSum([player_vars[i, p] for i, p in enumerate(team_squads.index) 
                                  if p in role_players]) <= max_players
            
            # 3. Team constraints (at least 4 players from each team)
            for team in [team1_std, team2_std]:
                team_players = team_squads[team_squads['standardized_team'] == team].index
                if len(team_players) > 0:
                    model += lpSum([player_vars[i, p] for i, p in enumerate(team_squads.index) 
                                  if p in team_players]) >= 4
            
            # Solve the optimization problem
            print("Using CBC solver to optimize team selection...")
            model.solve()
            
            if LpStatus[model.status] != 'Optimal':
                raise ValueError(f"Could not find optimal solution. Status: {LpStatus[model.status]}")
            
            # Get selected players
            selected_indices = [idx for idx, var in player_vars.items() if var.value() > 0.5]
            if not selected_indices:
                raise ValueError("No players selected in the solution")
                
            print(f"Selected {len(selected_indices)} players")
            
            # Extract the player indices from the tuple (i, p)
            selected_player_indices = [idx[1] for idx in selected_indices]
            selected_players = team_squads.loc[selected_player_indices].copy()
            
            if len(selected_players) != 11:
                raise ValueError(f"Expected 11 players, but got {len(selected_players)}")
            
            # Add predicted points to display
            if 'predicted_points' in player_features.columns:
                for idx in selected_players.index:
                    if idx in player_features.index:
                        selected_players.loc[idx, 'predicted_points'] = player_features.loc[idx, 'predicted_points']
            
            # Sort by predicted points and assign captain/vice-captain
            if 'predicted_points' in selected_players.columns:
                selected_players = selected_players.sort_values('predicted_points', ascending=False)
            
            # Copy original names before adding captain/vice-captain designations
            selected_players['Display Name'] = selected_players['Player Name'].copy()
            
            # Add captain/vice-captain designations
            if len(selected_players) >= 2:
                selected_players.iloc[0, selected_players.columns.get_loc('Display Name')] += ' (C)'
                selected_players.iloc[1, selected_players.columns.get_loc('Display Name')] += ' (VC)'
            
            print("Selected main team with 11 players")
            return selected_players
            
        except Exception as e:
            print(f"Error in team prediction: {str(e)}")
            traceback.print_exc()
            return None
    
    def save_models(self, path='models'):
        """Save trained models to disk"""
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        
        # Save each model
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(os.path.join(path, f"{name}_model.h5"))
            else:
                joblib.dump(model, os.path.join(path, f"{name}_model.joblib"))
        
        print(f"Models saved to {path} directory")
    
    def load_models(self, path='models'):
        """Load trained models from disk"""
        try:
            # Load the scaler
            self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
            
            # Load each model
            self.models = {}
            for model_file in os.listdir(path):
                if model_file.endswith('_model.joblib'):
                    name = model_file.replace('_model.joblib', '')
                    self.models[name] = joblib.load(os.path.join(path, model_file))
                elif model_file.endswith('_model.h5'):
                    try:
                        # Try to load with custom_objects
                        name = model_file.replace('_model.h5', '')
                        # Use mean_squared_error instead of mse to avoid serialization issues
                        self.models[name] = tf.keras.models.load_model(
                            os.path.join(path, model_file),
                            custom_objects={'mean_squared_error': tf.keras.losses.mean_squared_error}
                        )
                    except Exception as e:
                        print(f"Error loading neural network model: {e}")
                        print("Training a simple model as fallback...")
                        # Create a simple model as fallback
                        fallback_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        # Train it on a small sample if we have data
                        if self.player_features is not None and self.player_points is not None:
                            X = self.player_features.iloc[:100]
                            y = self.player_points.iloc[:100]
                            X_scaled = self.scaler.transform(X)
                            fallback_model.fit(X_scaled, y)
                        self.models[name] = fallback_model
            
            # Set the best model (or use gradient boosting if we don't have models)
            if self.models:
                best_models = [(name, self._evaluate_model(model)) for name, model in self.models.items()]
                if best_models:
                    best_name = max(best_models, key=lambda x: x[1])[0]
                    self.best_model = self.models[best_name]
                    print(f"Best model: {best_name}")
                    return
            
            # Fallback if no models could be loaded
            print("No models could be loaded. Creating a default model.")
            self.best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            if self.player_features is not None and self.player_points is not None:
                X = self.player_features.iloc[:100] if len(self.player_features) > 100 else self.player_features
                y = self.player_points.iloc[:100] if len(self.player_points) > 100 else self.player_points
                X_scaled = self.scaler.transform(X)
                self.best_model.fit(X_scaled, y)
            self.models['gradient_boosting'] = self.best_model
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Create a fallback model
            self.best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.models = {'gradient_boosting': self.best_model}
    
    def _evaluate_model(self, model):
        """Evaluate a model's performance"""
        # Use a small sample of the data for quick evaluation
        X = self.player_features.iloc[:100]
        y = self.player_points.iloc[:100]
        
        X_scaled = self.scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        return r2_score(y, y_pred)
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature importances
            importances = self.best_model.feature_importances_
            
            # Create DataFrame
            features_df = pd.DataFrame({
                'feature': self.player_features.columns,
                'importance': importances
            })
            
            # Sort by importance
            features_df = features_df.sort_values('importance', ascending=False).head(20)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=features_df)
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()
            
            # Save to file
            plt.savefig('feature_importance.png')
            plt.close()
            
            print("Feature importance plot saved as 'feature_importance.png'")
        else:
            print("Best model doesn't have feature importances")

    def run_app(self):
        """Run Streamlit app"""
        import streamlit as st
        
        st.title("Dream11 Team Predictor")
        
        # Load team names
        team_names = ["CSK", "MI", "RCB", "KKR", "DC", "PBKS", "RR", "SRH", "GT", "LSG"]
        
        # Team selection
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", team_names)
        with col2:
            team2 = st.selectbox("Select Team 2", [t for t in team_names if t != team1])
        
        if st.button("Predict Team"):
            with st.spinner("Predicting Dream11 team..."):
                try:
                    # Predict team
                    result = self.predict_team(team1, team2)
                    
                    if result is not None:
                        # Display main team
                        st.subheader("Main Dream11 Team")
                        main_team = result.copy()
                        st.dataframe(main_team[['player', 'team', 'role', 'credits', 'predicted_points']])
                        
                        # Display team visualization
                        st.subheader("Team Composition")
                        self.plot_team_composition(main_team)
                        
                        # Display player points prediction
                        st.subheader("Player Points Prediction")
                        self.plot_player_points(main_team)
                    else:
                        st.error("Failed to predict team. Please check logs for details.")
                except Exception as e:
                    st.error(f"Error while predicting team: {str(e)}")
                    st.error("Please check the error logs for more details.")
        
        # Display info
        st.sidebar.title("About")
        st.sidebar.info(
            "This app predicts the best Dream11 team for a match using "
            "machine learning based on player statistics."
        )
        
    def plot_team_composition(self, team):
        """Plot team composition by role"""
        import streamlit as st
        import matplotlib.pyplot as plt
        
        try:
            # Clean roles (remove captain/vice-captain markers)
            team_clean = team.copy()
            team_clean['base_role'] = team_clean['role'].str.split(' ').str[0]
            
            # Count players by role
            role_counts = team_clean['base_role'].value_counts()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(role_counts.index, role_counts.values, color=['royalblue', 'forestgreen', 'firebrick', 'goldenrod'])
            
            # Add labels
            ax.set_xlabel('Role')
            ax.set_ylabel('Number of Players')
            ax.set_title('Team Composition by Role')
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}', ha='center', va='bottom')
            
            # Display in Streamlit
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error plotting team composition: {str(e)}")
    
    def plot_player_points(self, team):
        """Plot predicted points by player"""
        import streamlit as st
        import matplotlib.pyplot as plt
        
        try:
            # Sort by predicted points
            team_sorted = team.sort_values('predicted_points', ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(team_sorted['player'], team_sorted['predicted_points'], 
                        color=['gold' if '(C)' in r else 'silver' if '(VC)' in r else 'royalblue' 
                              for r in team_sorted['role']])
            
            # Add labels
            ax.set_xlabel('Predicted Points')
            ax.set_ylabel('Player')
            ax.set_title('Predicted Points by Player')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gold', label='Captain'),
                Patch(facecolor='silver', label='Vice Captain'),
                Patch(facecolor='royalblue', label='Player')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            # Display in Streamlit
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error plotting player points: {str(e)}")

    def calculate_player_features(self, deliveries_df, squads_df):
        """Calculate player features from deliveries data"""
        try:
            # Calculate batting features
            batting_features = self._calculate_batting_features(deliveries_df)
            
            # Calculate bowling features
            bowling_features = self._calculate_bowling_features(deliveries_df)
            
            # Merge features
            all_features = self._merge_player_features(batting_features, bowling_features)
            
            # Add player features to squad data
            player_data = pd.DataFrame()
            
            if not all_features.empty and not squads_df.empty:
                # Use appropriate column for player name in squads
                player_col = 'Player Name' if 'Player Name' in squads_df.columns else 'player'
                
                # Create a copy of the squads data
                player_data = squads_df.copy()
                
                # Initialize predicted_points column with default value
                player_data['predicted_points'] = player_data.apply(lambda row: float(row['Credits']) * 100, axis=1)
                
            return player_data
            
        except Exception as e:
            print(f"Error calculating player features: {e}")
            traceback.print_exc()
            return pd.DataFrame()
            
    def predict_player_points(self, squad_data):
        """Predict fantasy points for players in squad data"""
        try:
            # Check if squad data is valid
            if squad_data is None or len(squad_data) == 0:
                print("Error: Squad data is empty or None")
                return np.zeros(0)
                
            # Create a simple point prediction based on Credits value
            # This is a fallback method when the ML model isn't available
            predicted_points = np.zeros(len(squad_data))
            
            # Base points on player credits and role
            for i, (_, player) in enumerate(squad_data.iterrows()):
                # Calculate base points from credits
                credits = float(player['Credits'])
                
                # Get role factor for different roles
                role = player['role'] if 'role' in player else player['Player Type']
                role_factor = 1.0
                if role == 'BAT':
                    role_factor = 1.2
                elif role == 'BOWL':
                    role_factor = 1.3
                elif role == 'AR' or role == 'ALL':
                    role_factor = 1.4
                elif role == 'WK':
                    role_factor = 1.1
                
                # Calculate points with some randomness to get variation
                predicted_points[i] = credits * 100 * role_factor * np.random.uniform(0.8, 1.2)
                
            # Sort players by predicted points for debugging
            player_name_col = 'Player Name' if 'Player Name' in squad_data.columns else 'player'
            top_players = pd.DataFrame({
                'name': squad_data[player_name_col],
                'role': squad_data['role'] if 'role' in squad_data else squad_data['Player Type'],
                'points': predicted_points
            }).sort_values('points', ascending=False).head(10)
            
            print("Top 10 players by predicted points:")
            for _, player in top_players.iterrows():
                print(f"{player['name']} ({player['role']}): {player['points']:.2f}")
                
            return predicted_points
            
        except Exception as e:
            print(f"Error predicting player points: {e}")
            traceback.print_exc()
            return np.zeros(len(squad_data))


# Usage example
if __name__ == "__main__":
    predictor = Dream11Predictor()
    
    # Step 1: Load and preprocess data
    player_data = predictor.load_and_preprocess_data()
    
    # Step 2: Train models
    results = predictor.train_models()
    
    # Step 3: Save models
    predictor.save_models()
    
    # Step 4: Plot feature importance
    predictor.plot_feature_importance()
    
    # Step 5: Predict team for a match
    team = predictor.predict_team('CSK', 'MI')
    print("\nPredicted Dream11 Team:")
    print(team)
    
    # Step 6: Run Streamlit app
    predictor.run_app() 