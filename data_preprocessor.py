import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple
import os

class DataPreprocessor:
    def __init__(self, data_dir="dataset"):
        self.label_encoders = {}
        self.data_dir = data_dir
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required datasets"""
        data = {}
        
        # Load main datasets
        try:
            data['deliveries'] = pd.read_csv(os.path.join(self.data_dir, 'ipl_2025_deliveries.csv'))
            data['fantasy'] = pd.read_csv(os.path.join(self.data_dir, 'Final_Fantasy_data.csv'))
            data['fielding'] = pd.read_csv(os.path.join(self.data_dir, 'Fielding_data.csv'))
            data['matches'] = pd.read_csv(os.path.join(self.data_dir, 'IPL_Matches_2008_2022.csv'))
            data['ball_by_ball'] = pd.read_csv(os.path.join(self.data_dir, 'IPL_Ball_by_Ball_2008_2022.csv'))
            data['squads'] = pd.read_csv(os.path.join(self.data_dir, 'SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv'))
            
            # Load performance datasets
            data['wickets'] = pd.read_csv(os.path.join(self.data_dir, 'Most Wickets All Seasons Combine.csv'))
            data['sixes'] = pd.read_csv(os.path.join(self.data_dir, 'Most Sixes Per Innings All Seasons Combine.csv'))
            data['runs_per_over'] = pd.read_csv(os.path.join(self.data_dir, 'Most Runs Per Over All Seasons Combine.csv'))
            data['runs_conceded'] = pd.read_csv(os.path.join(self.data_dir, 'Most Runs Conceded Per Innings All Seasons Combine.csv'))
            data['runs'] = pd.read_csv(os.path.join(self.data_dir, 'Most Runs All Seasons Combine.csv'))
            data['fours'] = pd.read_csv(os.path.join(self.data_dir, 'Most Fours Per Innings All Seasons Combine.csv'))
            data['dot_balls'] = pd.read_csv(os.path.join(self.data_dir, 'Most Dot Balls Per Innings All Seasons Combine.csv'))
            data['fastest_50'] = pd.read_csv(os.path.join(self.data_dir, 'Fastest Fifties All Seasons Combine.csv'))
            data['fastest_100'] = pd.read_csv(os.path.join(self.data_dir, 'Fastest Centuries All Seasons Combine.csv'))
            data['bowling_sr'] = pd.read_csv(os.path.join(self.data_dir, 'Best Bowling Strike Rate Per Innings All Seasons Combine.csv'))
            data['bowling_econ'] = pd.read_csv(os.path.join(self.data_dir, 'Best Bowling Economy Per Innings All Seasons Combine.csv'))
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            # Continue with available data
            
        return data
    
    def calculate_fantasy_points(self, row: pd.Series) -> float:
        """Calculate fantasy points based on Dream11 rules"""
        points = 0
        
        # Basic points
        points += 4  # Playing in starting XI
        
        # Batting points
        if 'runs' in row:
            points += row['runs']  # 1 point per run
            if row['runs'] >= 100:
                points += 16  # Century bonus
            elif row['runs'] >= 50:
                points += 8   # Half-century bonus
            if row['runs'] == 0:
                points -= 2   # Duck penalty
                
        # Bowling points
        if 'wickets' in row:
            points += row['wickets'] * 25  # 25 points per wicket
            if row['wickets'] >= 5:
                points += 16  # 5 wickets bonus
            elif row['wickets'] >= 4:
                points += 8   # 4 wickets bonus
                
        # Fielding points
        if 'catches' in row:
            points += row['catches'] * 8  # 8 points per catch
            
        # Economy rate points
        if 'economy' in row:
            if row['economy'] < 4:
                points += 6
            elif row['economy'] < 5:
                points += 4
            elif row['economy'] < 6:
                points += 2
            elif row['economy'] > 11:
                points -= 6
            elif row['economy'] > 10:
                points -= 4
            elif row['economy'] > 9:
                points -= 2
                
        # Strike rate points
        if 'strike_rate' in row:
            if row['strike_rate'] > 170:
                points += 6
            elif row['strike_rate'] > 150:
                points += 4
            elif row['strike_rate'] > 130:
                points += 2
            elif row['strike_rate'] < 50:
                points -= 6
            elif row['strike_rate'] < 60:
                points -= 4
            elif row['strike_rate'] < 70:
                points -= 2
                
        return points
    
    def create_player_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive player features"""
        player_stats = {}
        
        # Define column mappings for each metric
        metric_columns = {
            'runs': {'value_col': 'Runs', 'count_col': 'Mat'},
            'wickets': {'value_col': 'Wkts', 'count_col': 'Mat'},
            'sixes': {'value_col': '6s', 'count_col': 'Mat'},
            'fours': {'value_col': '4s', 'count_col': 'Mat'},
            'dot_balls': {'value_col': 'Dot Balls', 'count_col': 'Mat'}
        }
        
        # Process each performance metric
        for metric in ['runs', 'wickets', 'sixes', 'fours', 'dot_balls']:
            if metric in data:
                # Get the appropriate column names for this metric
                value_col = metric_columns[metric]['value_col']
                count_col = metric_columns[metric]['count_col']
                
                # Check if the columns exist in the dataframe
                if value_col in data[metric].columns and count_col in data[metric].columns:
                    df = data[metric].groupby('Player').agg({
                        value_col: ['mean', 'std', 'max'],
                        count_col: 'count'
                    }).reset_index()
                    
                    for col in ['mean', 'std', 'max']:
                        player_stats[f'{metric}_{col}'] = dict(zip(df['Player'], df[(value_col, col)]))
                    player_stats[f'{metric}_matches'] = dict(zip(df['Player'], df[(count_col, 'count')]))
        
        # Create final features DataFrame
        features_df = pd.DataFrame.from_dict(player_stats, orient='index').T
        features_df = features_df.fillna(0)
        
        return features_df
    
    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features using Label Encoding"""
        df_encoded = df.copy()
        
        for col in columns:
            # Only encode if the column exists in the dataframe
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
                
        return df_encoded
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare final training dataset"""
        # Load all data
        data = self.load_data()
        
        # Create player features
        features_df = self.create_player_features(data)
        
        # Get all players from squads
        squads = data['squads']
        squads = squads.rename(columns={
            'Player Name': 'Player',
            'Player Type': 'role',
            'Credits': 'credits',
            'Team': 'team'
        })
        
        # Create synthetic fantasy points based on available statistics
        fantasy_points = []
        player_names = []
        
        for _, player in squads.iterrows():
            if player['Player'] in features_df.index:
                # Calculate base points from player credits
                base_points = float(player['credits']) * 10
                
                # Add role-based bonus
                role_bonus = {
                    'BAT': 1.2,
                    'BOWL': 1.3,
                    'ALL': 1.4,
                    'WK': 1.1
                }
                multiplier = role_bonus.get(player['role'], 1.0)
                points = base_points * multiplier
                
                # Add some random variation
                points *= np.random.normal(1, 0.2)  # 20% random variation
                fantasy_points.append(max(0, points))  # Ensure non-negative points
                player_names.append(player['Player'])
        
        # Create target variable with aligned indices
        target = pd.Series(fantasy_points, index=player_names)
        
        # Ensure features and target have the same players
        common_players = list(set(features_df.index) & set(target.index))
        features_df = features_df.loc[common_players]
        target = target.loc[common_players]
        
        # Only encode categorical columns that exist in the features DataFrame
        categorical_columns = ['role', 'team']  # Add other categorical columns as needed
        existing_categorical_columns = [col for col in categorical_columns if col in features_df.columns]
        
        if existing_categorical_columns:
            features_df = self.encode_categorical_features(features_df, existing_categorical_columns)
        
        return features_df, target
        
    def get_match_players(self, team1: str, team2: str) -> pd.DataFrame:
        """Get players from both teams with their roles and credits"""
        # Load squad data
        data = self.load_data()
        squads = data['squads']
        
        # Rename columns to match our expected format
        squads = squads.rename(columns={
            'Player Name': 'Player',
            'Player Type': 'role',
            'Credits': 'credits',
            'Team': 'team'
        })
        
        # Filter players from selected teams
        match_players = squads[squads['team'].isin([team1, team2])]
        
        # Add player features
        features_df = self.create_player_features(data)
        
        # Ensure feature columns match those used in training
        feature_columns = features_df.columns.tolist()
        
        # Merge with features, keeping both metadata and feature columns
        match_players = match_players.merge(
            features_df[feature_columns], 
            left_on='Player', 
            right_index=True, 
            how='left'
        )
        
        # Fill missing values with reasonable defaults
        match_players = match_players.fillna({
            col: 0 for col in feature_columns
        })
        
        # Convert credits to float if it's not already
        match_players['credits'] = match_players['credits'].astype(float)
        
        # Set Player as index
        match_players = match_players.set_index('Player')
        
        return match_players 