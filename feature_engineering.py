import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_engineering')

class FeatureEngineer:
    """
    Enhanced feature engineering for Dream11 fantasy cricket predictions
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the feature engineer
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.venue_data = None
        self.historical_data = None
        self.team_strengths = None
        
        # Load reference data
        self._load_reference_data()
        
    def _load_reference_data(self):
        """Load venue and historical data"""
        try:
            # Load venue data if available
            venue_file = os.path.join(self.data_dir, "venue_stats.csv")
            if os.path.exists(venue_file):
                self.venue_data = pd.read_csv(venue_file)
                logger.info(f"Loaded venue data: {len(self.venue_data)} venues")
            else:
                logger.warning(f"Venue data file not found at {venue_file}")
                # Create minimal venue data with defaults
                self.venue_data = pd.DataFrame({
                    'venue': ['default', 'MA Chidambaram Stadium', 'Wankhede Stadium', 'Eden Gardens'], # Add common venues
                    'pitch_type': ['balanced', 'balanced', 'batting_friendly', 'bowling_friendly'],
                    'avg_first_innings_score': [160, 165, 180, 155],
                    'avg_second_innings_score': [150, 155, 170, 145],
                    'avg_spin_wickets_per_match': [4, 5, 3, 6],
                    'avg_pace_wickets_per_match': [6, 5, 7, 5]
                })
                
            # Load historical player performance data if available
            historical_file = os.path.join(self.data_dir, "player_history.csv")
            if os.path.exists(historical_file):
                self.historical_data = pd.read_csv(historical_file)
                # Convert match_date if it exists
                if 'match_date' in self.historical_data.columns:
                     try:
                         self.historical_data['match_date'] = pd.to_datetime(self.historical_data['match_date'], errors='coerce')
                         # Drop rows where date conversion failed
                         self.historical_data.dropna(subset=['match_date'], inplace=True)
                     except Exception as date_err:
                         logger.error(f"Error converting historical match_date: {date_err}")
                logger.info(f"Loaded historical data: {len(self.historical_data)} records")
            else:
                logger.warning(f"Historical data file not found at {historical_file}")
                
            # Load team strengths/weaknesses if available
            team_file = os.path.join(self.data_dir, "team_strengths.csv")
            if os.path.exists(team_file):
                self.team_strengths = pd.read_csv(team_file)
                logger.info(f"Loaded team strengths data: {len(self.team_strengths)} teams")
            else:
                logger.warning(f"Team strengths file not found at {team_file}")
                # Create minimal team data with defaults
                default_teams = ['CSK', 'MI', 'RCB', 'KKR', 'DC', 'PBKS', 'RR', 'SRH', 'GT', 'LSG', 'CHE'] # Added GT, LSG, CHE
                self.team_strengths = pd.DataFrame({
                    'team': default_teams,
                    'batting_strength': [0.8, 0.9, 1.0, 0.7, 0.8, 0.9, 0.7, 0.8, 0.85, 0.8, 0.8 ],
                    'bowling_strength': [0.9, 0.8, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.85, 0.85, 0.9],
                    'vs_left_arm_pace': [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.9, 0.8, 0.8, 0.8],
                    'vs_right_arm_pace': [0.9, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.85, 0.9],
                    'vs_left_arm_spin': [0.7, 0.8, 0.9, 0.8, 0.7, 0.9, 0.8, 0.9, 0.75, 0.8, 0.7],
                    'vs_right_arm_spin': [0.8, 0.9, 0.8, 0.9, 0.8, 0.7, 0.9, 0.8, 0.8, 0.8, 0.8]
                })
                # Standardize team column
                if 'team' in self.team_strengths.columns:
                    self.team_strengths['team'] = self.team_strengths['team'].str.upper().str.strip()

        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def enhance_player_features(self, player_data: pd.DataFrame, 
                              home_team: Optional[str] = None, away_team: Optional[str] = None, 
                              venue: Optional[str] = None) -> pd.DataFrame:
        """
        Add enhanced features to player data
        
        Args:
            player_data (pd.DataFrame): Base player data
            home_team (str, optional): Home team code
            away_team (str, optional): Away team code
            venue (str, optional): Match venue
            
        Returns:
            pd.DataFrame: Enhanced player data with additional features
        """
        try:
            # Create a copy to avoid modifying the original
            df = player_data.copy()

            # Standardize common columns first
            df = self._standardize_columns(df)

            # 1. Add role-based features
            df = self.add_role_features(df)
            
            # 2. Add recent form features if historical data is available
            if self.historical_data is not None:
                df = self.add_recent_form_features(df)
            else:
                 # Add dummy form features if no historical data
                 if 'credits' in df.columns:
                     df['recent_form'] = df['credits'] * 0.1
                     df['last_3_avg'] = df['credits'] * 10
                     df['form_trend'] = 0  # Neutral trend
                 else:
                     df['recent_form'] = 1.0 # Default if credits also missing
                     df['last_3_avg'] = 10.0
                     df['form_trend'] = 0
                
            # 3. Add opposition team strength/weakness features
            if home_team and away_team:
                 df = self.add_opposition_features(df, home_team, away_team)
            else:
                 # Add dummy opposition features if teams missing
                 df['vs_team_strength'] = 1.0
                 df['vs_role_advantage'] = 0.0
            
            # 4. Add venue-specific features
            if venue is not None:
                df = self.add_venue_features(df, venue)
            else:
                 # Add dummy venue features if venue missing
                 df['venue_advantage'] = 1.0
                 df['pitch_factor'] = 1.0
                 df['bowling_factor'] = 1.0
                 df['pitch_is_batting_friendly'] = 0
                 df['pitch_is_bowling_friendly'] = 0
                 df['pitch_is_balanced'] = 1 # Default to balanced
                
            # 5. Add interaction features
            df = self.add_interaction_features(df)
            
            logger.info(f"Added enhanced features to {len(df)} player records")
            return df
            
        except Exception as e:
            logger.error(f"Error enhancing player features: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original data but ensure 'predicted_points' exists with NaN
            player_data['predicted_points'] = np.nan
            return player_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize common column names"""
        column_mapping = {
             'Player Name': 'player_name',
             'Player Type': 'role',
             'Team': 'team',
             'Credits': 'credits'
        }
        df = df.rename(columns=column_mapping, errors='ignore')

        # Standardize role values (basic example, can be expanded)
        if 'role' in df.columns:
             df['role'] = df['role'].replace({
                 'Batsman': 'BAT',
                 'Bowler': 'BOWL',
                 'All Rounder': 'AR',
                 'Wicket Keeper': 'WK'
             })
             # Ensure roles are uppercase
             df['role'] = df['role'].str.upper()
             # Fill missing roles with a default, e.g., 'BAT'
             df['role'] = df['role'].fillna('BAT')
             # Handle unexpected values not in [WK, BAT, AR, BOWL]
             valid_roles = {'WK', 'BAT', 'AR', 'BOWL'}
             df['role'] = df['role'].apply(lambda x: x if x in valid_roles else 'BAT')

        # Standardize team codes
        if 'team' in df.columns:
             df['team'] = df['team'].str.upper().str.strip()
             # Example: Map variations if needed
             # df['team'] = df['team'].replace({'BANGALORE': 'RCB', 'CHENNAI SUPER KINGS': 'CSK'})
             df['team'] = df['team'].fillna('UNKNOWN')

        return df

    def add_role_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add role-based features"""
        try:
            # Ensure role column exists and is standardized
            if 'role' not in df.columns:
                logger.warning("Role column missing, cannot add role features.")
                # Add dummy columns to prevent errors downstream
                df['is_bat'] = 0
                df['is_bowl'] = 0
                df['is_ar'] = 0
                df['is_wk'] = 0
                return df
                
            # Create role-specific indicator features using one-hot encoding
            roles = ['BAT', 'BOWL', 'AR', 'WK']
            for role in roles:
                feature_name = f'is_{role.lower()}'
                df[feature_name] = df['role'].apply(lambda x: 1 if x == role else 0)
                
            # Add role-specific value features (example - can be expanded)
            # Simplified: just use the indicators for now
            # role_value_map = { ... } # Could add typical point ranges here
            # ...
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding role features: {str(e)}")
            # Return df without role features if error
            return df
            
    def add_recent_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recent form features based on historical data"""
        try:
            # Check if we have player name for joining
            if 'player_name' not in df.columns:
                logger.warning("No player_name column found for joining with historical data")
                df['recent_form'] = 1.0
                df['last_3_avg'] = 10.0
                df['form_trend'] = 0
                return df
                
            # Check if historical data is available and valid
            if self.historical_data is None or self.historical_data.empty:
                logger.warning("No historical data available for recent form features")
                df['recent_form'] = 1.0
                df['last_3_avg'] = 10.0
                df['form_trend'] = 0
                return df
            
            # Ensure required columns in historical data
            required_cols = ['player_name', 'match_date', 'fantasy_points']
            if not all(col in self.historical_data.columns for col in required_cols):
                logger.warning(f"Historical data missing required columns: {required_cols}")
                df['recent_form'] = 1.0
                df['last_3_avg'] = 10.0
                df['form_trend'] = 0
                return df
                
            # Sort historical data by date (newest first)
            hist_df = self.historical_data.sort_values('match_date', ascending=False)
            
            # Calculate recent form metrics for each player
            # Group by player and calculate rolling/expanding averages efficiently
            hist_df['fantasy_points'] = pd.to_numeric(hist_df['fantasy_points'], errors='coerce')
            hist_df = hist_df.dropna(subset=['fantasy_points'])
            
            # Calculate metrics more efficiently
            form_metrics = hist_df.groupby('player_name')['fantasy_points'].agg(
                 last_match = lambda x: x.iloc[0] if len(x) > 0 else 0,
                 last_3_avg = lambda x: x.head(3).mean() if len(x) > 0 else 0,
                 last_5_avg = lambda x: x.head(5).mean() if len(x) > 0 else 0
            ).reset_index()

            # Calculate form trend (e.g., difference between last 2 and prev 3)
            def calculate_trend(series):
                if len(series) >= 5:
                    last_2_avg = series.head(2).mean()
                    prev_3_avg = series.iloc[2:5].mean()
                    return last_2_avg - prev_3_avg
                elif len(series) >= 2:
                     return series.head(2).mean() - series.head(2).mean() # Simplified trend if < 5 games
                return 0
                
            trend_df = hist_df.groupby('player_name')['fantasy_points'].apply(calculate_trend).reset_index()
            trend_df.rename(columns={'fantasy_points': 'form_trend'}, inplace=True)

            # Merge form metrics
            form_metrics = form_metrics.merge(trend_df, on='player_name', how='left')

            # Merge form metrics with player data
            df = df.merge(form_metrics, on='player_name', how='left')
            
            # Fill missing values with sensible defaults (e.g., using credits or overall averages)
            # Use credits as a proxy for players without historical form data
            default_form = df['credits'] * 1.0 if 'credits' in df.columns else 10.0
            df['last_3_avg'].fillna(default_form, inplace=True)
            df['last_5_avg'].fillna(default_form, inplace=True)
            df['form_trend'].fillna(0, inplace=True)
            df['last_match'].fillna(default_form * 0.8, inplace=True)
                
            # Create recent form score (weighted average of recent performances)
            df['recent_form'] = (
                0.5 * df['last_3_avg'] +
                0.3 * df['last_5_avg'] +
                0.2 * df['last_match']
            ) / 10.0 # Normalize score roughly based on credits scale
            df['recent_form'].fillna(1.0, inplace=True) # Final fallback fillna
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding recent form features: {str(e)}")
            # Add dummy features on error
            df['recent_form'] = 1.0
            df['last_3_avg'] = 10.0
            df['form_trend'] = 0
            return df
            
    def add_opposition_features(self, df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
        """Add features based on opposition team strengths/weaknesses"""
        try:
            # Standardize team codes
            home_team_std = home_team.upper().strip()
            away_team_std = away_team.upper().strip()
            
            # Check if team strengths data is available
            if self.team_strengths is None or self.team_strengths.empty:
                logger.warning("No team strengths data available")
                df['vs_team_strength'] = 1.0
                df['vs_role_advantage'] = 0.0
                return df
                
            # Ensure we have the required columns in team strengths data
            required_cols = ['team', 'batting_strength', 'bowling_strength']
            if not all(col in self.team_strengths.columns for col in required_cols):
                logger.warning(f"Team strengths data missing required columns: {required_cols}")
                df['vs_team_strength'] = 1.0
                df['vs_role_advantage'] = 0.0
                return df
                
            # Get team data
            home_data = self.team_strengths[self.team_strengths['team'] == home_team_std]
            away_data = self.team_strengths[self.team_strengths['team'] == away_team_std]
            
            if home_data.empty or away_data.empty:
                logger.warning(f"Missing team strength data for {home_team_std} or {away_team_std}. Using defaults.")
                # Use default strengths if team data is missing
                default_strength = 0.8 # Example default
                home_batting = home_data.iloc[0]['batting_strength'] if not home_data.empty else default_strength
                home_bowling = home_data.iloc[0]['bowling_strength'] if not home_data.empty else default_strength
                away_batting = away_data.iloc[0]['batting_strength'] if not away_data.empty else default_strength
                away_bowling = away_data.iloc[0]['bowling_strength'] if not away_data.empty else default_strength
            else:
                home_batting = home_data.iloc[0]['batting_strength']
                home_bowling = home_data.iloc[0]['bowling_strength']
                away_batting = away_data.iloc[0]['batting_strength']
                away_bowling = away_data.iloc[0]['bowling_strength']
            
            # For each player, determine their team and opposition
            if 'team' not in df.columns:
                logger.warning("No team column found in player data")
                df['vs_team_strength'] = 1.0
                df['vs_role_advantage'] = 0.0
                return df
                
            # Assign opposition strengths based on player's team
            df['vs_team_batting'] = df['team'].apply(lambda x: away_batting if x == home_team_std else home_batting)
            df['vs_team_bowling'] = df['team'].apply(lambda x: away_bowling if x == home_team_std else home_bowling)
            
            # Calculate role-specific advantage against opposition
            df['vs_role_advantage'] = 0.0 # Initialize
            if 'role' in df.columns:
                # Batsmen (BAT, WK) advantage vs opposition bowling
                bat_mask = df['role'].isin(['BAT', 'WK'])
                # Avoid division by zero
                df.loc[bat_mask, 'vs_role_advantage'] = 1.0 / df.loc[bat_mask, 'vs_team_bowling'].replace(0, 1e-6) 
                
                # Bowlers advantage vs opposition batting
                bowl_mask = df['role'] == 'BOWL'
                df.loc[bowl_mask, 'vs_role_advantage'] = 1.0 / df.loc[bowl_mask, 'vs_team_batting'].replace(0, 1e-6)
                
                # All-rounders get average of both advantages
                ar_mask = df['role'] == 'AR'
                bat_adv = 1.0 / df.loc[ar_mask, 'vs_team_bowling'].replace(0, 1e-6)
                bowl_adv = 1.0 / df.loc[ar_mask, 'vs_team_batting'].replace(0, 1e-6)
                df.loc[ar_mask, 'vs_role_advantage'] = (bat_adv + bowl_adv) / 2.0
            else:
                df['vs_role_advantage'] = 1.0  # Neutral if role not available
                
            # Create an overall opposition strength score
            df['vs_team_strength'] = (df['vs_team_batting'] + df['vs_team_bowling']) / 2.0
            
            # Fill NaNs that might have occurred
            df['vs_role_advantage'].fillna(1.0, inplace=True)
            df['vs_team_strength'].fillna(1.0, inplace=True)

            return df
            
        except Exception as e:
            logger.error(f"Error adding opposition features: {str(e)}")
            # Add dummy features on error
            df['vs_team_strength'] = 1.0
            df['vs_role_advantage'] = 0.0
            return df
            
    def add_venue_features(self, df: pd.DataFrame, venue: str) -> pd.DataFrame:
        """Add venue-specific features"""
        try:
            # Check if venue data is available
            if self.venue_data is None or self.venue_data.empty:
                logger.warning("No venue data available")
                df['venue_advantage'] = 1.0
                df['pitch_factor'] = 1.0
                df['bowling_factor'] = 1.0
                df['pitch_is_batting_friendly'] = 0
                df['pitch_is_bowling_friendly'] = 0
                df['pitch_is_balanced'] = 1
                return df
                
            # Get venue data
            venue_row = self.venue_data[self.venue_data['venue'] == venue]
            
            if venue_row.empty:
                logger.warning(f"No data found for venue: {venue}. Using default venue stats.")
                venue_row = self.venue_data[self.venue_data['venue'] == 'default']
                
                if venue_row.empty:
                     # If even default is missing, use hardcoded defaults
                     logger.error("Default venue data also missing. Using hardcoded defaults.")
                     pitch_type = 'balanced'
                     spin_wickets = 4
                     pace_wickets = 6
                else:
                     venue_metrics = venue_row.iloc[0]
                     pitch_type = venue_metrics.get('pitch_type', 'balanced')
                     spin_wickets = venue_metrics.get('avg_spin_wickets_per_match', 4)
                     pace_wickets = venue_metrics.get('avg_pace_wickets_per_match', 6)
            else:
                 venue_metrics = venue_row.iloc[0]
                 pitch_type = venue_metrics.get('pitch_type', 'balanced')
                 spin_wickets = venue_metrics.get('avg_spin_wickets_per_match', 4)
                 pace_wickets = venue_metrics.get('avg_pace_wickets_per_match', 6)
                 
            # Create pitch type features (one-hot encoded)
            df['pitch_is_batting_friendly'] = 1 if pitch_type == 'batting_friendly' else 0
            df['pitch_is_bowling_friendly'] = 1 if pitch_type == 'bowling_friendly' else 0
            df['pitch_is_balanced'] = 1 if pitch_type == 'balanced' else 0
            
            # Calculate role-specific venue advantages
            df['pitch_factor'] = 1.0
            df['bowling_factor'] = 1.0 # Placeholder for potential bowling style features

            if 'role' in df.columns:
                # Calculate spin/pace factor (simplistic)
                total_wickets = spin_wickets + pace_wickets
                if total_wickets > 0:
                     spin_factor = spin_wickets / total_wickets
                     pace_factor = pace_wickets / total_wickets
                else:
                     spin_factor = 0.5
                     pace_factor = 0.5
                
                # Example: Add bowling factor based on average pitch behavior (could refine with player style)
                # Spinners benefit slightly more on spin-friendly, pacers on pace-friendly
                # For now, let's keep bowling_factor simple or integrate later if bowling_style is added
                # df['bowling_factor'] = df['role'].apply(lambda r: spin_factor * 1.1 if r == 'BOWL' and pitch_type == 'bowling_friendly' else (pace_factor * 1.1 if r == 'BOWL' and pitch_type == 'batting_friendly' else 1.0)) 

                # Apply pitch type advantages based on player role
                if pitch_type == 'batting_friendly':
                    df.loc[df['role'].isin(['BAT', 'WK']), 'pitch_factor'] = 1.2
                    df.loc[df['role'] == 'BOWL', 'pitch_factor'] = 0.8
                elif pitch_type == 'bowling_friendly':
                    df.loc[df['role'].isin(['BAT', 'WK']), 'pitch_factor'] = 0.8
                    df.loc[df['role'] == 'BOWL', 'pitch_factor'] = 1.2
                # Balanced pitch keeps factor at 1.0 (or could have slight adjustments)
            
            # Combine factors into venue advantage
            # Using only pitch_factor for now, as bowling_factor is basic
            df['venue_advantage'] = df['pitch_factor'] 
            
            # Fill NaNs
            df['pitch_factor'].fillna(1.0, inplace=True)
            df['bowling_factor'].fillna(1.0, inplace=True)
            df['venue_advantage'].fillna(1.0, inplace=True)

            return df
            
        except Exception as e:
            logger.error(f"Error adding venue features: {str(e)}")
            # Add dummy features on error
            df['venue_advantage'] = 1.0
            df['pitch_factor'] = 1.0
            df['bowling_factor'] = 1.0
            df['pitch_is_batting_friendly'] = 0
            df['pitch_is_bowling_friendly'] = 0
            df['pitch_is_balanced'] = 1
            return df
            
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between existing features"""
        try:
            # Ensure base features exist before creating interactions
            base_features = ['credits', 'recent_form', 'role', 'pitch_factor', 'vs_role_advantage', 'venue_advantage']
            if not all(col in df.columns for col in base_features if col is not None): # Check if base cols exist
                 logger.warning("Base features missing, skipping interaction features.")
                 return df

            # Safely create interaction features
            if 'credits' in df.columns and 'recent_form' in df.columns:
                df['credits_form_interaction'] = df['credits'] * df['recent_form']
            else: df['credits_form_interaction'] = 0
                
            if 'role' in df.columns and 'credits' in df.columns: # Example: Credits * is_wk indicator
                if 'is_wk' in df.columns:
                     df['credits_wk_interaction'] = df['credits'] * df['is_wk']
                else: df['credits_wk_interaction'] = 0
                    
            if 'pitch_factor' in df.columns and 'role' in df.columns:
                 # Example: pitch_factor * is_bat indicator
                 if 'is_bat' in df.columns:
                     df['pitch_bat_interaction'] = df['pitch_factor'] * df['is_bat']
                 else: df['pitch_bat_interaction'] = 0
                 if 'is_bowl' in df.columns:
                      df['pitch_bowl_interaction'] = df['pitch_factor'] * df['is_bowl']
                 else: df['pitch_bowl_interaction'] = 0
                
            if 'recent_form' in df.columns and 'vs_role_advantage' in df.columns:
                df['form_opposition_interaction'] = df['recent_form'] * df['vs_role_advantage']
            else: df['form_opposition_interaction'] = 0
                
            if 'credits' in df.columns and 'venue_advantage' in df.columns:
                df['credits_venue_interaction'] = df['credits'] * df['venue_advantage']
            else: df['credits_venue_interaction'] = 0
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding interaction features: {str(e)}")
            # Return df without interactions if error
            return df
            
# Test function (Optional: Can be run standalone)
def test_feature_engineering():
    """Test the feature engineering module with sample data"""
    print("Testing Feature Engineering Module...")
    # Create sample player data
    sample_data = pd.DataFrame({
        'Player Name': ['Virat Kohli', 'Rohit Sharma', 'Jasprit Bumrah', 'MS Dhoni'],
        'Team': ['RCB', 'MI', 'MI', 'CSK'],
        'Player Type': ['BAT', 'BAT', 'BOWL', 'WK'],
        'Credits': [10.0, 9.5, 9.0, 8.5]
    })
    
    # Initialize feature engineer (assuming dataset dir exists or defaults work)
    try:
         engineer = FeatureEngineer()
    except Exception as init_err:
         print(f"Error initializing FeatureEngineer: {init_err}")
         return

    # Apply feature engineering for a specific match context
    enhanced_data = engineer.enhance_player_features(
        sample_data.copy(), 
        home_team='CSK', 
        away_team='MI', 
        venue='MA Chidambaram Stadium' # Balanced pitch example
    )
    
    # Print results
    print("\nOriginal Data:")
    print(sample_data)
    
    print("\nEnhanced Data (Balanced Pitch):")
    print(enhanced_data)
    
    new_features = [col for col in enhanced_data.columns if col not in sample_data.columns and col not in ['player_name', 'role', 'team', 'credits']] # Show only added features
    print(f"\nNew Features Added ({len(new_features)}): {new_features}")
    print("\nSpecific Feature Values (Example - Virat Kohli):")
    print(enhanced_data[enhanced_data['player_name'] == 'Virat Kohli'][new_features].iloc[0])

    # Test another context
    enhanced_data_batting = engineer.enhance_player_features(
        sample_data.copy(), 
        home_team='MI', 
        away_team='RCB', 
        venue='Wankhede Stadium' # Batting pitch example
    )
    print("\nEnhanced Data (Batting Pitch - Wankhede):")
    print(enhanced_data_batting[new_features].head())
    
    return enhanced_data

if __name__ == "__main__":
    test_feature_engineering() 