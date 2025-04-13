import pandas as pd
import logging
import traceback
import os
from typing import Optional
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Tuple

# Configure logging
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
        self._load_reference_data()

    def _load_reference_data(self):
        """Load venue and historical data"""
        try:
            # Load venue data if available
            venue_file = os.path.join(self.data_dir, "venue_stats.csv")
            if os.path.exists(venue_file):
                self.venue_data = pd.read_csv(venue_file)
                # Ensure the venue column name matches expected usage later
                if 'venue_name' in self.venue_data.columns and 'venue' not in self.venue_data.columns:
                     self.venue_data = self.venue_data.rename(columns={'venue_name': 'venue'})
                elif 'venue' not in self.venue_data.columns:
                     logger.error("Venue stats file missing 'venue' or 'venue_name' column.")
                     self.venue_data = None # Invalidate if no venue identifier

                if self.venue_data is not None:
                    logger.info(f"Loaded venue data: {len(self.venue_data)} venues")
            else:
                logger.warning(f"Venue data file not found at {venue_file}")
                self.venue_data = None # Set to None if file not found

            # Load historical player performance data if available
            historical_file = os.path.join(self.data_dir, "player_history.csv")
            if os.path.exists(historical_file):
                self.historical_data = pd.read_csv(historical_file)
                if 'match_date' in self.historical_data.columns:
                    original_rows = len(self.historical_data)
                    # Attempt date conversion with multiple formats
                    formats_to_try = [None, '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d-%b-%y', '%Y/%m/%d'] # Add more if needed
                    converted = False
                    for fmt in formats_to_try:
                        try:
                            # Try converting on a copy first to avoid partial conversions
                            date_col_converted = pd.to_datetime(self.historical_data['match_date'], format=fmt, errors='coerce')
                            # If this format worked for a significant portion, use it
                            if not date_col_converted.isnull().all():
                                self.historical_data['match_date'] = date_col_converted
                                logger.info(f"Successfully parsed historical dates using format: {fmt or 'default'}")
                                converted = True
                                break # Stop trying formats once one works
                        except ValueError:
                            continue # Ignore format errors and try the next one
                        except Exception as date_err:
                             logger.error(f"Unexpected error during date parsing trial with format {fmt}: {date_err}")
                             continue # Try next format
                             
                    if not converted:
                         logger.error("Failed to parse historical match_date with any known format. Dates will be unusable.")
                         # Keep the column but it will likely be NaT or object type
                         self.historical_data['match_date'] = pd.to_datetime(self.historical_data['match_date'], errors='coerce') # Final attempt, likely results in NaT
                         
                    # Drop rows where date conversion ultimately failed
                    self.historical_data.dropna(subset=['match_date'], inplace=True)
                    rows_after_drop = len(self.historical_data)
                    if rows_after_drop == 0 and original_rows > 0:
                        logger.error("Historical data became empty after dropping rows with unparseable dates. Check source data and formats in player_history.csv!")
                    elif original_rows > rows_after_drop:
                        logger.warning(f"Dropped {original_rows - rows_after_drop} rows from historical data due to unparseable dates.")
                else:
                     logger.warning("'match_date' column not found in historical data.")
                     
                logger.info(f"Loaded historical data: {len(self.historical_data)} records")
            else:
                logger.warning(f"Historical data file not found at {historical_file}")
                self.historical_data = None # Explicitly set to None
                
            # Load team strengths...
            # ... (team strength loading code remains the same) ...

        except Exception as load_err:
            logger.error(f"Error loading reference data: {load_err}")
            traceback.print_exc() # Print full traceback for loading errors
            # Ensure attributes are None if loading failed
            self.venue_data = None
            self.historical_data = None
            self.team_strengths = None
            
    def enhance_player_features(self, df, **kwargs):
        """
        Add engineered features to player DataFrame
        
        Args:
            df (pd.DataFrame): Player data with basic features
            **kwargs: Additional keyword arguments including:
                - home_team (str): Home team code
                - away_team (str): Away team code
                - venue (str): Match venue
        """
        if not isinstance(df, pd.DataFrame):
            logger.error(f"enhance_player_features received non-DataFrame input: {type(df)}")
            return None
            
        if df.empty:
            logger.warning("enhance_player_features received empty DataFrame")
            return df
            
        try:
            logger.debug(f"Starting enhance_player_features with DataFrame shape: {df.shape}")
            logger.debug(f"Initial columns: {df.columns.tolist()}")
            
            # Add match info from kwargs if not already in DataFrame
            if 'home_team' in kwargs and 'home_team' not in df.columns:
                df['home_team'] = kwargs['home_team']
            if 'away_team' in kwargs and 'away_team' not in df.columns:
                df['away_team'] = kwargs['away_team']
            if 'venue' in kwargs and 'venue' not in df.columns:
                df['venue'] = kwargs['venue']
            
            # Step 1: Standardize column names
            try:
                df = self._standardize_columns(df)
                if not isinstance(df, pd.DataFrame):
                    logger.error(f"_standardize_columns returned non-DataFrame: {type(df)}")
                    return None
                logger.debug(f"After standardizing columns, shape: {df.shape}")
                logger.debug(f"Columns after standardization: {df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Error in _standardize_columns: {str(e)}")
                traceback.print_exc()
                return None
                
            # Step 2: Add role-based features
            try:
                df = self.add_role_features(df)
                if not isinstance(df, pd.DataFrame):
                    logger.error(f"add_role_features returned non-DataFrame: {type(df)}")
                    return None
                logger.debug(f"After adding role features, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error in add_role_features: {str(e)}")
                traceback.print_exc()
                return None
                
            # Step 3: Add recent form features
            try:
                df = self.add_recent_form_features(df)
                if not isinstance(df, pd.DataFrame):
                    logger.error(f"add_recent_form_features returned non-DataFrame: {type(df)}")
                    return None
                logger.debug(f"After adding recent form features, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error in add_recent_form_features: {str(e)}")
                traceback.print_exc()
                return None
                
            # Step 4: Add opposition features
            try:
                df = self.add_opposition_features(df)
                if not isinstance(df, pd.DataFrame):
                    logger.error(f"add_opposition_features returned non-DataFrame: {type(df)}")
                    return None
                logger.debug(f"After adding opposition features, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error in add_opposition_features: {str(e)}")
                traceback.print_exc()
                return None
                
            # Step 5: Add venue features
            try:
                # Extract venue from DataFrame if available, otherwise use default
                venue = 'Unknown Venue'
                if 'venue' in df.columns:
                    # Get the most common venue value (in case there are multiple)
                    venue_vals = df['venue'].value_counts()
                    if not venue_vals.empty:
                        venue = venue_vals.index[0]
                elif 'venue' in kwargs:
                    venue = kwargs['venue']
                
                df = self.add_venue_features(df, venue)
                if not isinstance(df, pd.DataFrame):
                    logger.error(f"add_venue_features returned non-DataFrame: {type(df)}")
                    return None
                logger.debug(f"After adding venue features, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error in add_venue_features: {str(e)}")
                traceback.print_exc()
                return None
                
            # Step 6: Add interaction features
            try:
                df = self.add_interaction_features(df)
                if not isinstance(df, pd.DataFrame):
                    logger.error(f"add_interaction_features returned non-DataFrame: {type(df)}")
                    return None
                logger.debug(f"After adding interaction features, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error in add_interaction_features: {str(e)}")
                traceback.print_exc()
                return None
                
            logger.debug(f"Successfully completed enhance_player_features, final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Unexpected error in enhance_player_features: {str(e)}")
            traceback.print_exc()
            return None

    def _standardize_columns(self, df):
        """Standardize column names and format values for consistency"""
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Cannot standardize columns: Input is not a valid DataFrame")
            return df
            
        try:
            # Map common column name variations to standard names
            column_mapping = {
                'Player Name': 'player_name',
                'PlayerName': 'player_name',
                'Name': 'player_name',
                'player': 'player_name',
                
                'Player Type': 'role',
                'PlayerType': 'role',
                'Type': 'role',
                'player_type': 'role',
                
                'Team': 'team',
                'team_code': 'team',
                'TeamCode': 'team',
                'TeamName': 'team',
                
                'Credits': 'credits',
                'Cost': 'credits',
                'Price': 'credits',
                'credit': 'credits'
            }
            
            # Apply column name mapping where needed
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    # For Credits, preserve both columns for backward compatibility
                    if old_col == 'Credits':
                        df[new_col] = df[old_col].copy()
                    else:
                        df = df.rename(columns={old_col: new_col})
                    
            # Ensure required columns exist
            for required_col in ['player_name', 'role', 'team', 'credits']:
                if required_col not in df.columns:
                    logger.warning(f"Required column '{required_col}' not found in data after standardization. Adding placeholder.")
                    if required_col == 'player_name':
                        # Use index as placeholder
                        df[required_col] = 'Player_' + df.index.astype(str)
                    elif required_col == 'role':
                        # Default to generic role
                        df[required_col] = 'BAT'
                    elif required_col == 'team':
                        # Default to unknown team
                        df[required_col] = 'UNKNOWN'
                    elif required_col == 'credits':
                        # Default to average credits
                        df[required_col] = 8.0
                        
            # Standardize role values
            if 'role' in df.columns:
                # Convert role values to uppercase strings
                try:
                    df['role'] = df['role'].astype(str).str.upper()
                    
                    # Map role values to standard values (WK, BAT, AR, BOWL)
                    role_mapping = {
                        'WK': ['WK', 'WICKET KEEPER', 'WICKET-KEEPER', 'KEEPER', 'WICKETKEEPER', 'W/K', 'WICKET_KEEPER'],
                        'BAT': ['BAT', 'BATSMAN', 'BATSMEN', 'BATTER', 'BATTERS', 'BATTING'],
                        'AR': ['AR', 'ALL', 'ALL ROUNDER', 'ALL-ROUNDER', 'ALLROUNDER', 'ALL ROUNDERS', 'ALL-ROUNDERS', 'A/R'],
                        'BOWL': ['BOWL', 'BOWLER', 'BOWLERS', 'BOWLING', 'FAST BOWLER', 'SPINNER', 'PACE BOWLER']
                    }
                    
                    # Handle special case for 'ROLE' literal value
                    df.loc[df['role'] == 'ROLE', 'role'] = 'BAT'
                    
                    # Apply role mapping
                    for std_role, variants in role_mapping.items():
                        for variant in variants:
                            df.loc[df['role'] == variant, 'role'] = std_role
                            
                    # Ensure only valid roles (default to BAT for unrecognized roles)
                    valid_roles = ['WK', 'BAT', 'AR', 'BOWL']
                    df.loc[~df['role'].isin(valid_roles), 'role'] = 'BAT'
                except Exception as e:
                    logger.error(f"Error standardizing roles: {str(e)}")
                    # Ensure role column exists but doesn't break the pipeline
                    if 'role' in df.columns:
                        df['role'] = 'BAT'  # Safe default
                        
            # Ensure credits are numeric
            if 'credits' in df.columns:
                try:
                    df['credits'] = pd.to_numeric(df['credits'], errors='coerce')
                    # Fill any NaN values with a default
                    df['credits'].fillna(8.0, inplace=True)
                    
                    # Also ensure 'Credits' is available and numeric if it exists
                    if 'Credits' in df.columns:
                        df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce')
                        df['Credits'].fillna(8.0, inplace=True)
                    else:
                        # Add 'Credits' column for backwards compatibility
                        df['Credits'] = df['credits'].copy()
                except Exception as e:
                    logger.error(f"Error converting credits to numeric: {str(e)}")
                    
            # Add role value feature based on role
            df['role_value'] = 1.0  # Default
            if 'role' in df.columns:
                role_values = {'WK': 1.0, 'BAT': 1.2, 'AR': 1.4, 'BOWL': 1.1}
                for role, value in role_values.items():
                    df.loc[df['role'] == role, 'role_value'] = value
                    
            return df
            
        except Exception as e:
            logger.error(f"Exception during column standardization: {str(e)}")
            traceback.print_exc()
            return df  # Return original DataFrame on error to avoid pipeline interruption

    def add_role_features(self, df):
        """
        Add role-specific features based on player role.
        
        Args:
            df (pd.DataFrame): Player DataFrame with 'role' column
            
        Returns:
            pd.DataFrame: DataFrame with added role-specific features
        """
        logger.debug(f"Entering add_role_features. Input df shape: {df.shape if isinstance(df, pd.DataFrame) else 'Invalid'}")
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Input df to add_role_features is invalid or empty.")
            return df
            
        try:
            # Check if role column exists
            if 'role' not in df.columns:
                logger.warning("'role' column not found. Adding default role features.")
                # Add default role features
                df['is_batsman'] = 0
                df['is_bowler'] = 0
                df['is_all_rounder'] = 0
                df['is_wicket_keeper'] = 0
                df['role_value'] = 1.0  # Default value
                return df
                
            # Ensure role values are strings and standard formats
            # (This should have been done in _standardize_columns already, but double-check)
            try:
                df['role'] = df['role'].astype(str).str.upper()
            except Exception as e:
                logger.warning(f"Error standardizing role values: {str(e)}")
                # Continue with what we have
            
            # Add one-hot encoded role features
            df['is_batsman'] = (df['role'] == 'BAT').astype(int)
            df['is_bowler'] = (df['role'] == 'BOWL').astype(int)
            df['is_all_rounder'] = (df['role'] == 'AR').astype(int)
            df['is_wicket_keeper'] = (df['role'] == 'WK').astype(int)
            
            # Add role value if not already added in _standardize_columns
            if 'role_value' not in df.columns:
                df['role_value'] = 1.0  # Default value
                role_values = {'WK': 1.0, 'BAT': 1.2, 'AR': 1.4, 'BOWL': 1.1}
                for role, value in role_values.items():
                    df.loc[df['role'] == role, 'role_value'] = value
            
            # Add expected fantasy point ranges by role
            # These are arbitrary starting values that models will adjust based on data
            point_ranges = {
                'BAT': {'min': 0, 'max': 100, 'avg': 25},
                'BOWL': {'min': 0, 'max': 75, 'avg': 20},
                'AR': {'min': 0, 'max': 125, 'avg': 30},
                'WK': {'min': 0, 'max': 90, 'avg': 22}
            }
            
            df['role_avg_points'] = df['role'].map({role: values['avg'] for role, values in point_ranges.items()})
            df['role_min_points'] = df['role'].map({role: values['min'] for role, values in point_ranges.items()})
            df['role_max_points'] = df['role'].map({role: values['max'] for role, values in point_ranges.items()})
            
            # Fill NaN values with defaults
            df['role_avg_points'] = df['role_avg_points'].fillna(25)
            df['role_min_points'] = df['role_min_points'].fillna(0)
            df['role_max_points'] = df['role_max_points'].fillna(100)
            
            # Add team role weights
            # This feature represents how valuable the role is in a balanced team
            # WK: Limited slots (1-2), BAT: Multiple slots (3-5), etc.
            df['team_role_weight'] = 1.0
            role_weights = {'WK': 1.2, 'BAT': 1.0, 'BOWL': 1.1, 'AR': 1.3}
            for role, weight in role_weights.items():
                df.loc[df['role'] == role, 'team_role_weight'] = weight
                
            logger.debug(f"Exiting add_role_features successfully. Df shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Exception during role feature addition: {str(e)}")
            traceback.print_exc()
            # Ensure basic role features exist even on error
            try:
                df['is_batsman'] = 0
                df['is_bowler'] = 0
                df['is_all_rounder'] = 0
                df['is_wicket_keeper'] = 0
                df['role_value'] = 1.0  # Default value
                logger.debug(f"Added fallback role features after exception. Df shape: {df.shape}")
                return df
            except:
                logger.error("Failed to add even fallback role features due to invalid DataFrame.")
                return df  # Return whatever we have

    def add_recent_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recent form features based on historical data"""
        # --- Start Debugging Enhancements ---
        logger.debug("Entering add_recent_form_features") # DEBUG
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Input df to add_recent_form_features is invalid or empty.")
            # Ensure necessary columns exist even if we return early
            for col in ['recent_form', 'last_3_avg', 'form_trend']:
                 if col not in df.columns:
                     # Use safe assignment if df is potentially not a DataFrame
                     try:
                         df[col] = 1.0 if col == 'recent_form' else (10.0 if col == 'last_3_avg' else 0)
                     except: # Catch potential errors if df is not DataFrame
                         pass 
            return df
        # --- End Debugging Enhancements ---
        
        original_player_data_for_fallback = df.copy() # Keep original for safer fallback

        try:
            # Check if we have player name for joining
            if 'player_name' not in df.columns:
                logger.warning("'player_name' column not found in input DataFrame. Cannot calculate form features.")
                # Add dummy features and return
                for col in ['recent_form', 'last_3_avg', 'form_trend', 'last_match', 'last_5_avg']:
                    df[col] = 1.0 if col == 'recent_form' else (10.0 if col in ['last_3_avg', 'last_5_avg', 'last_match'] else 0)
                return df
                
            # Check if historical data is available and valid (already done in enhance_player_features, but double check)
            if self.historical_data is None or self.historical_data.empty:
                logger.warning("Historical data is None or empty in add_recent_form_features. Adding dummy form features.")
                for col in ['recent_form', 'last_3_avg', 'form_trend', 'last_match', 'last_5_avg']:
                    df[col] = 1.0 if col == 'recent_form' else (10.0 if col in ['last_3_avg', 'last_5_avg', 'last_match'] else 0)
                return df
            
            # Ensure required columns in historical data
            required_cols = ['player_name', 'match_date', 'fantasy_points']
            if not all(col in self.historical_data.columns for col in required_cols):
                logger.warning(f"Historical data missing required columns: {required_cols}. Adding dummy form features.")
                for col in ['recent_form', 'last_3_avg', 'form_trend', 'last_match', 'last_5_avg']:
                    df[col] = 1.0 if col == 'recent_form' else (10.0 if col in ['last_3_avg', 'last_5_avg', 'last_match'] else 0)
                return df
                
            # Sort historical data by date (newest first)
            logger.debug("Sorting historical data...") # DEBUG
            hist_df = self.historical_data.sort_values('match_date', ascending=False)
            
            # Calculate recent form metrics for each player
            logger.debug("Calculating form metrics (agg)...") # DEBUG
            hist_df['fantasy_points'] = pd.to_numeric(hist_df['fantasy_points'], errors='coerce')
            hist_df = hist_df.dropna(subset=['fantasy_points'])
            
            form_metrics = hist_df.groupby('player_name')['fantasy_points'].agg(
                 last_match = lambda x: x.iloc[0] if len(x) > 0 else 0,
                 last_3_avg = lambda x: x.head(3).mean() if len(x) > 0 else 0,
                 last_5_avg = lambda x: x.head(5).mean() if len(x) > 0 else 0
            ).reset_index()
            logger.debug(f"Calculated form_metrics shape: {form_metrics.shape}") # DEBUG
            if not isinstance(form_metrics, pd.DataFrame):
                 logger.error("form_metrics calculation did not return a DataFrame!")
                 raise TypeError("form_metrics is not a DataFrame")

            # Calculate form trend (e.g., difference between last 2 and prev 3)
            logger.debug("Calculating form trend...") # DEBUG
            def calculate_trend(series):
                if len(series) >= 5:
                    last_2_avg = series.head(2).mean()
                    prev_3_avg = series.iloc[2:5].mean()
                    return last_2_avg - prev_3_avg
                elif len(series) >= 2:
                     # Simplified trend if < 5 games: diff between last game and avg of last 2
                     return series.iloc[0] - series.head(2).mean() 
                return 0
                
            trend_df = hist_df.groupby('player_name')['fantasy_points'].apply(calculate_trend).reset_index()
            trend_df.rename(columns={'fantasy_points': 'form_trend'}, inplace=True)
            logger.debug(f"Calculated trend_df shape: {trend_df.shape}") # DEBUG
            if not isinstance(trend_df, pd.DataFrame):
                 logger.error("trend_df calculation did not return a DataFrame!")
                 raise TypeError("trend_df is not a DataFrame")

            # Merge form metrics
            logger.debug("Merging form_metrics and trend_df...") # DEBUG
            form_metrics = form_metrics.merge(trend_df, on='player_name', how='left')
            logger.debug(f"Shape after merging trend: {form_metrics.shape}") # DEBUG
            if not isinstance(form_metrics, pd.DataFrame):
                 logger.error("form_metrics became invalid after merging trend!")
                 raise TypeError("form_metrics is not a DataFrame after trend merge")

            # Merge form metrics with player data
            logger.debug(f"Merging form metrics with input df (shape {df.shape})...") # DEBUG
            # Check for player_name column before merge
            if 'player_name' not in df.columns:
                 logger.error("'player_name' column missing just before final merge.")
                 # Handle error - maybe return original df with dummies
                 raise KeyError("Missing 'player_name' column before merging form features")
            
            df = df.merge(form_metrics, on='player_name', how='left')
            logger.debug(f"Shape after merging form metrics: {df.shape}") # DEBUG
            # --- CRITICAL CHECK --- 
            if not isinstance(df, pd.DataFrame) or df.empty:
                 logger.error("df became invalid or empty AFTER merging form_metrics!")
                 # Attempt to return original df structure with dummy values
                 original_df_with_dummies = original_player_data_for_fallback.copy()
                 for col in ['recent_form', 'last_3_avg', 'form_trend', 'last_match', 'last_5_avg']:
                     original_df_with_dummies[col] = 1.0 if col == 'recent_form' else (10.0 if col in ['last_3_avg', 'last_5_avg', 'last_match'] else 0)
                 return original_df_with_dummies
            # --- END CRITICAL CHECK ---
            
            # Fill missing values with sensible defaults (e.g., using credits or overall averages)
            logger.debug("Filling NA values for form features...") # DEBUG
            # ... (rest of the function including final form score calculation and return) ...

            return df
            
        except Exception as e:
            logger.error(f"Exception during recent_form_features calculation: {e}")
            traceback.print_exc() # Print full traceback
            # Return the original DataFrame on any exception within the try block
            logger.warning("Returning original player data due to exception during recent_form_features calculation.")
            return player_data

    def add_opposition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to opposition team strength
        
        Args:
            df (pd.DataFrame): Player DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added opposition features
        """
        logger.debug(f"Entering add_opposition_features. Input df shape: {df.shape if isinstance(df, pd.DataFrame) else 'Invalid'}")
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Input df to add_opposition_features is invalid or empty.")
            return df
        
        try:
            # Check if team strengths data is available
            if self.team_strengths is None or self.team_strengths.empty:
                logger.warning("Team strengths data is not available. Adding default opposition features.")
                # Add default opposition features
                df['opposition_strength'] = 1.0
                df['is_favorable_matchup'] = 0
                logger.debug(f"Exiting add_opposition_features after adding defaults. Df shape: {df.shape}")
                return df
                
            # Check if we have team and opposition info
            required_cols = ['team', 'home_team', 'away_team']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns for opposition features: {missing_cols}. Adding default features.")
                df['opposition_strength'] = 1.0
                df['is_favorable_matchup'] = 0
                logger.debug(f"Exiting add_opposition_features after missing columns. Df shape: {df.shape}")
                return df
                
            # Standardize team names
            logger.debug("Standardizing team names for opposition analysis") # DEBUG
            for col in ['team', 'home_team', 'away_team']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.upper()
            
            # Determine opposition team for each player
            logger.debug("Determining opposition for each player") # DEBUG
            df['opposition'] = df.apply(
                lambda row: row['away_team'] if row['team'] == row['home_team'] else row['home_team'], 
                axis=1
            )
            
            # Add opposition strength metric from team_strengths data
            logger.debug("Adding opposition strength metrics") # DEBUG
            for index, row in df.iterrows():
                player_team = row['team']
                opposition = row['opposition']
                
                # Get player team strength
                player_team_strength = 1.0  # Default
                if player_team in self.team_strengths.index:
                    player_team_strength = self.team_strengths.loc[player_team, 'overall_strength']
                else:
                    logger.warning(f"Player team '{player_team}' not found in team strengths data.")
                
                # Get opposition team strength
                opposition_strength = 1.0  # Default
                if opposition in self.team_strengths.index:
                    opposition_strength = self.team_strengths.loc[opposition, 'overall_strength']
                else:
                    logger.warning(f"Opposition team '{opposition}' not found in team strengths data.")
                
                # Set opposition strength relative to player's team (normalized)
                df.at[index, 'opposition_strength'] = opposition_strength / max(player_team_strength, 0.01)
                
                # Determine if this is a favorable matchup (player team stronger than opposition)
                df.at[index, 'is_favorable_matchup'] = 1 if player_team_strength > opposition_strength else 0
            
            # Add role-specific opposition features
            if 'role' in df.columns and 'opposition' in df.columns:
                for index, row in df.iterrows():
                    role = row['role'] if pd.notna(row['role']) else 'BAT'  # Default to BAT if missing
                    opposition = row['opposition']
                    
                    # Default metrics
                    bowling_difficulty = 1.0
                    batting_difficulty = 1.0
                    
                    # Get team-specific metrics if available
                    if opposition in self.team_strengths.index:
                        bowling_difficulty = self.team_strengths.loc[opposition, 'bowling_strength']
                        batting_difficulty = self.team_strengths.loc[opposition, 'batting_strength']
                    
                    # Apply role-specific adjustments
                    if role == 'BAT' or role == 'WK':
                        # Batsmen face opposition bowling
                        df.at[index, 'opposition_role_factor'] = 1.0 / max(bowling_difficulty, 0.01)
                    elif role == 'BOWL':
                        # Bowlers face opposition batting
                        df.at[index, 'opposition_role_factor'] = 1.0 / max(batting_difficulty, 0.01)
                    elif role == 'AR':
                        # All-rounders affected by both
                        df.at[index, 'opposition_role_factor'] = 1.0 / max((bowling_difficulty + batting_difficulty) / 2, 0.01)
                    else:
                        # Default
                        df.at[index, 'opposition_role_factor'] = 1.0
            else:
                logger.warning("'role' or 'opposition' column not found for role-specific opposition factors.")
                df['opposition_role_factor'] = 1.0
                        
            logger.debug(f"Exiting add_opposition_features successfully. Df shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Exception during opposition feature addition: {str(e)}")
            traceback.print_exc()
            # Ensure default features are added even on error
            df['opposition_strength'] = 1.0
            df['is_favorable_matchup'] = 0
            df['opposition_role_factor'] = 1.0
            logger.warning(f"Exiting add_opposition_features after exception. Df shape: {df.shape}")
            return df

    def add_venue_features(self, df: pd.DataFrame, venue: str) -> pd.DataFrame:
        """
        Add venue-specific features based on historical venue data
        
        Args:
            df (pd.DataFrame): Player DataFrame
            venue (str): Match venue
            
        Returns:
            pd.DataFrame: DataFrame with added venue features
        """
        logger.debug(f"Entering add_venue_features. Input df shape: {df.shape if isinstance(df, pd.DataFrame) else 'Invalid'}, Venue: {venue}")
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Input df to add_venue_features is invalid or empty.")
            return df
        
        try:
            logger.debug("Checking venue data...") # DEBUG
            # Check if venue data is available
            if self.venue_data is None or self.venue_data.empty:
                logger.warning("Venue data is not available. Adding default venue features.")
                # Add default venue features
                df['venue_advantage'] = 1.0
                df['pitch_factor'] = 1.0
                df['bowling_factor'] = 1.0
                df['pitch_is_batting_friendly'] = 0
                df['pitch_is_bowling_friendly'] = 0
                df['pitch_is_balanced'] = 1  # Default to balanced
                logger.debug(f"Exiting add_venue_features after adding defaults. Df shape: {df.shape}")
                return df
                
            # Normalize venue name to improve matching chances
            venue_normalized = venue.strip().lower() if isinstance(venue, str) else ""
            logger.debug(f"Normalized venue: {venue_normalized}") # DEBUG
            
            # Find venue in venue data (with some flexibility)
            venue_match = None
            if 'venue' in self.venue_data.columns:
                # Try exact match first
                logger.debug("Attempting exact venue match...") # DEBUG
                venue_match = self.venue_data[self.venue_data['venue'].str.lower() == venue_normalized]
                
                # If no exact match, try partial match
                if venue_match.empty:
                    logger.debug("Exact match failed, attempting partial venue match...") # DEBUG
                    for v in self.venue_data['venue'].values:
                        if isinstance(v, str) and (venue_normalized in v.lower() or v.lower() in venue_normalized):
                            venue_match = self.venue_data[self.venue_data['venue'].str.lower() == v.lower()]
                            logger.info(f"Found partial venue match: '{venue}' matched with '{v}'")
                            break
            
            # If still no match, use default values
            if venue_match is None or venue_match.empty:
                logger.warning(f"Venue '{venue}' not found in venue data. Using default venue features.")
                df['venue_advantage'] = 1.0
                df['pitch_factor'] = 1.0
                df['bowling_factor'] = 1.0
                df['pitch_is_batting_friendly'] = 0
                df['pitch_is_bowling_friendly'] = 0
                df['pitch_is_balanced'] = 1  # Default to balanced
                logger.debug(f"Exiting add_venue_features after no match. Df shape: {df.shape}")
                return df
                
            # Extract venue statistics
            logger.debug(f"Found venue match. Extracting stats from {len(venue_match)} rows.") # DEBUG
            venue_stats = venue_match.iloc[0].to_dict()
            
            # Add venue-specific features to player data
            df['venue_advantage'] = 1.0  # Default value
            
            # Get batting and bowling rates from venue data
            batting_rate = venue_stats.get('batting_rate', 1.0)
            bowling_rate = venue_stats.get('bowling_rate', 1.0)
            logger.debug(f"Venue rates - Batting: {batting_rate}, Bowling: {bowling_rate}") # DEBUG
            
            # Set pitch type indicators based on venue stats
            if batting_rate > 1.1:  # Significantly above average
                df['pitch_is_batting_friendly'] = 1
                df['pitch_is_bowling_friendly'] = 0
                df['pitch_is_balanced'] = 0
            elif bowling_rate > 1.1:  # Significantly above average
                df['pitch_is_batting_friendly'] = 0
                df['pitch_is_bowling_friendly'] = 1
                df['pitch_is_balanced'] = 0
            else:
                df['pitch_is_batting_friendly'] = 0
                df['pitch_is_bowling_friendly'] = 0
                df['pitch_is_balanced'] = 1
                
            # Set pitch factors
            df['pitch_factor'] = batting_rate
            df['bowling_factor'] = bowling_rate
            
            # Calculate role-specific venue advantages
            logger.debug("Calculating role-specific venue advantages...") # DEBUG
            if 'role' in df.columns:
                role_map = {'BAT': 'batting_rate', 'BOWL': 'bowling_rate', 'AR': 'all_rounder_rate', 'WK': 'batting_rate'}
                
                for role, rate_column in role_map.items():
                    rate = venue_stats.get(rate_column, 1.0)
                    # Apply venue advantage to players of this role
                    df.loc[df['role'] == role, 'venue_advantage'] = rate
            else:
                 logger.warning("'role' column not found for role-specific venue advantage.")
                    
            logger.debug(f"Exiting add_venue_features successfully. Df shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Exception during venue feature addition: {str(e)}")
            traceback.print_exc()
            # Ensure default features are added even on error
            df['venue_advantage'] = 1.0
            df['pitch_factor'] = 1.0
            df['bowling_factor'] = 1.0
            df['pitch_is_batting_friendly'] = 0
            df['pitch_is_bowling_friendly'] = 0
            df['pitch_is_balanced'] = 1  # Default to balanced
            logger.warning(f"Exiting add_venue_features after exception. Df shape: {df.shape}")
            return df
            
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features that combine other features
        
        Args:
            df (pd.DataFrame): Player DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added interaction features
        """
        logger.debug(f"Entering add_interaction_features. Input df shape: {df.shape if isinstance(df, pd.DataFrame) else 'Invalid'}")
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Input df to add_interaction_features is invalid or empty.")
            return df
        
        try:
            # Confirm required columns exist
            required_cols = ['credits', 'role_value', 'recent_form', 'venue_advantage']
            logger.debug(f"Checking for required columns: {required_cols}") # DEBUG
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for interaction features: {missing_cols}. Adding placeholders.")
                for col in missing_cols:
                    if col == 'credits':
                        df['credits'] = 8.0  # Default credits
                    elif col == 'role_value':
                        df['role_value'] = 1.0  # Default role value
                    elif col == 'recent_form':
                        df['recent_form'] = 1.0  # Default form
                    elif col == 'venue_advantage':
                        df['venue_advantage'] = 1.0  # Default venue advantage
            
            # Ensure columns are numeric before calculations
            logger.debug("Ensuring interaction feature columns are numeric...") # DEBUG
            for col in required_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaNs that might result from coercion or already exist
                    default_val = 1.0 if col != 'credits' else 8.0 # Sensible defaults
                    df[col].fillna(default_val, inplace=True)
                else:
                    logger.error(f"Column '{col}' still missing after placeholder logic - this shouldn't happen!")
                    # Add the column with default if somehow missed
                    default_val = 1.0 if col != 'credits' else 8.0 
                    df[col] = default_val 

            # Create interaction features
            logger.debug("Calculating interaction features...") # DEBUG
            
            # 1. Form-adjusted role value
            df['form_role_value'] = df['role_value'] * df['recent_form']
            
            # 2. Credit-efficiency (form per credit)
            # Avoid division by zero if credits somehow became 0
            df['credit_efficiency'] = df['recent_form'] / df['credits'].replace(0, 1) 
            
            # 3. Venue-adjusted form
            df['venue_adjusted_form'] = df['recent_form'] * df['venue_advantage']
            
            # 4. Combined value score
            df['combined_value'] = (df['role_value'] + df['recent_form'] + df['venue_advantage']) / 3
            
            # 5. Weighted feature score (credits have less weight since they're a cost)
            df['weighted_score'] = (
                (0.25 * df['role_value']) + 
                (0.35 * df['recent_form']) + 
                (0.25 * df['venue_advantage']) - 
                (0.15 * (df['credits'] / 10))  # Normalize credits to 0-1 range approximately
            )
            
            logger.debug(f"Exiting add_interaction_features successfully. Df shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Exception during interaction feature addition: {str(e)}")
            traceback.print_exc()
            # Ensure some basic interaction features exist even on error
            df['form_role_value'] = 1.0
            df['credit_efficiency'] = 0.1
            df['venue_adjusted_form'] = 1.0
            df['combined_value'] = 1.0
            df['weighted_score'] = 0.5
            logger.warning(f"Exiting add_interaction_features after exception. Df shape: {df.shape}")
            return df

    # --- New Partnership Analysis Implementation ---
    def analyze_batting_partnerships(self, deliveries_df):
        """
        Analyze batting partnerships to identify strong player combinations
        
        Args:
            deliveries_df (pd.DataFrame): Dataframe with ball-by-ball delivery data
            
        Returns:
            dict: Dictionary with partnership metrics by player pair
        """
        logger.info("Analyzing batting partnerships...")
        
        if deliveries_df is None or deliveries_df.empty:
            logger.warning("No delivery data available for partnership analysis")
            return {}
            
        try:
            # Ensure required columns exist
            required_cols = ['batter', 'non_striker', 'runs_off_bat', 'match_id']
            if not all(col in deliveries_df.columns for col in required_cols):
                logger.warning(f"Missing required columns for partnership analysis. Required: {required_cols}")
                return {}
                
            # Create partnership identifier (sorted player names to ensure consistent pairing)
            deliveries_df['partnership'] = deliveries_df.apply(
                lambda x: tuple(sorted([str(x['batter']), str(x['non_striker'])])), 
                axis=1
            )
            
            # Calculate runs per ball for each partnership
            partnership_runs = deliveries_df.groupby(['match_id', 'partnership'])['runs_off_bat'].sum().reset_index()
            
            # Calculate balls faced by each partnership
            partnership_balls = deliveries_df.groupby(['match_id', 'partnership']).size().reset_index(name='balls')
            
            # Merge runs and balls data
            partnership_stats = pd.merge(partnership_runs, partnership_balls, on=['match_id', 'partnership'])
            
            # Calculate partnership metrics
            partnership_metrics = partnership_stats.groupby('partnership').agg({
                'runs_off_bat': ['sum', 'mean'],
                'balls': ['sum', 'count']  # count gives number of innings
            }).reset_index()
            
            # Flatten multi-level columns
            partnership_metrics.columns = [
                '_'.join(col).strip('_') if col[1] else col[0] for col in partnership_metrics.columns
            ]
            
            # Calculate strike rate and other metrics
            partnership_metrics['strike_rate'] = (
                partnership_metrics['runs_off_bat_sum'] / partnership_metrics['balls_sum'] * 100
            ).fillna(0)
            
            partnership_metrics['avg_runs_per_inning'] = (
                partnership_metrics['runs_off_bat_sum'] / partnership_metrics['balls_count']
            ).fillna(0)
            
            # Identify strong partnerships (above average strike rate and runs)
            avg_sr = partnership_metrics['strike_rate'].mean()
            avg_runs = partnership_metrics['avg_runs_per_inning'].mean()
            
            partnership_metrics['is_strong_partnership'] = (
                (partnership_metrics['strike_rate'] > avg_sr) & 
                (partnership_metrics['avg_runs_per_inning'] > avg_runs) &
                (partnership_metrics['balls_count'] >= 3)  # At least 3 innings together
            )
            
            # Convert to dictionary for easier lookup
            partnerships_dict = {}
            for _, row in partnership_metrics.iterrows():
                player1, player2 = row['partnership']
                metrics = row.drop('partnership').to_dict()
                
                # Store in both directions for easier lookup
                if player1 not in partnerships_dict:
                    partnerships_dict[player1] = {}
                partnerships_dict[player1][player2] = metrics
                
                if player2 not in partnerships_dict:
                    partnerships_dict[player2] = {}
                partnerships_dict[player2][player1] = metrics
                
            logger.info(f"Completed partnership analysis for {len(partnership_metrics)} partnerships")
            return partnerships_dict
            
        except Exception as e:
            logger.error(f"Error in partnership analysis: {str(e)}", exc_info=True)
            return {}
    
    # --- New Detailed Player vs Team Analysis ---
    def analyze_player_vs_team(self, deliveries_df):
        """
        Analyze player performance against specific teams
        
        Args:
            deliveries_df (pd.DataFrame): Dataframe with ball-by-ball delivery data
            
        Returns:
            dict: Dictionary with player vs team metrics
        """
        logger.info("Analyzing player vs team performance...")
        
        if deliveries_df is None or deliveries_df.empty:
            logger.warning("No delivery data available for player vs team analysis")
            return {}
            
        try:
            # Ensure required columns exist
            required_cols = ['batter', 'bowler', 'runs_off_bat', 'batting_team', 'bowling_team', 'player_dismissed']
            if not all(col in deliveries_df.columns for col in required_cols):
                logger.warning(f"Missing required columns for player vs team analysis. Required: {required_cols}")
                return {}
                
            # === Batting vs Team Analysis ===
            # Calculate batting stats against each team
            batting_vs_team = deliveries_df.groupby(['batter', 'bowling_team']).agg({
                'runs_off_bat': ['sum', 'mean'],
                'match_id': 'nunique',
                'ball': 'count',
            }).reset_index()
            
            # Flatten multi-level columns
            batting_vs_team.columns = [
                '_'.join(col).strip('_') if col[1] else col[0] for col in batting_vs_team.columns
            ]
            
            # Calculate batting strike rate vs each team
            batting_vs_team['bat_strike_rate_vs_team'] = (
                batting_vs_team['runs_off_bat_sum'] / batting_vs_team['ball_count'] * 100
            ).fillna(0)
            
            # Calculate batting average vs each team (need dismissals data)
            # Group by batter, dismissal team
            dismissals = deliveries_df[deliveries_df['player_dismissed'].notna()]
            dismissals_by_team = dismissals.groupby(['player_dismissed', 'bowling_team']).size().reset_index(name='dismissals')
            
            # Merge with batting stats
            batting_vs_team = pd.merge(
                batting_vs_team, 
                dismissals_by_team.rename(columns={'player_dismissed': 'batter'}),
                on=['batter', 'bowling_team'], 
                how='left'
            )
            
            batting_vs_team['dismissals'] = batting_vs_team['dismissals'].fillna(0)
            batting_vs_team['bat_avg_vs_team'] = np.where(
                batting_vs_team['dismissals'] > 0,
                batting_vs_team['runs_off_bat_sum'] / batting_vs_team['dismissals'],
                batting_vs_team['runs_off_bat_sum'] * 2  # If not dismissed, double runs as proxy for avg
            )
            
            # === Bowling vs Team Analysis ===
            # For bowlers, get wickets against each team
            wickets = deliveries_df[
                deliveries_df['player_dismissed'].notna() & 
                ~deliveries_df['wicket_type'].isin(['run out', 'retired hurt', 'obstructing the field'])
            ]
            
            bowling_vs_team = wickets.groupby(['bowler', 'batting_team']).size().reset_index(name='wickets_vs_team')
            
            # Get runs conceded against each team
            bowling_runs = deliveries_df.groupby(['bowler', 'batting_team'])['runs_off_bat'].sum().reset_index(name='runs_conceded_vs_team')
            
            # Get balls bowled against each team
            bowling_balls = deliveries_df.groupby(['bowler', 'batting_team']).size().reset_index(name='balls_bowled_vs_team')
            
            # Merge bowling stats
            bowling_vs_team = pd.merge(bowling_vs_team, bowling_runs, on=['bowler', 'batting_team'], how='outer')
            bowling_vs_team = pd.merge(bowling_vs_team, bowling_balls, on=['bowler', 'batting_team'], how='outer')
            
            # Calculate bowling average and economy vs each team
            bowling_vs_team['bowl_avg_vs_team'] = np.where(
                bowling_vs_team['wickets_vs_team'] > 0,
                bowling_vs_team['runs_conceded_vs_team'] / bowling_vs_team['wickets_vs_team'],
                bowling_vs_team['runs_conceded_vs_team'] * 2  # If no wickets, double runs as proxy
            )
            
            bowling_vs_team['bowl_economy_vs_team'] = (
                bowling_vs_team['runs_conceded_vs_team'] / (bowling_vs_team['balls_bowled_vs_team'] / 6)
            ).fillna(0)
            
            # Convert to dictionary for easier lookup
            player_vs_team = {}
            
            # Process batting stats
            for _, row in batting_vs_team.iterrows():
                batter = row['batter']
                team = row['bowling_team']
                
                if batter not in player_vs_team:
                    player_vs_team[batter] = {'batting': {}, 'bowling': {}}
                    
                player_vs_team[batter]['batting'][team] = {
                    'runs': row['runs_off_bat_sum'],
                    'balls': row['ball_count'],
                    'strike_rate': row['bat_strike_rate_vs_team'],
                    'average': row['bat_avg_vs_team'],
                    'matches': row['match_id_nunique'],
                    'dismissals': row['dismissals']
                }
            
            # Process bowling stats
            for _, row in bowling_vs_team.iterrows():
                bowler = row['bowler']
                team = row['batting_team']
                
                if bowler not in player_vs_team:
                    player_vs_team[bowler] = {'batting': {}, 'bowling': {}}
                    
                player_vs_team[bowler]['bowling'][team] = {
                    'wickets': row['wickets_vs_team'],
                    'runs_conceded': row['runs_conceded_vs_team'],
                    'balls': row['balls_bowled_vs_team'],
                    'average': row['bowl_avg_vs_team'],
                    'economy': row['bowl_economy_vs_team']
                }
                
            logger.info(f"Completed player vs team analysis for {len(player_vs_team)} players")
            return player_vs_team
            
        except Exception as e:
            logger.error(f"Error in player vs team analysis: {str(e)}", exc_info=True)
            return {}
            
    def add_player_specific_features(self, df, partnership_data=None, player_vs_team_data=None):
        """
        Add player-specific features to the main dataframe including partnership
        and vs-team metrics
        
        Args:
            df (pd.DataFrame): Player dataframe
            partnership_data (dict): Partnership metrics by player
            player_vs_team_data (dict): Player vs team metrics
            
        Returns:
            pd.DataFrame: Enhanced dataframe with added features
        """
        logger.info("Adding player-specific features...")
        
        if df is None or df.empty:
            logger.warning("No player data available for feature enhancement")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            enhanced_df = df.copy()
            
            # Get player names - try different possible column names
            player_col = None
            for col in ['player_name', 'Player Name', 'Player', 'name', 'player']:
                if col in enhanced_df.columns:
                    player_col = col
                    break
                    
            if player_col is None:
                logger.warning("Could not find player name column. Cannot add player-specific features.")
                return df
                
            # Initialize new feature columns
            enhanced_df['has_strong_partnership'] = 0
            enhanced_df['partnership_strike_rate'] = 0
            enhanced_df['partnership_avg_runs'] = 0
            enhanced_df['vs_opposition_bat_avg'] = 0
            enhanced_df['vs_opposition_bowl_avg'] = 999
            enhanced_df['vs_opposition_bat_sr'] = 0
            enhanced_df['favorable_matchup'] = 0
            
            # Add partnership features if data available
            if partnership_data:
                logger.info("Adding partnership features...")
                for idx, row in enhanced_df.iterrows():
                    player = row[player_col]
                    
                    # Get partnerships for this player
                    player_partnerships = partnership_data.get(player, {})
                    
                    if player_partnerships:
                        # Calculate average partnership metrics
                        partnership_sr = np.mean([
                            p.get('strike_rate', 0) for p in player_partnerships.values()
                        ])
                        
                        partnership_runs = np.mean([
                            p.get('avg_runs_per_inning', 0) for p in player_partnerships.values()
                        ])
                        
                        # Check if player has any strong partnerships
                        has_strong = any(
                            p.get('is_strong_partnership', False) for p in player_partnerships.values()
                        )
                        
                        # Update dataframe
                        enhanced_df.at[idx, 'partnership_strike_rate'] = partnership_sr
                        enhanced_df.at[idx, 'partnership_avg_runs'] = partnership_runs
                        enhanced_df.at[idx, 'has_strong_partnership'] = 1 if has_strong else 0
            
            # Add player vs team features if data available
            if player_vs_team_data and 'opposition' in enhanced_df.columns:
                logger.info("Adding player vs opposition team features...")
                
                for idx, row in enhanced_df.iterrows():
                    player = row[player_col]
                    opposition = row['opposition']
                    
                    # Get player vs team data
                    player_data = player_vs_team_data.get(player, {})
                    
                    # Add batting vs team metrics
                    if 'batting' in player_data and opposition in player_data['batting']:
                        bat_stats = player_data['batting'][opposition]
                        enhanced_df.at[idx, 'vs_opposition_bat_avg'] = bat_stats.get('average', 0)
                        enhanced_df.at[idx, 'vs_opposition_bat_sr'] = bat_stats.get('strike_rate', 0)
                        
                    # Add bowling vs team metrics
                    if 'bowling' in player_data and opposition in player_data['bowling']:
                        bowl_stats = player_data['bowling'][opposition]
                        enhanced_df.at[idx, 'vs_opposition_bowl_avg'] = bowl_stats.get('average', 999)
                        
                    # Determine if this is a favorable matchup
                    # For batsmen: higher avg and SR vs this team is good
                    # For bowlers: lower bowling avg vs this team is good
                    player_role = row.get('role', row.get('Player Type', 'BAT'))
                    
                    if 'BAT' in player_role or 'WK' in player_role:
                        # For batsmen, compare their average vs this team to their overall average
                        overall_avg = row.get('batting_avg', 0)
                        vs_team_avg = enhanced_df.at[idx, 'vs_opposition_bat_avg']
                        enhanced_df.at[idx, 'favorable_matchup'] = 1 if vs_team_avg > overall_avg else 0
                        
                    elif 'BOWL' in player_role:
                        # For bowlers, compare their average vs this team to their overall average
                        overall_avg = row.get('bowling_avg', 999)
                        vs_team_avg = enhanced_df.at[idx, 'vs_opposition_bowl_avg']
                        enhanced_df.at[idx, 'favorable_matchup'] = 1 if vs_team_avg < overall_avg else 0
                        
                    else:  # All-rounders
                        # Consider both batting and bowling
                        bat_favorable = enhanced_df.at[idx, 'vs_opposition_bat_avg'] > row.get('batting_avg', 0)
                        bowl_favorable = enhanced_df.at[idx, 'vs_opposition_bowl_avg'] < row.get('bowling_avg', 999)
                        enhanced_df.at[idx, 'favorable_matchup'] = 1 if (bat_favorable or bowl_favorable) else 0
            
            logger.info("Completed adding player-specific features")
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error adding player-specific features: {str(e)}", exc_info=True)
            return df