import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple
import traceback

# Configure logging
logger = logging.getLogger('player_momentum')

class PlayerMomentum:
    """
    Advanced player momentum calculation for Dream11 predictions.
    Implements rolling averages and time-weighted performance metrics.
    """
    
    def __init__(self, data_dir="dataset", window_size=5, decay_factor=0.85):
        """
        Initialize the PlayerMomentum module
        
        Args:
            data_dir (str): Directory containing data files
            window_size (int): Number of matches to consider for rolling averages
            decay_factor (float): Exponential decay factor for weighting recent performances
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.historical_data = None
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical player performance data if available"""
        try:
            import os
            # Load historical player performance data
            historical_file = os.path.join(self.data_dir, "player_history.csv")
            if os.path.exists(historical_file):
                self.historical_data = pd.read_csv(historical_file)
                # Ensure we have a date column for time-series analysis
                if 'match_date' in self.historical_data.columns:
                    # Convert to datetime if not already
                    self.historical_data['match_date'] = pd.to_datetime(
                        self.historical_data['match_date'], errors='coerce'
                    )
                    # Sort by date for proper rolling calculations
                    self.historical_data.sort_values('match_date', inplace=True)
                    
                logger.info(f"Loaded historical data: {len(self.historical_data)} records")
            else:
                logger.warning(f"Historical data file not found at {historical_file}")
                self.historical_data = None
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            traceback.print_exc()
            self.historical_data = None

    def calculate_player_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum scores for players based on recent performances
        
        Args:
            df (pd.DataFrame): Player data with basic stats
            
        Returns:
            pd.DataFrame: DataFrame with added momentum features
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Invalid or empty DataFrame provided to calculate_player_momentum")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            player_data = df.copy()
            
            # Check if historical data is available
            if self.historical_data is None or self.historical_data.empty:
                logger.warning("Historical data not available for momentum calculation")
                # Add default momentum features
                player_data['momentum_score'] = 1.0
                player_data['batting_momentum'] = 1.0
                player_data['bowling_momentum'] = 1.0
                player_data['recent_form_indicator'] = 0.5
                return player_data
            
            # Get player identifiers - try multiple common column names
            player_id_col = None
            for col_name in ['player_name', 'name', 'Player Name', 'player']:
                if col_name in player_data.columns:
                    player_id_col = col_name
                    break
                    
            if player_id_col is None:
                logger.error("Could not identify player name column in input data")
                player_data['momentum_score'] = 1.0
                return player_data
            
            # Calculate rolling averages for each player
            batting_momentum = {}
            bowling_momentum = {}
            overall_momentum = {}
            
            for player_name in player_data[player_id_col].unique():
                # Get player's historical data
                player_history = self.historical_data[
                    self.historical_data['player_name'].str.lower() == player_name.lower()
                ].copy()
                
                if len(player_history) > 0:
                    # Calculate rolling averages
                    batting_metrics = self.calculate_rolling_averages(
                        player_history, 
                        metrics=['runs', 'fours', 'sixes', 'strike_rate'],
                        prefix='batting'
                    )
                    
                    bowling_metrics = self.calculate_rolling_averages(
                        player_history,
                        metrics=['wickets', 'economy', 'bowling_avg'],
                        prefix='bowling'
                    )
                    
                    # Apply time decay to give more weight to recent performances
                    batting_score = self.apply_exponential_decay(batting_metrics, 'batting_runs')
                    bowling_score = self.apply_exponential_decay(bowling_metrics, 'bowling_wickets')
                    
                    # Store momentum scores
                    batting_momentum[player_name] = batting_score
                    bowling_momentum[player_name] = bowling_score
                    overall_momentum[player_name] = (batting_score + bowling_score) / 2
                else:
                    # No historical data available for this player
                    batting_momentum[player_name] = 1.0
                    bowling_momentum[player_name] = 1.0
                    overall_momentum[player_name] = 1.0
            
            # Add momentum scores to player data
            player_data['batting_momentum'] = player_data[player_id_col].map(batting_momentum).fillna(1.0)
            player_data['bowling_momentum'] = player_data[player_id_col].map(bowling_momentum).fillna(1.0)
            player_data['momentum_score'] = player_data[player_id_col].map(overall_momentum).fillna(1.0)
            
            # Calculate recent form indicator (0-1 scale)
            player_data['recent_form_indicator'] = player_data['momentum_score'].apply(
                lambda x: min(max(x/2, 0), 1)  # Normalize to 0-1 range
            )
            
            return player_data
            
        except Exception as e:
            logger.error(f"Error calculating player momentum: {e}")
            traceback.print_exc()
            # Ensure default values are added
            df['momentum_score'] = 1.0
            df['batting_momentum'] = 1.0
            df['bowling_momentum'] = 1.0
            df['recent_form_indicator'] = 0.5
            return df

    def calculate_rolling_averages(
        self, 
        player_history: pd.DataFrame, 
        metrics: List[str],
        prefix: str = ''
    ) -> pd.DataFrame:
        """
        Calculate rolling averages for specified metrics
        
        Args:
            player_history (pd.DataFrame): Player's historical performance data
            metrics (List[str]): List of metrics to calculate averages for
            prefix (str): Prefix to add to output column names
            
        Returns:
            pd.DataFrame: DataFrame with added rolling average columns
        """
        try:
            # Create a copy to avoid modifying original
            history = player_history.copy()
            
            # Check if we have enough data for rolling calculations
            if len(history) < 2:
                # Not enough data for rolling calculations
                return history
                
            # Process each metric
            for metric in metrics:
                if metric in history.columns:
                    # Calculate rolling average
                    col_name = f"{prefix}_{metric}_rolling" if prefix else f"{metric}_rolling"
                    history[col_name] = history[metric].rolling(
                        window=min(self.window_size, len(history)), 
                        min_periods=1
                    ).mean()
                    
                    # Calculate rolling std for volatility
                    std_col = f"{prefix}_{metric}_std" if prefix else f"{metric}_std"
                    history[std_col] = history[metric].rolling(
                        window=min(self.window_size, len(history)), 
                        min_periods=1
                    ).std().fillna(0)
                    
                    # Calculate trend (positive/negative momentum)
                    trend_col = f"{prefix}_{metric}_trend" if prefix else f"{metric}_trend"
                    history[trend_col] = history[metric].diff().rolling(
                        window=min(3, len(history)), 
                        min_periods=1
                    ).mean()
            
            return history
            
        except Exception as e:
            logger.error(f"Error calculating rolling averages: {e}")
            traceback.print_exc()
            return player_history

    def apply_exponential_decay(self, df: pd.DataFrame, primary_metric: str) -> float:
        """
        Apply exponential decay to give more weight to recent performances
        
        Args:
            df (pd.DataFrame): Player history with rolling metrics
            primary_metric (str): The main metric to apply decay to
            
        Returns:
            float: Weighted momentum score
        """
        try:
            # Create weights that decrease exponentially going backward in time
            if len(df) <= 1:
                return 1.0
                
            # Get the latest value of the rolling metric
            if primary_metric in df.columns:
                latest_value = df[primary_metric].iloc[-1]
                
                # If we have at least 3 matches, calculate weighted average
                if len(df) >= 3:
                    # Create decay weights
                    weights = np.array([self.decay_factor ** i for i in range(min(len(df), self.window_size))])
                    # Reverse so most recent match has highest weight
                    weights = weights[::-1]
                    # Normalize weights to sum to 1
                    weights = weights / weights.sum()
                    
                    # Get values for last n matches
                    values = df[primary_metric].iloc[-min(len(df), self.window_size):].values
                    
                    # Calculate weighted average
                    weighted_avg = np.sum(values * weights)
                    
                    # Calculate momentum score (comparing to unweighted average)
                    momentum = weighted_avg / max(values.mean(), 0.01)
                    
                    return momentum
                else:
                    # Not enough matches for weighted calculation
                    return latest_value / max(df[primary_metric].mean(), 0.01)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error applying exponential decay: {e}")
            traceback.print_exc()
            return 1.0

    def get_consistency_score(self, df: pd.DataFrame, metric: str) -> float:
        """
        Calculate consistency score based on coefficient of variation
        
        Args:
            df (pd.DataFrame): Player history
            metric (str): The metric to evaluate consistency for
            
        Returns:
            float: Consistency score (0-1 scale, 1 being most consistent)
        """
        try:
            if len(df) < 3 or metric not in df.columns:
                return 0.5  # Default for insufficient data
                
            # Get recent values
            recent_values = df[metric].iloc[-min(len(df), self.window_size):].values
            
            # Calculate coefficient of variation (lower is more consistent)
            mean = recent_values.mean()
            if mean == 0:
                return 0.5  # Avoid division by zero
                
            std = recent_values.std()
            cv = std / max(abs(mean), 0.01)
            
            # Convert to 0-1 scale (0: inconsistent, 1: consistent)
            # CV is typically between 0-1 for most sports data, but can go higher
            consistency = max(0, min(1, 1 - (cv / 2)))
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            traceback.print_exc()
            return 0.5 