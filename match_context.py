import pandas as pd
import numpy as np
import logging
import traceback
from typing import Dict, List, Union, Optional, Tuple

# Configure logging
logger = logging.getLogger('match_context')

class MatchContext:
    """
    Match context analysis for Dream11 predictions.
    Analyzes different match types (league, qualifier, eliminator, final)
    and their impact on player performance.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the MatchContext module
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.match_data = None
        self.match_type_multipliers = {
            'League': 1.0,       # Base case
            'Qualifier': 1.15,   # 15% boost for qualifier matches
            'Eliminator': 1.2,   # 20% boost for eliminator matches
            'Final': 1.25        # 25% boost for final match
        }
        self.player_pressure_profiles = {}
        self._load_match_data()
    
    def _load_match_data(self):
        """Load match data if available"""
        try:
            import os
            # Try to load match data
            match_file = os.path.join(self.data_dir, "match_results.csv")
            if os.path.exists(match_file):
                self.match_data = pd.read_csv(match_file)
                logger.info(f"Loaded match data: {len(self.match_data)} matches")
                
                # Calculate pressure performance for players
                self._calculate_pressure_performance()
            else:
                logger.warning(f"Match data file not found at {match_file}")
                self.match_data = None
        except Exception as e:
            logger.error(f"Error loading match data: {e}")
            traceback.print_exc()
            self.match_data = None

    def _calculate_pressure_performance(self):
        """Calculate pressure performance metrics for players"""
        try:
            import os
            # Check if we have player performance data
            player_performance_file = os.path.join(self.data_dir, "player_performance.csv")
            if not os.path.exists(player_performance_file):
                logger.warning("Player performance data not found for pressure calculations")
                return
                
            # Load player performance data
            player_performance = pd.read_csv(player_performance_file)
            
            # Ensure we have match_id in both datasets for joining
            if 'match_id' not in player_performance.columns or 'match_id' not in self.match_data.columns:
                logger.warning("match_id column missing from datasets, cannot join for pressure performance")
                return
                
            # Create pressure match indicator (close/important matches)
            self.match_data['is_pressure_match'] = False
            
            # Identify close matches (win margin < 15 runs or < 3 wickets)
            if 'win_margin' in self.match_data.columns and 'win_type' in self.match_data.columns:
                # Close run margin matches
                run_margin_mask = (self.match_data['win_type'] == 'runs') & (self.match_data['win_margin'] < 15)
                
                # Close wicket margin matches
                wicket_margin_mask = (self.match_data['win_type'] == 'wickets') & (self.match_data['win_margin'] < 3)
                
                # Super over matches
                super_over_mask = self.match_data['win_type'] == 'super_over'
                
                # Combine all pressure match indicators
                self.match_data.loc[run_margin_mask | wicket_margin_mask | super_over_mask, 'is_pressure_match'] = True
                
            # Identify playoff matches
            if 'match_type' in self.match_data.columns:
                playoff_mask = self.match_data['match_type'].isin(['Qualifier', 'Eliminator', 'Final'])
                self.match_data.loc[playoff_mask, 'is_pressure_match'] = True
            
            # Join match data with player performance
            merged_data = player_performance.merge(
                self.match_data[['match_id', 'is_pressure_match', 'match_type']], 
                on='match_id', 
                how='left'
            )
            
            # Group by player and calculate pressure metrics
            pressure_profiles = {}
            
            for player_name in merged_data['player_name'].unique():
                player_data = merged_data[merged_data['player_name'] == player_name]
                
                # Skip if too few matches
                if len(player_data) < 3:
                    continue
                
                # Calculate metrics for pressure and non-pressure situations
                pressure_matches = player_data[player_data['is_pressure_match']]
                normal_matches = player_data[~player_data['is_pressure_match']]
                
                # Skip if no pressure matches
                if len(pressure_matches) == 0:
                    continue
                
                # Calculate batting pressure performance
                batting_pressure_ratio = 1.0
                if 'runs' in player_data.columns:
                    avg_pressure_runs = pressure_matches['runs'].mean() if len(pressure_matches) > 0 else 0
                    avg_normal_runs = normal_matches['runs'].mean() if len(normal_matches) > 0 else 0
                    
                    if avg_normal_runs > 0:
                        batting_pressure_ratio = avg_pressure_runs / avg_normal_runs
                        
                # Calculate bowling pressure performance
                bowling_pressure_ratio = 1.0
                if 'wickets' in player_data.columns:
                    avg_pressure_wickets = pressure_matches['wickets'].mean() if len(pressure_matches) > 0 else 0
                    avg_normal_wickets = normal_matches['wickets'].mean() if len(normal_matches) > 0 else 0
                    
                    if avg_normal_wickets > 0:
                        bowling_pressure_ratio = avg_pressure_wickets / avg_normal_wickets
                
                # Calculate playoff performance
                playoff_performance = 1.0
                if 'match_type' in player_data.columns:
                    playoff_matches = player_data[player_data['match_type'].isin(['Qualifier', 'Eliminator', 'Final'])]
                    
                    if len(playoff_matches) > 0:
                        if 'fantasy_points' in playoff_matches.columns:
                            avg_playoff_points = playoff_matches['fantasy_points'].mean()
                            avg_normal_points = normal_matches['fantasy_points'].mean() if len(normal_matches) > 0 else 0
                            
                            if avg_normal_points > 0:
                                playoff_performance = avg_playoff_points / avg_normal_points
                
                # Store player pressure profile
                pressure_profiles[player_name] = {
                    'batting_pressure_ratio': batting_pressure_ratio,
                    'bowling_pressure_ratio': bowling_pressure_ratio,
                    'playoff_performance': playoff_performance,
                    'pressure_matches_played': len(pressure_matches),
                    'playoff_matches_played': len(player_data[player_data['match_type'].isin(['Qualifier', 'Eliminator', 'Final'])])
                }
            
            self.player_pressure_profiles = pressure_profiles
            logger.info(f"Calculated pressure profiles for {len(pressure_profiles)} players")
            
        except Exception as e:
            logger.error(f"Error calculating pressure performance: {e}")
            traceback.print_exc()
            self.player_pressure_profiles = {}

    def apply_match_context(self, player_data: pd.DataFrame, match_type: str) -> pd.DataFrame:
        """
        Apply match context adjustments to player data
        
        Args:
            player_data (pd.DataFrame): Player data
            match_type (str): Match type (League, Qualifier, Eliminator, Final)
            
        Returns:
            pd.DataFrame: DataFrame with match context adjustments
        """
        if not isinstance(player_data, pd.DataFrame) or player_data.empty:
            logger.error("Invalid or empty DataFrame provided to apply_match_context")
            return player_data
            
        try:
            # Make a copy to avoid modifying the original
            df = player_data.copy()
            
            # Add match type info
            df['match_type'] = match_type
            
            # Get match type multiplier
            match_multiplier = self.match_type_multipliers.get(match_type, 1.0)
            df['match_type_multiplier'] = match_multiplier
            
            # Apply player-specific pressure adjustments if available
            if self.player_pressure_profiles:
                # Get player identifier column
                player_id_col = None
                for col_name in ['player_name', 'name', 'Player Name', 'player']:
                    if col_name in df.columns:
                        player_id_col = col_name
                        break
                        
                if player_id_col is not None:
                    # Add pressure performance columns
                    df['batting_pressure_ratio'] = 1.0
                    df['bowling_pressure_ratio'] = 1.0
                    df['playoff_performance'] = 1.0
                    
                    # Apply for each player
                    for idx, player in df.iterrows():
                        player_name = player[player_id_col]
                        
                        if player_name in self.player_pressure_profiles:
                            profile = self.player_pressure_profiles[player_name]
                            
                            # Add pressure metrics
                            df.loc[idx, 'batting_pressure_ratio'] = profile['batting_pressure_ratio']
                            df.loc[idx, 'bowling_pressure_ratio'] = profile['bowling_pressure_ratio']
                            df.loc[idx, 'playoff_performance'] = profile['playoff_performance']
                            
                            # Add experience metrics
                            df.loc[idx, 'pressure_matches_played'] = profile['pressure_matches_played']
                            df.loc[idx, 'playoff_matches_played'] = profile['playoff_matches_played']
                            
                            # Calculate a compound pressure factor
                            if match_type in ['Qualifier', 'Eliminator', 'Final']:
                                # For playoff matches, use the playoff performance metric
                                df.loc[idx, 'pressure_adjustment'] = profile['playoff_performance']
                            else:
                                # For regular matches, use 1.0 (no adjustment)
                                df.loc[idx, 'pressure_adjustment'] = 1.0
                
            # If no player pressure profiles, add default values
            if 'pressure_adjustment' not in df.columns:
                df['pressure_adjustment'] = 1.0
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying match context: {e}")
            traceback.print_exc()
            # Return original data with match type added
            player_data['match_type'] = match_type
            player_data['match_type_multiplier'] = self.match_type_multipliers.get(match_type, 1.0)
            return player_data

    def get_match_importance(self, match_type: str) -> Dict:
        """
        Get match importance metrics for UI display
        
        Args:
            match_type (str): Match type
            
        Returns:
            Dict: Match importance metrics
        """
        try:
            # Base importance metrics
            importance = {
                'League': {
                    'importance_level': 'normal',
                    'pressure_level': 'normal',
                    'description': 'Regular league match',
                    'key_factor': 'Consistent performers excel'
                },
                'Qualifier': {
                    'importance_level': 'high',
                    'pressure_level': 'high',
                    'description': 'Top teams compete for direct final spot',
                    'key_factor': 'Experienced players and big-match performers'
                },
                'Eliminator': {
                    'importance_level': 'high',
                    'pressure_level': 'very high',
                    'description': 'Knockout match with no second chances',
                    'key_factor': 'Players who perform well under pressure'
                },
                'Final': {
                    'importance_level': 'very high',
                    'pressure_level': 'extreme',
                    'description': 'Championship deciding match',
                    'key_factor': 'Elite performers and seasoned champions'
                }
            }
            
            # Return importance metrics for the specified match type
            if match_type in importance:
                return importance[match_type]
            else:
                # Return default values for unknown match types
                return {
                    'importance_level': 'normal',
                    'pressure_level': 'normal',
                    'description': 'Regular match',
                    'key_factor': 'Consistent performers'
                }
                
        except Exception as e:
            logger.error(f"Error getting match importance: {e}")
            traceback.print_exc()
            return {
                'importance_level': 'normal',
                'pressure_level': 'normal',
                'description': 'Standard match',
                'key_factor': 'Overall performance'
            } 