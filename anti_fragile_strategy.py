import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logger = logging.getLogger('anti_fragile_strategy')

class AntiFragileStrategy:
    """
    Implements anti-fragile team selection strategy to ensure team composition
    can perform well across different match conditions.
    
    This strategy focuses on creating teams that are robust to uncertainty and
    can adapt to changing match conditions, rather than being optimized for a
    specific scenario.
    """
    
    def __init__(self):
        """
        Initialize the AntiFragileStrategy
        """
        self.consistency_threshold = 0.7  # Coefficient of variation threshold
        self.form_weight = 0.6  # Weight given to recent form
        self.variance_tolerance = 0.2  # Acceptable variance in player performance
        
    def calculate_player_consistency(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate consistency scores for players based on their match history
        
        Args:
            player_history: DataFrame containing player performance history
            
        Returns:
            Dictionary mapping player names to consistency scores
        """
        consistency_scores = {}
        
        if player_history.empty:
            logger.warning("Empty player history provided for consistency calculation")
            return consistency_scores
            
        # Group by player and calculate consistency metrics
        for player, data in player_history.groupby('player_name'):
            if len(data) < 3:  # Need at least 3 matches for meaningful consistency
                continue
                
            # Calculate fantasy points statistics
            points = data['fantasy_points'].values
            mean_points = np.mean(points)
            std_points = np.std(points)
            
            # Calculate coefficient of variation (lower is more consistent)
            cv = std_points / mean_points if mean_points > 0 else float('inf')
            
            # Calculate lower bound (more conservative estimate)
            lower_bound = mean_points - 1.96 * std_points / np.sqrt(len(points))
            lower_bound = max(0, lower_bound)
            
            # Calculate consistency score (inverse of CV, normalized)
            consistency_score = 1 / (1 + cv)
            
            consistency_scores[player] = {
                'mean': mean_points,
                'std': std_points,
                'cv': cv,
                'lower_bound': lower_bound,
                'consistency_score': consistency_score
            }
            
        return consistency_scores
    
    def apply_anti_fragile_strategy(self, player_predictions: Dict[str, Dict], 
                                   player_history: Optional[pd.DataFrame] = None,
                                   variance_factor: float = 0.2) -> Dict[str, Dict]:
        """
        Apply anti-fragile strategy to player predictions
        
        Args:
            player_predictions: Dictionary of player predictions
            player_history: DataFrame containing player performance history
            variance_factor: Factor to control variance adjustment
            
        Returns:
            Dictionary of adjusted player predictions
        """
        adjusted_predictions = player_predictions.copy()
        
        # If no history data available, apply simple variance-based adjustment
        if player_history is None or player_history.empty:
            logger.info("No player history available, applying simple variance adjustment")
            for player_id, player_data in adjusted_predictions.items():
                # Apply conservative adjustment based on credits
                credits = player_data.get('credits', 8.0)
                points = player_data.get('points', 0)
                
                # Higher credit players have higher variance
                player_variance = points * variance_factor * (credits / 10.0)
                
                # Adjust points with lower bound estimate
                adjusted_points = points - (player_variance * 0.5)
                adjusted_predictions[player_id]['anti_fragile_points'] = max(0, adjusted_points)
                
            return adjusted_predictions
        
        # Calculate consistency scores if history is available
        consistency_scores = self.calculate_player_consistency(player_history)
        
        # Apply anti-fragile adjustments based on consistency
        for player_id, player_data in adjusted_predictions.items():
            player_name = player_data.get('name', '')
            points = player_data.get('points', 0)
            
            if player_name in consistency_scores:
                # Use consistency metrics for adjustment
                cs = consistency_scores[player_name]
                
                # Calculate anti-fragile score using lower bound and consistency
                anti_fragile_points = (cs['lower_bound'] * 0.7) + (points * 0.3)
                
                # Apply consistency bonus/penalty
                if cs['consistency_score'] > self.consistency_threshold:
                    # Bonus for consistent players
                    anti_fragile_points *= 1.1
                else:
                    # Penalty for inconsistent players
                    anti_fragile_points *= 0.9
            else:
                # No history data for this player, use conservative estimate
                anti_fragile_points = points * (1 - variance_factor)
                
            adjusted_predictions[player_id]['anti_fragile_points'] = max(0, anti_fragile_points)
            
        return adjusted_predictions
    
    def get_robust_team_composition(self, pitch_type: str) -> Dict[str, Dict[str, int]]:
        """
        Get robust team composition requirements for different pitch types
        
        Args:
            pitch_type: Type of pitch (balanced, batting_friendly, bowling_friendly)
            
        Returns:
            Dictionary with role requirements for robust team composition
        """
        # Define anti-fragile team compositions for different pitch types
        robust_compositions = {
            'balanced': {
                'WK': {'min': 1, 'max': 2, 'optimal': 1},
                'BAT': {'min': 3, 'max': 5, 'optimal': 4},
                'AR': {'min': 2, 'max': 4, 'optimal': 3},
                'BOWL': {'min': 3, 'max': 5, 'optimal': 3}
            },
            'batting_friendly': {
                'WK': {'min': 1, 'max': 2, 'optimal': 2},
                'BAT': {'min': 4, 'max': 5, 'optimal': 4},
                'AR': {'min': 2, 'max': 3, 'optimal': 2},
                'BOWL': {'min': 2, 'max': 4, 'optimal': 3}
            },
            'bowling_friendly': {
                'WK': {'min': 1, 'max': 1, 'optimal': 1},
                'BAT': {'min': 3, 'max': 4, 'optimal': 3},
                'AR': {'min': 2, 'max': 3, 'optimal': 3},
                'BOWL': {'min': 4, 'max': 5, 'optimal': 4}
            }
        }
        
        # Return the appropriate composition or default to balanced
        return robust_compositions.get(pitch_type, robust_compositions['balanced'])
    
    def identify_core_players(self, simulation_results: Dict) -> List[str]:
        """
        Identify core players that appear in most simulation iterations
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            
        Returns:
            List of core player names
        """
        core_players = []
        
        if 'player_frequencies' not in simulation_results:
            logger.warning("No player frequencies in simulation results")
            return core_players
            
        player_freq = simulation_results['player_frequencies']
        num_iterations = len(simulation_results.get('teams', []))
        
        if num_iterations == 0:
            return core_players
            
        # Players appearing in at least 70% of simulations are considered core
        threshold = 0.7 * num_iterations
        
        for player, count in player_freq.items():
            if count >= threshold:
                core_players.append(player)
                
        return core_players