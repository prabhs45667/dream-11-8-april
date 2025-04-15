import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logger = logging.getLogger('pitch_type_optimizer')

class PitchTypeOptimizer:
    """
    Adjusts player values based on pitch conditions.
    
    This module identifies pitch type (batting/bowling/balanced/spin/pace-friendly),
    adjusts player values based on their skills and the pitch type, and optimizes
    team composition for specific pitch conditions.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the PitchTypeOptimizer
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.pitch_profiles = {}
        self.player_pitch_performance = {}
        
        # Define pitch types and their characteristics
        self.pitch_types = {
            'batting_friendly': {
                'description': 'High-scoring pitch favoring batsmen',
                'avg_first_innings_score': 180,
                'role_weights': {'WK': 1.2, 'BAT': 1.3, 'AR': 1.1, 'BOWL': 0.8},
                'skill_weights': {
                    'batting_skills': 1.3,
                    'bowling_skills': 0.8,
                    'power_hitting': 1.4,
                    'spin_bowling': 0.9,
                    'pace_bowling': 0.7
                }
            },
            'bowling_friendly': {
                'description': 'Low-scoring pitch favoring bowlers',
                'avg_first_innings_score': 150,
                'role_weights': {'WK': 0.9, 'BAT': 0.8, 'AR': 1.1, 'BOWL': 1.3},
                'skill_weights': {
                    'batting_skills': 0.8,
                    'bowling_skills': 1.3,
                    'power_hitting': 0.7,
                    'spin_bowling': 1.2,
                    'pace_bowling': 1.2
                }
            },
            'balanced': {
                'description': 'Balanced pitch offering equal assistance to batsmen and bowlers',
                'avg_first_innings_score': 165,
                'role_weights': {'WK': 1.0, 'BAT': 1.0, 'AR': 1.1, 'BOWL': 1.0},
                'skill_weights': {
                    'batting_skills': 1.0,
                    'bowling_skills': 1.0,
                    'power_hitting': 1.0,
                    'spin_bowling': 1.0,
                    'pace_bowling': 1.0
                }
            },
            'spin_friendly': {
                'description': 'Pitch offering assistance to spin bowlers',
                'avg_first_innings_score': 160,
                'role_weights': {'WK': 0.9, 'BAT': 0.9, 'AR': 1.2, 'BOWL': 1.2},
                'skill_weights': {
                    'batting_skills': 0.9,
                    'bowling_skills': 1.2,
                    'power_hitting': 0.8,
                    'spin_bowling': 1.4,
                    'pace_bowling': 0.9
                }
            },
            'pace_friendly': {
                'description': 'Pitch offering assistance to pace bowlers',
                'avg_first_innings_score': 155,
                'role_weights': {'WK': 0.9, 'BAT': 0.9, 'AR': 1.1, 'BOWL': 1.2},
                'skill_weights': {
                    'batting_skills': 0.9,
                    'bowling_skills': 1.2,
                    'power_hitting': 0.8,
                    'spin_bowling': 0.9,
                    'pace_bowling': 1.4
                }
            }
        }
        
        # Define team balance requirements for different pitch types
        self.team_balance_requirements = {
            'balanced': {
                'WK': {'min': 1, 'max': 2, 'optimal': 1},
                'BAT': {'min': 3, 'max': 5, 'optimal': 4},
                'AR': {'min': 1, 'max': 4, 'optimal': 3},
                'BOWL': {'min': 3, 'max': 5, 'optimal': 3}
            },
            'batting_friendly': {
                'WK': {'min': 1, 'max': 3, 'optimal': 2},
                'BAT': {'min': 4, 'max': 6, 'optimal': 5},
                'AR': {'min': 1, 'max': 3, 'optimal': 2},
                'BOWL': {'min': 2, 'max': 4, 'optimal': 2}
            },
            'bowling_friendly': {
                'WK': {'min': 1, 'max': 2, 'optimal': 1},
                'BAT': {'min': 3, 'max': 4, 'optimal': 3},
                'AR': {'min': 1, 'max': 3, 'optimal': 2},
                'BOWL': {'min': 4, 'max': 6, 'optimal': 5}
            },
            'spin_friendly': {
                'WK': {'min': 1, 'max': 2, 'optimal': 1},
                'BAT': {'min': 3, 'max': 5, 'optimal': 4},
                'AR': {'min': 2, 'max': 4, 'optimal': 3},
                'BOWL': {'min': 3, 'max': 5, 'optimal': 3}
            },
            'pace_friendly': {
                'WK': {'min': 1, 'max': 2, 'optimal': 1},
                'BAT': {'min': 3, 'max': 5, 'optimal': 4},
                'AR': {'min': 1, 'max': 3, 'optimal': 2},
                'BOWL': {'min': 4, 'max': 6, 'optimal': 4}
            }
        }
    
    def load_player_pitch_performance(self, file_path=None):
        """
        Load player performance data on different pitch types
        
        Args:
            file_path (str, optional): Path to player pitch performance data file
            
        Returns:
            dict: Dictionary of player performance on different pitch types
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "player_history.csv")
            
        try:
            if os.path.exists(file_path):
                # Load player history data
                history_df = pd.read_csv(file_path)
                logger.info(f"Loaded player history data from {file_path}: {history_df.shape[0]} records")
                
                # Check if pitch type information is available
                pitch_type_col = next((col for col in ['pitch_type', 'pitch_condition', 'pitch']
                                     if col in history_df.columns), None)
                
                if pitch_type_col:
                    # Group by player and pitch type
                    player_col = next((col for col in ['player_name', 'name', 'Player']
                                     if col in history_df.columns), None)
                    
                    if player_col:
                        # Calculate average fantasy points by pitch type
                        points_col = next((col for col in ['fantasy_points', 'points', 'Points']
                                         if col in history_df.columns), None)
                        
                        if points_col:
                            for (player, pitch_type), group in history_df.groupby([player_col, pitch_type_col]):
                                if player not in self.player_pitch_performance:
                                    self.player_pitch_performance[player] = {}
                                
                                # Calculate average points on this pitch type
                                avg_points = group[points_col].mean()
                                matches = len(group)
                                
                                self.player_pitch_performance[player][pitch_type] = {
                                    'avg_points': avg_points,
                                    'matches': matches
                                }
                            
                            logger.info(f"Processed pitch performance for {len(self.player_pitch_performance)} players")
                        else:
                            logger.warning("Could not find fantasy points column in history data")
                    else:
                        logger.warning("Could not find player name column in history data")
                else:
                    logger.warning("No pitch type information found in history data")
                    
                    # Create synthetic pitch performance data based on player roles
                    if 'role' in history_df.columns:
                        for player, group in history_df.groupby(player_col):
                            role = group['role'].iloc[0] if not group['role'].empty else 'BAT'
                            
                            # Create synthetic performance data based on role
                            self.player_pitch_performance[player] = self._create_synthetic_pitch_performance(role)
                        
                        logger.info(f"Created synthetic pitch performance for {len(self.player_pitch_performance)} players")
            else:
                logger.warning(f"Player history file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading player pitch performance: {str(e)}")
        
        return self.player_pitch_performance
    
    def _create_synthetic_pitch_performance(self, role):
        """
        Create synthetic pitch performance data based on player role
        
        Args:
            role (str): Player role (WK, BAT, AR, BOWL)
            
        Returns:
            dict: Synthetic pitch performance data
        """
        # Base performance value
        base_value = 25.0
        
        # Adjust based on role and pitch type
        performance = {}
        
        if role in ['BAT', 'WK']:
            # Batsmen perform better on batting-friendly pitches
            performance['batting_friendly'] = {'avg_points': base_value * 1.3, 'matches': 5}
            performance['bowling_friendly'] = {'avg_points': base_value * 0.8, 'matches': 5}
            performance['balanced'] = {'avg_points': base_value * 1.0, 'matches': 5}
            performance['spin_friendly'] = {'avg_points': base_value * 0.9, 'matches': 5}
            performance['pace_friendly'] = {'avg_points': base_value * 0.9, 'matches': 5}
        elif role == 'BOWL':
            # Bowlers perform better on bowling-friendly pitches
            performance['batting_friendly'] = {'avg_points': base_value * 0.8, 'matches': 5}
            performance['bowling_friendly'] = {'avg_points': base_value * 1.3, 'matches': 5}
            performance['balanced'] = {'avg_points': base_value * 1.0, 'matches': 5}
            performance['spin_friendly'] = {'avg_points': base_value * 1.2, 'matches': 5}
            performance['pace_friendly'] = {'avg_points': base_value * 1.2, 'matches': 5}
        elif role == 'AR':
            # All-rounders are more balanced
            performance['batting_friendly'] = {'avg_points': base_value * 1.1, 'matches': 5}
            performance['bowling_friendly'] = {'avg_points': base_value * 1.1, 'matches': 5}
            performance['balanced'] = {'avg_points': base_value * 1.2, 'matches': 5}
            performance['spin_friendly'] = {'avg_points': base_value * 1.1, 'matches': 5}
            performance['pace_friendly'] = {'avg_points': base_value * 1.1, 'matches': 5}
        
        return performance
    
    def get_pitch_profile(self, pitch_type):
        """
        Get pitch profile for a specific pitch type
        
        Args:
            pitch_type (str): Type of pitch (batting_friendly, bowling_friendly, etc.)
            
        Returns:
            dict: Pitch profile with characteristics and weights
        """
        # Standardize pitch type
        pitch_type = self._standardize_pitch_type(pitch_type)
        
        # Return pitch profile
        return self.pitch_types.get(pitch_type, self.pitch_types['balanced'])
    
    def _standardize_pitch_type(self, pitch_type):
        """
        Standardize pitch type to one of the defined types
        
        Args:
            pitch_type (str): Input pitch type
            
        Returns:
            str: Standardized pitch type
        """
        # Convert to lowercase for comparison
        pitch_type_lower = pitch_type.lower()
        
        # Map to standard pitch types
        if 'bat' in pitch_type_lower or 'high' in pitch_type_lower or 'flat' in pitch_type_lower:
            return 'batting_friendly'
        elif 'bowl' in pitch_type_lower or 'low' in pitch_type_lower or 'green' in pitch_type_lower:
            return 'bowling_friendly'
        elif 'spin' in pitch_type_lower or 'turn' in pitch_type_lower or 'dry' in pitch_type_lower:
            return 'spin_friendly'
        elif 'pace' in pitch_type_lower or 'fast' in pitch_type_lower or 'bouncy' in pitch_type_lower:
            return 'pace_friendly'
        elif 'balance' in pitch_type_lower or 'even' in pitch_type_lower or 'neutral' in pitch_type_lower:
            return 'balanced'
        else:
            # Default to balanced if unknown
            logger.warning(f"Unknown pitch type: {pitch_type}. Defaulting to 'balanced'.")
            return 'balanced'
    
    def adjust_player_predictions(self, player_predictions, pitch_type):
        """
        Adjust player predictions based on pitch type
        
        Args:
            player_predictions (dict): Dictionary of player predictions
            pitch_type (str): Type of pitch
            
        Returns:
            dict: Adjusted player predictions
        """
        # Standardize pitch type
        pitch_type = self._standardize_pitch_type(pitch_type)
        
        # Get pitch profile
        pitch_profile = self.get_pitch_profile(pitch_type)
        role_weights = pitch_profile.get('role_weights', {})
        skill_weights = pitch_profile.get('skill_weights', {})
        
        # Create a copy of predictions to adjust
        adjusted_predictions = player_predictions.copy()
        
        try:
            for player_name, prediction in player_predictions.items():
                # Get player role
                role = prediction.get('role', 'BAT')
                
                # Get role weight for this pitch type
                role_weight = role_weights.get(role, 1.0)
                
                # Get player's historical performance on this pitch type
                pitch_performance = 1.0
                if player_name in self.player_pitch_performance and pitch_type in self.player_pitch_performance[player_name]:
                    # Calculate performance factor based on historical data
                    player_pitch_data = self.player_pitch_performance[player_name][pitch_type]
                    avg_points = player_pitch_data.get('avg_points', 0)
                    matches = player_pitch_data.get('matches', 0)
                    
                    # Only consider if player has played enough matches on this pitch type
                    if matches >= 3 and avg_points > 0:
                        # Get player's overall average
                        overall_avg = prediction.get('average_points', 0) or 25.0
                        
                        # Calculate performance factor (ratio of pitch-specific to overall average)
                        if overall_avg > 0:
                            pitch_performance = avg_points / overall_avg
                            # Limit the range of adjustment
                            pitch_performance = min(1.5, max(0.7, pitch_performance))
                
                # Calculate skill-based adjustment
                skill_adjustment = 1.0
                if 'skills' in prediction:
                    skills = prediction['skills']
                    skill_adjustment = self._calculate_skill_adjustment(skills, skill_weights)
                
                # Calculate combined adjustment factor
                adjustment_factor = role_weight * pitch_performance * skill_adjustment
                
                # Apply adjustment to prediction
                if 'predicted_points' in prediction:
                    adjusted_predictions[player_name]['predicted_points'] = \
                        prediction['predicted_points'] * adjustment_factor
                    
                    # Add pitch adjustment factor to prediction
                    adjusted_predictions[player_name]['pitch_factor'] = adjustment_factor
                    
                    # Log significant adjustments
                    if abs(adjustment_factor - 1.0) > 0.1:
                        logger.info(f"Adjusted {player_name}'s prediction by factor {adjustment_factor:.2f} "
                                  f"based on {pitch_type} pitch conditions")
        
        except Exception as e:
            logger.error(f"Error adjusting player predictions: {str(e)}")
        
        return adjusted_predictions
    
    def _calculate_skill_adjustment(self, player_skills, skill_weights):
        """
        Calculate skill-based adjustment factor
        
        Args:
            player_skills (dict): Player skill ratings
            skill_weights (dict): Skill weights for the pitch type
            
        Returns:
            float: Skill adjustment factor
        """
        # Default skills if not provided
        if not player_skills or not skill_weights:
            return 1.0
        
        # Calculate weighted skill score
        skill_score = 0.0
        weight_sum = 0.0
        
        for skill, rating in player_skills.items():
            if skill in skill_weights:
                weight = skill_weights[skill]
                skill_score += rating * weight
                weight_sum += weight
        
        # Calculate adjustment factor
        if weight_sum > 0:
            # Normalize to 0.8-1.2 range
            adjustment = (skill_score / weight_sum) / 5.0  # Assuming skills are rated 1-10
            return min(1.2, max(0.8, adjustment))
        else:
            return 1.0
    
    def get_team_balance_requirements(self, pitch_type):
        """
        Get team balance requirements for a specific pitch type
        
        Args:
            pitch_type (str): Type of pitch
            
        Returns:
            dict: Team balance requirements with role counts
        """
        # Standardize pitch type
        pitch_type = self._standardize_pitch_type(pitch_type)
        
        # Return balance requirements
        return self.team_balance_requirements.get(pitch_type, self.team_balance_requirements['balanced'])
    
    def get_pitch_report(self, pitch_type):
        """
        Get detailed report for a specific pitch type
        
        Args:
            pitch_type (str): Type of pitch
            
        Returns:
            dict: Detailed pitch report
        """
        # Standardize pitch type
        pitch_type = self._standardize_pitch_type(pitch_type)
        
        # Get pitch profile
        profile = self.get_pitch_profile(pitch_type)
        
        # Create report
        report = {
            'pitch_type': pitch_type,
            'description': profile.get('description', ''),
            'avg_first_innings_score': profile.get('avg_first_innings_score', 0),
            'role_weights': profile.get('role_weights', {}),
            'skill_weights': profile.get('skill_weights', {}),
            'team_balance': self.get_team_balance_requirements(pitch_type)
        }
        
        # Add strategy recommendations
        strategies = []
        if pitch_type == 'batting_friendly':
            strategies.append("Prioritize aggressive batsmen with high strike rates")
            strategies.append("Include power-hitters who can clear boundaries easily")
            strategies.append("Select bowlers with good variations and slower balls")
        elif pitch_type == 'bowling_friendly':
            strategies.append("Prioritize technically sound batsmen with good defensive skills")
            strategies.append("Include bowlers with good seam movement and swing")
            strategies.append("Select all-rounders who can contribute with both bat and ball")
        elif pitch_type == 'spin_friendly':
            strategies.append("Prioritize spin bowlers and batsmen who play spin well")
            strategies.append("Include all-rounders who can bowl spin")
            strategies.append("Select wicketkeepers with good stumping abilities")
        elif pitch_type == 'pace_friendly':
            strategies.append("Prioritize pace bowlers and batsmen who play pace well")
            strategies.append("Include all-rounders who can bowl pace")
            strategies.append("Select batsmen with good technique against short balls")
        else:  # balanced
            strategies.append("Select a well-rounded team with balance across all departments")
            strategies.append("Prioritize consistent performers who adapt well to different conditions")
            strategies.append("Include versatile all-rounders who can contribute in multiple ways")
        
        report['recommended_strategies'] = strategies
        
        return report