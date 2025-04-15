import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logger = logging.getLogger('stadium_strategies')

class StadiumStrategies:
    """
    Optimizes team selection based on venue characteristics.
    
    This module maintains profiles for different stadiums, adjusts role importance
    based on venue conditions, and recommends team composition based on stadium type.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the StadiumStrategies
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.stadium_profiles = {}
        self.default_role_weights = {
            'WK': 1.0,
            'BAT': 1.0,
            'AR': 1.0,
            'BOWL': 1.0
        }
        
        # Initialize with some known stadium profiles
        self._initialize_stadium_profiles()
    
    def _initialize_stadium_profiles(self):
        """
        Initialize stadium profiles with known characteristics
        """
        self.stadium_profiles = {
            "M. Chinnaswamy Stadium, Bangalore": {
                "high_scoring": True,
                "spin_friendly": False,
                "pace_friendly": True,
                "average_first_innings_score": 180,
                "role_weights": {'WK': 1.1, 'BAT': 1.3, 'AR': 1.2, 'BOWL': 0.9}
            },
            "Eden Gardens, Kolkata": {
                "high_scoring": False,
                "spin_friendly": True,
                "pace_friendly": False,
                "average_first_innings_score": 165,
                "role_weights": {'WK': 1.0, 'BAT': 1.0, 'AR': 1.1, 'BOWL': 1.3}
            },
            "Wankhede Stadium, Mumbai": {
                "high_scoring": True,
                "spin_friendly": False,
                "pace_friendly": True,
                "average_first_innings_score": 175,
                "role_weights": {'WK': 1.1, 'BAT': 1.2, 'AR': 1.1, 'BOWL': 1.0}
            },
            "MA Chidambaram Stadium, Chennai": {
                "high_scoring": False,
                "spin_friendly": True,
                "pace_friendly": False,
                "average_first_innings_score": 160,
                "role_weights": {'WK': 1.0, 'BAT': 0.9, 'AR': 1.2, 'BOWL': 1.3}
            },
            "Narendra Modi Stadium, Ahmedabad": {
                "high_scoring": True,
                "spin_friendly": False,
                "pace_friendly": True,
                "average_first_innings_score": 170,
                "role_weights": {'WK': 1.0, 'BAT': 1.2, 'AR': 1.1, 'BOWL': 1.0}
            },
            "Arun Jaitley Stadium, Delhi": {
                "high_scoring": True,
                "spin_friendly": True,
                "pace_friendly": False,
                "average_first_innings_score": 175,
                "role_weights": {'WK': 1.1, 'BAT': 1.2, 'AR': 1.1, 'BOWL': 1.0}
            }
        }
    
    def load_stadium_data(self, file_path=None):
        """
        Load stadium data from file or use default profiles
        
        Args:
            file_path (str, optional): Path to stadium data file
            
        Returns:
            dict: Dictionary of stadium profiles
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "venue_stats.csv")
            
        try:
            if os.path.exists(file_path):
                # Load stadium data
                stadium_df = pd.read_csv(file_path)
                logger.info(f"Loaded stadium data from {file_path}: {stadium_df.shape[0]} records")
                
                # Process stadium data
                for _, row in stadium_df.iterrows():
                    stadium_name = row.get('venue', '') or row.get('stadium', '')
                    if not stadium_name:
                        continue
                    
                    # Extract stadium characteristics
                    avg_score = row.get('average_first_innings_score', 0) or row.get('avg_score', 0)
                    high_scoring = avg_score > 170 if avg_score else row.get('high_scoring', False)
                    
                    # Determine pitch characteristics
                    spin_friendly = row.get('spin_friendly', False)
                    pace_friendly = row.get('pace_friendly', False)
                    
                    # Calculate role weights based on characteristics
                    role_weights = self._calculate_role_weights(high_scoring, spin_friendly, pace_friendly)
                    
                    # Create stadium profile
                    self.stadium_profiles[stadium_name] = {
                        "high_scoring": high_scoring,
                        "spin_friendly": spin_friendly,
                        "pace_friendly": pace_friendly,
                        "average_first_innings_score": avg_score,
                        "role_weights": role_weights
                    }
                
                logger.info(f"Processed {len(self.stadium_profiles)} stadium profiles")
            else:
                logger.warning(f"Stadium data file not found: {file_path}. Using default profiles.")
        except Exception as e:
            logger.error(f"Error loading stadium data: {str(e)}")
        
        return self.stadium_profiles
    
    def _calculate_role_weights(self, high_scoring, spin_friendly, pace_friendly):
        """
        Calculate role weights based on stadium characteristics
        
        Args:
            high_scoring (bool): Whether the stadium is high-scoring
            spin_friendly (bool): Whether the stadium is spin-friendly
            pace_friendly (bool): Whether the stadium is pace-friendly
            
        Returns:
            dict: Role weights for the stadium
        """
        role_weights = self.default_role_weights.copy()
        
        # Adjust weights based on characteristics
        if high_scoring:
            # High-scoring venues favor batsmen and aggressive all-rounders
            role_weights['BAT'] *= 1.2
            role_weights['WK'] *= 1.1
            role_weights['AR'] *= 1.1
            role_weights['BOWL'] *= 0.9
        else:
            # Low-scoring venues favor bowlers
            role_weights['BOWL'] *= 1.2
            role_weights['AR'] *= 1.1
            role_weights['BAT'] *= 0.9
            role_weights['WK'] *= 1.0
        
        if spin_friendly:
            # Spin-friendly venues favor spin bowlers
            # This would ideally be more granular with bowler types
            role_weights['BOWL'] *= 1.1
        
        if pace_friendly:
            # Pace-friendly venues favor pace bowlers
            # This would ideally be more granular with bowler types
            role_weights['BOWL'] *= 1.1
        
        return role_weights
    
    def get_stadium_profile(self, stadium_name):
        """
        Get stadium profile for a specific venue
        
        Args:
            stadium_name (str): Name of the stadium
            
        Returns:
            dict: Stadium profile with characteristics and role weights
        """
        # Try exact match first
        if stadium_name in self.stadium_profiles:
            return self.stadium_profiles[stadium_name]
        
        # Try partial match
        for name, profile in self.stadium_profiles.items():
            if stadium_name.lower() in name.lower() or name.lower() in stadium_name.lower():
                return profile
        
        # Return default profile if no match found
        logger.warning(f"No profile found for stadium: {stadium_name}. Using default profile.")
        return {
            "high_scoring": True,
            "spin_friendly": False,
            "pace_friendly": False,
            "average_first_innings_score": 170,
            "role_weights": self.default_role_weights.copy()
        }
    
    def adjust_player_predictions(self, player_predictions, stadium_name):
        """
        Adjust player predictions based on stadium characteristics
        
        Args:
            player_predictions (dict): Dictionary of player predictions
            stadium_name (str): Name of the stadium
            
        Returns:
            dict: Adjusted player predictions
        """
        # Get stadium profile
        stadium_profile = self.get_stadium_profile(stadium_name)
        role_weights = stadium_profile.get('role_weights', self.default_role_weights)
        
        # Create a copy of predictions to adjust
        adjusted_predictions = player_predictions.copy()
        
        try:
            for player_name, prediction in player_predictions.items():
                # Get player role
                role = prediction.get('role', 'BAT')
                
                # Get role weight for this stadium
                role_weight = role_weights.get(role, 1.0)
                
                # Apply adjustment to prediction
                if 'predicted_points' in prediction:
                    adjusted_predictions[player_name]['predicted_points'] = \
                        prediction['predicted_points'] * role_weight
                    
                    # Add stadium adjustment factor to prediction
                    adjusted_predictions[player_name]['stadium_factor'] = role_weight
                    
                    # Log significant adjustments
                    if abs(role_weight - 1.0) > 0.1:
                        logger.info(f"Adjusted {player_name}'s prediction by factor {role_weight:.2f} "
                                  f"based on {stadium_name} characteristics")
        
        except Exception as e:
            logger.error(f"Error adjusting player predictions: {str(e)}")
        
        return adjusted_predictions
    
    def recommend_team_composition(self, stadium_name):
        """
        Recommend team composition based on stadium characteristics
        
        Args:
            stadium_name (str): Name of the stadium
            
        Returns:
            dict: Recommended team composition with role counts
        """
        # Get stadium profile
        stadium_profile = self.get_stadium_profile(stadium_name)
        
        # Determine team composition based on stadium characteristics
        high_scoring = stadium_profile.get('high_scoring', False)
        spin_friendly = stadium_profile.get('spin_friendly', False)
        pace_friendly = stadium_profile.get('pace_friendly', False)
        
        # Default balanced composition
        composition = {
            'WK': {'min': 1, 'max': 2, 'optimal': 1},
            'BAT': {'min': 3, 'max': 5, 'optimal': 4},
            'AR': {'min': 1, 'max': 4, 'optimal': 3},
            'BOWL': {'min': 3, 'max': 5, 'optimal': 3}
        }
        
        # Adjust composition based on stadium characteristics
        if high_scoring:
            # High-scoring venues favor batsmen
            composition['BAT']['optimal'] = 5
            composition['WK']['optimal'] = 1
            composition['AR']['optimal'] = 2
            composition['BOWL']['optimal'] = 3
        elif spin_friendly:
            # Spin-friendly venues favor spin bowlers
            composition['BAT']['optimal'] = 3
            composition['WK']['optimal'] = 1
            composition['AR']['optimal'] = 3
            composition['BOWL']['optimal'] = 4
        elif pace_friendly:
            # Pace-friendly venues favor pace bowlers
            composition['BAT']['optimal'] = 3
            composition['WK']['optimal'] = 1
            composition['AR']['optimal'] = 3
            composition['BOWL']['optimal'] = 4
        
        # Add description
        description = f"Recommended team composition for {stadium_name}: "
        description += f"{composition['WK']['optimal']} WK, {composition['BAT']['optimal']} BAT, "
        description += f"{composition['AR']['optimal']} AR, {composition['BOWL']['optimal']} BOWL"
        
        return {
            'composition': composition,
            'description': description,
            'stadium_profile': stadium_profile
        }
    
    def get_stadium_report(self, stadium_name):
        """
        Get detailed report for a specific stadium
        
        Args:
            stadium_name (str): Name of the stadium
            
        Returns:
            dict: Detailed stadium report
        """
        # Get stadium profile
        profile = self.get_stadium_profile(stadium_name)
        
        # Create report
        report = {
            'name': stadium_name,
            'high_scoring': profile.get('high_scoring', False),
            'spin_friendly': profile.get('spin_friendly', False),
            'pace_friendly': profile.get('pace_friendly', False),
            'average_first_innings_score': profile.get('average_first_innings_score', 0),
            'role_weights': profile.get('role_weights', self.default_role_weights),
            'recommended_composition': self.recommend_team_composition(stadium_name)['composition']
        }
        
        # Add stadium type description
        if report['high_scoring']:
            if report['spin_friendly']:
                report['stadium_type'] = "High-scoring, spin-friendly"
            elif report['pace_friendly']:
                report['stadium_type'] = "High-scoring, pace-friendly"
            else:
                report['stadium_type'] = "High-scoring, balanced"
        else:
            if report['spin_friendly']:
                report['stadium_type'] = "Low-scoring, spin-friendly"
            elif report['pace_friendly']:
                report['stadium_type'] = "Low-scoring, pace-friendly"
            else:
                report['stadium_type'] = "Low-scoring, balanced"
        
        # Add strategy recommendations
        strategies = []
        if report['high_scoring']:
            strategies.append("Prioritize aggressive batsmen with high strike rates")
            strategies.append("Include power-hitters who can clear boundaries easily")
        else:
            strategies.append("Prioritize technically sound batsmen with good defensive skills")
            strategies.append("Include anchors who can build innings on difficult pitches")
        
        if report['spin_friendly']:
            strategies.append("Include quality spin bowlers and all-rounders")
            strategies.append("Prioritize batsmen who play spin well")
        
        if report['pace_friendly']:
            strategies.append("Include quality pace bowlers and all-rounders")
            strategies.append("Prioritize batsmen who play pace well")
        
        report['recommended_strategies'] = strategies
        
        return report