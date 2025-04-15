import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

# Import enhanced feature modules
from enhanced_features.partnership_analyzer import PartnershipAnalyzer
from enhanced_features.form_analyzer import FormAnalyzer
from enhanced_features.matchup_analyzer import MatchupAnalyzer
from enhanced_features.stadium_strategies import StadiumStrategies
from enhanced_features.pitch_type_optimizer import PitchTypeOptimizer

# Configure logging
logger = logging.getLogger('enhanced_features')

class EnhancedFeatures:
    """
    Integration of all enhanced features for Dream11 Fantasy Cricket Prediction System.
    
    This class combines all enhanced feature modules and provides a unified interface
    for the main application to use. It coordinates the application of various
    enhancements to player predictions and team optimization.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the EnhancedFeatures
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        
        # Initialize all enhanced feature modules
        self.partnership_analyzer = PartnershipAnalyzer(data_dir)
        self.form_analyzer = FormAnalyzer(data_dir)
        self.matchup_analyzer = MatchupAnalyzer(data_dir)
        self.stadium_strategies = StadiumStrategies(data_dir)
        self.pitch_type_optimizer = PitchTypeOptimizer(data_dir)
        
        # Track which features are loaded/initialized
        self.features_loaded = {
            'partnership': False,
            'form': False,
            'matchup': False,
            'stadium': True,  # Stadium strategies are pre-initialized
            'pitch': True     # Pitch optimizer is pre-initialized
        }
        
        logger.info("Enhanced features initialized")
    
    def load_data(self):
        """
        Load data for all enhanced feature modules
        
        Returns:
            bool: True if all data loaded successfully, False otherwise
        """
        try:
            # Load partnership data
            partnership_data = self.partnership_analyzer.load_partnership_data()
            self.features_loaded['partnership'] = partnership_data is not None
            
            # Load player history for form analysis
            player_history = self.form_analyzer.load_player_history()
            self.features_loaded['form'] = len(player_history) > 0
            
            # Load matchup data
            matchup_data = self.matchup_analyzer.load_matchup_data()
            self.features_loaded['matchup'] = matchup_data is not None
            
            # Load stadium data
            stadium_profiles = self.stadium_strategies.load_stadium_data()
            self.features_loaded['stadium'] = len(stadium_profiles) > 0
            
            # Load player pitch performance data
            pitch_performance = self.pitch_type_optimizer.load_player_pitch_performance()
            self.features_loaded['pitch'] = len(pitch_performance) > 0
            
            # Log loaded features
            loaded_features = [f for f, loaded in self.features_loaded.items() if loaded]
            logger.info(f"Loaded enhanced features: {', '.join(loaded_features)}")
            
            return all(self.features_loaded.values())
        
        except Exception as e:
            logger.error(f"Error loading enhanced feature data: {str(e)}")
            return False
    
    def enhance_player_predictions(self, player_predictions, match_context):
        """
        Apply all enhanced features to player predictions
        
        Args:
            player_predictions (dict): Dictionary of player predictions
            match_context (dict): Match context information including:
                - team1: First team name
                - team2: Second team name
                - venue: Stadium name
                - pitch_type: Type of pitch
                - squad_data: DataFrame with squad information
            
        Returns:
            dict: Enhanced player predictions
        """
        # Create a copy of predictions to enhance
        enhanced_predictions = player_predictions.copy()
        
        try:
            # Extract match context
            team1 = match_context.get('team1', '')
            team2 = match_context.get('team2', '')
            venue = match_context.get('venue', '')
            pitch_type = match_context.get('pitch_type', 'balanced')
            squad_data = match_context.get('squad_data', pd.DataFrame())
            
            # Apply partnership analysis if loaded
            if self.features_loaded['partnership']:
                # Create partnership graph if not already created
                if self.partnership_analyzer.partnership_graph is None:
                    self.partnership_analyzer.create_partnership_graph()
                
                # Adjust predictions based on partnerships
                enhanced_predictions = self.partnership_analyzer.adjust_player_predictions(
                    enhanced_predictions, squad_data)
                logger.info("Applied partnership analysis to player predictions")
            
            # Apply form analysis if loaded
            if self.features_loaded['form']:
                # Calculate form metrics if not already calculated
                if not self.form_analyzer.form_metrics:
                    self.form_analyzer.calculate_form_metrics()
                
                # Adjust predictions based on form
                enhanced_predictions = self.form_analyzer.adjust_player_predictions(enhanced_predictions)
                logger.info("Applied form analysis to player predictions")
            
            # Apply matchup analysis if loaded
            if self.features_loaded['matchup'] and team1 and team2:
                # Adjust predictions based on matchups
                enhanced_predictions = self.matchup_analyzer.adjust_player_predictions(
                    enhanced_predictions, team1, team2)
                logger.info("Applied matchup analysis to player predictions")
            
            # Apply stadium strategies if loaded
            if self.features_loaded['stadium'] and venue:
                # Adjust predictions based on stadium
                enhanced_predictions = self.stadium_strategies.adjust_player_predictions(
                    enhanced_predictions, venue)
                logger.info(f"Applied stadium strategies for {venue} to player predictions")
            
            # Apply pitch type optimization if loaded
            if self.features_loaded['pitch'] and pitch_type:
                # Adjust predictions based on pitch type
                enhanced_predictions = self.pitch_type_optimizer.adjust_player_predictions(
                    enhanced_predictions, pitch_type)
                logger.info(f"Applied pitch type optimization for {pitch_type} to player predictions")
            
            return enhanced_predictions
        
        except Exception as e:
            logger.error(f"Error enhancing player predictions: {str(e)}")
            return player_predictions  # Return original predictions on error
    
    def get_team_balance_requirements(self, pitch_type):
        """
        Get team balance requirements based on pitch type
        
        Args:
            pitch_type (str): Type of pitch
            
        Returns:
            dict: Team balance requirements with role counts
        """
        return self.pitch_type_optimizer.get_team_balance_requirements(pitch_type)
    
    def recommend_partnerships(self, squad_data, top_n=5):
        """
        Recommend strong partnerships to include in team selection
        
        Args:
            squad_data (pd.DataFrame): Squad data with player information
            top_n (int): Number of top partnerships to recommend
            
        Returns:
            list: List of recommended partnerships
        """
        if self.features_loaded['partnership']:
            return self.partnership_analyzer.recommend_partnerships(squad_data, top_n)
        return []
    
    def identify_in_form_players(self, squad_data, top_n=5):
        """
        Identify in-form players from the squad
        
        Args:
            squad_data (pd.DataFrame): Squad data with player information
            top_n (int): Number of top in-form players to identify
            
        Returns:
            list: List of in-form players with metrics
        """
        if self.features_loaded['form']:
            return self.form_analyzer.identify_in_form_players(squad_data, top_n)
        return []
    
    def identify_favorable_matchups(self, squad_data, opposition_team, top_n=5):
        """
        Identify favorable matchups against opposition team
        
        Args:
            squad_data (pd.DataFrame): Squad data with player information
            opposition_team (str): Opposition team name
            top_n (int): Number of top favorable matchups to identify
            
        Returns:
            list: List of favorable matchups with metrics
        """
        if self.features_loaded['matchup']:
            return self.matchup_analyzer.identify_favorable_matchups(squad_data, opposition_team, top_n)
        return []
    
    def recommend_team_composition(self, venue, pitch_type):
        """
        Recommend team composition based on venue and pitch type
        
        Args:
            venue (str): Stadium name
            pitch_type (str): Type of pitch
            
        Returns:
            dict: Recommended team composition with role counts
        """
        # Get stadium-based recommendation
        stadium_recommendation = self.stadium_strategies.recommend_team_composition(venue)
        
        # Get pitch-based recommendation
        pitch_recommendation = {
            'composition': self.pitch_type_optimizer.get_team_balance_requirements(pitch_type),
            'description': f"Recommended team composition for {pitch_type} pitch"
        }
        
        # Combine recommendations (giving more weight to pitch type)
        combined_composition = {}
        for role in ['WK', 'BAT', 'AR', 'BOWL']:
            stadium_optimal = stadium_recommendation['composition'][role]['optimal']
            pitch_optimal = pitch_recommendation['composition'][role]['optimal']
            
            # Weight pitch type more heavily (60% pitch, 40% stadium)
            combined_optimal = round(0.6 * pitch_optimal + 0.4 * stadium_optimal)
            
            combined_composition[role] = {
                'min': min(stadium_recommendation['composition'][role]['min'],
                          pitch_recommendation['composition'][role]['min']),
                'max': max(stadium_recommendation['composition'][role]['max'],
                          pitch_recommendation['composition'][role]['max']),
                'optimal': combined_optimal
            }
        
        return {
            'composition': combined_composition,
            'description': f"Recommended team composition for {venue} ({pitch_type} pitch)",
            'stadium_profile': stadium_recommendation['stadium_profile'],
            'pitch_report': self.pitch_type_optimizer.get_pitch_report(pitch_type)
        }
    
    def get_enhanced_feature_reports(self, player_name=None, venue=None, pitch_type=None):
        """
        Get detailed reports from all enhanced feature modules
        
        Args:
            player_name (str, optional): Name of player for player-specific reports
            venue (str, optional): Stadium name for venue-specific reports
            pitch_type (str, optional): Type of pitch for pitch-specific reports
            
        Returns:
            dict: Dictionary of reports from all enhanced feature modules
        """
        reports = {}
        
        # Get player-specific reports if player name provided
        if player_name:
            if self.features_loaded['form']:
                reports['form_report'] = self.form_analyzer.get_player_form_report(player_name)
        
        # Get venue-specific report if venue provided
        if venue:
            reports['stadium_report'] = self.stadium_strategies.get_stadium_report(venue)
        
        # Get pitch-specific report if pitch type provided
        if pitch_type:
            reports['pitch_report'] = self.pitch_type_optimizer.get_pitch_report(pitch_type)
        
        return reports