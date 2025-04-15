import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger('form_analyzer')

class FormAnalyzer:
    """
    Analyzes player form and consistency to improve predictions.
    
    This module tracks player performance over time, calculates consistency scores,
    identifies form trends, and adjusts predictions based on recent form to identify
    in-form players for optimal team selection.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the FormAnalyzer
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.player_history = {}
        self.form_metrics = {}
        self.consistency_scores = {}
        self.recent_form_window = 5  # Last 5 matches by default
        self.form_decay_factor = 0.8  # Exponential decay factor for recent matches
        
    def load_player_history(self, file_path=None):
        """
        Load player match history data
        
        Args:
            file_path (str, optional): Path to player history data file
            
        Returns:
            dict: Dictionary of player history data by player name
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "player_history.csv")
            
        try:
            if os.path.exists(file_path):
                # Load player history data
                history_df = pd.read_csv(file_path)
                logger.info(f"Loaded player history data from {file_path}: {history_df.shape[0]} records")
                
                # Convert date column to datetime if it exists
                date_columns = ['date', 'match_date', 'Date']
                date_col = next((col for col in date_columns if col in history_df.columns), None)
                
                if date_col:
                    history_df[date_col] = pd.to_datetime(history_df[date_col], errors='coerce')
                    # Sort by date
                    history_df = history_df.sort_values(by=[date_col])
                
                # Group by player name
                player_col = next((col for col in ['player_name', 'name', 'Player'] 
                                 if col in history_df.columns), None)
                
                if player_col:
                    for player, group in history_df.groupby(player_col):
                        self.player_history[player] = group.reset_index(drop=True)
                    
                    logger.info(f"Processed history for {len(self.player_history)} players")
                else:
                    logger.warning("Could not find player name column in history data")
                
                return self.player_history
            else:
                logger.warning(f"Player history file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading player history: {str(e)}")
            return {}
    
    def calculate_form_metrics(self):
        """
        Calculate form metrics for all players
        
        Returns:
            dict: Dictionary of form metrics by player name
        """
        self.form_metrics = {}
        
        try:
            for player_name, history_df in self.player_history.items():
                # Skip if player has no history
                if history_df.empty:
                    continue
                
                # Identify fantasy points column
                points_cols = ['fantasy_points', 'points', 'Points', 'FantasyPoints']
                points_col = next((col for col in points_cols if col in history_df.columns), None)
                
                if points_col is None:
                    logger.warning(f"Could not find fantasy points column for player {player_name}")
                    continue
                
                # Get recent matches (last N matches)
                recent_matches = history_df.tail(self.recent_form_window)
                
                # Calculate metrics
                metrics = {
                    'recent_avg': recent_matches[points_col].mean(),
                    'overall_avg': history_df[points_col].mean(),
                    'max_points': history_df[points_col].max(),
                    'min_points': history_df[points_col].min(),
                    'std_dev': history_df[points_col].std(),
                    'matches_played': len(history_df),
                    'recent_matches': len(recent_matches),
                    'recent_form_trend': self._calculate_form_trend(recent_matches, points_col),
                    'consistency': self._calculate_consistency(history_df, points_col),
                    'weighted_recent_avg': self._calculate_weighted_average(recent_matches, points_col)
                }
                
                # Calculate coefficient of variation (lower is more consistent)
                if metrics['overall_avg'] > 0:
                    metrics['cv'] = metrics['std_dev'] / metrics['overall_avg']
                else:
                    metrics['cv'] = float('inf')
                
                # Calculate form factor (ratio of recent to overall average)
                if metrics['overall_avg'] > 0:
                    metrics['form_factor'] = metrics['recent_avg'] / metrics['overall_avg']
                else:
                    metrics['form_factor'] = 1.0
                
                # Store metrics
                self.form_metrics[player_name] = metrics
                
                # Calculate Bayesian consistency score
                self.consistency_scores[player_name] = self._calculate_bayesian_consistency(metrics)
            
            logger.info(f"Calculated form metrics for {len(self.form_metrics)} players")
            return self.form_metrics
        
        except Exception as e:
            logger.error(f"Error calculating form metrics: {str(e)}")
            return {}
    
    def _calculate_form_trend(self, recent_matches, points_col):
        """
        Calculate form trend from recent matches
        
        Args:
            recent_matches (pd.DataFrame): Recent match data
            points_col (str): Column name for fantasy points
            
        Returns:
            float: Form trend value (positive = improving, negative = declining)
        """
        if len(recent_matches) < 2:
            return 0.0
        
        # Get points in chronological order
        points = recent_matches[points_col].values
        
        # Calculate trend using linear regression slope
        x = np.arange(len(points))
        if len(x) > 0 and len(points) > 0:
            slope, _ = np.polyfit(x, points, 1)
            return slope
        return 0.0
    
    def _calculate_consistency(self, history_df, points_col):
        """
        Calculate player consistency score
        
        Args:
            history_df (pd.DataFrame): Player history data
            points_col (str): Column name for fantasy points
            
        Returns:
            float: Consistency score (0-1, higher is more consistent)
        """
        if len(history_df) < 3:
            return 0.5  # Default for insufficient data
        
        # Calculate coefficient of variation (lower is more consistent)
        mean = history_df[points_col].mean()
        std = history_df[points_col].std()
        
        if mean > 0:
            cv = std / mean
            # Convert to 0-1 scale (higher is more consistent)
            consistency = 1 / (1 + cv)
            return min(1.0, max(0.0, consistency))
        else:
            return 0.0
    
    def _calculate_weighted_average(self, recent_matches, points_col):
        """
        Calculate weighted average of recent performances with exponential decay
        
        Args:
            recent_matches (pd.DataFrame): Recent match data
            points_col (str): Column name for fantasy points
            
        Returns:
            float: Weighted average points
        """
        if recent_matches.empty:
            return 0.0
        
        # Get points in chronological order
        points = recent_matches[points_col].values
        
        # Calculate weights with exponential decay (more recent = higher weight)
        weights = np.array([self.form_decay_factor ** i for i in range(len(points)-1, -1, -1)])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate weighted average
        weighted_avg = np.sum(points * weights)
        
        return weighted_avg
    
    def _calculate_bayesian_consistency(self, metrics):
        """
        Calculate Bayesian consistency score with confidence intervals
        
        Args:
            metrics (dict): Player form metrics
            
        Returns:
            dict: Bayesian consistency metrics
        """
        # Extract metrics
        mean = metrics['overall_avg']
        std = metrics['std_dev']
        n = metrics['matches_played']
        
        # Calculate lower bound (more conservative estimate)
        if n > 0:
            lower_bound = mean - 1.96 * std / np.sqrt(n)
            lower_bound = max(0, lower_bound)
        else:
            lower_bound = 0
        
        # Calculate consistency score
        if mean > 0 and std > 0:
            # Bayesian consistency score
            consistency = 1 / (1 + (std / mean) * np.sqrt(n / (n + 1)))
        else:
            consistency = 0.5
        
        return {
            'mean': mean,
            'std': std,
            'lower_bound': lower_bound,
            'consistency': min(1.0, max(0.0, consistency)),
            'sample_size': n
        }
    
    def adjust_player_predictions(self, player_predictions):
        """
        Adjust player predictions based on form and consistency
        
        Args:
            player_predictions (dict): Dictionary of player predictions
            
        Returns:
            dict: Adjusted player predictions
        """
        # Calculate form metrics if not already calculated
        if not self.form_metrics:
            self.calculate_form_metrics()
        
        # Create a copy of predictions to adjust
        adjusted_predictions = player_predictions.copy()
        
        try:
            for player_name, prediction in player_predictions.items():
                if player_name in self.form_metrics:
                    metrics = self.form_metrics[player_name]
                    
                    # Calculate adjustment factor based on form and consistency
                    form_adjustment = self._calculate_form_adjustment(metrics)
                    
                    # Apply adjustment to prediction
                    if 'predicted_points' in prediction:
                        adjusted_predictions[player_name]['predicted_points'] = \
                            prediction['predicted_points'] * form_adjustment
                        
                        # Add form metrics to prediction
                        adjusted_predictions[player_name]['form_factor'] = metrics['form_factor']
                        adjusted_predictions[player_name]['consistency'] = \
                            self.consistency_scores[player_name]['consistency'] \
                            if player_name in self.consistency_scores else 0.5
                        
                        # Log significant adjustments
                        if abs(form_adjustment - 1.0) > 0.1:
                            logger.info(f"Adjusted {player_name}'s prediction by factor {form_adjustment:.2f} "
                                      f"based on form (factor: {metrics['form_factor']:.2f})")
        
        except Exception as e:
            logger.error(f"Error adjusting player predictions: {str(e)}")
        
        return adjusted_predictions
    
    def _calculate_form_adjustment(self, metrics):
        """
        Calculate form adjustment factor
        
        Args:
            metrics (dict): Player form metrics
            
        Returns:
            float: Form adjustment factor
        """
        # Base adjustment on form factor (recent avg / overall avg)
        form_factor = metrics.get('form_factor', 1.0)
        
        # Consider form trend
        trend = metrics.get('recent_form_trend', 0.0)
        
        # Consider consistency
        consistency = metrics.get('consistency', 0.5)
        
        # Calculate adjustment factor
        # - Form factor has the largest impact
        # - Positive trend increases adjustment
        # - Higher consistency increases adjustment reliability
        adjustment = (form_factor * 0.7) + (np.sign(trend) * min(0.2, abs(trend/10))) + (consistency * 0.1)
        
        # Limit adjustment range
        adjustment = min(1.5, max(0.7, adjustment))
        
        return adjustment
    
    def identify_in_form_players(self, squad_data, top_n=5):
        """
        Identify in-form players from the squad
        
        Args:
            squad_data (pd.DataFrame): Squad data with player information
            top_n (int): Number of top in-form players to identify
            
        Returns:
            list: List of in-form players with metrics
        """
        # Calculate form metrics if not already calculated
        if not self.form_metrics:
            self.calculate_form_metrics()
        
        try:
            # Get list of players in the squad
            player_col = next((col for col in ['player_name', 'name', 'Player'] 
                             if col in squad_data.columns), None)
            
            if player_col is None:
                logger.warning("Could not find player name column in squad data")
                return []
            
            squad_players = squad_data[player_col].tolist()
            
            # Filter form metrics to include only squad players
            squad_form = {player: metrics for player, metrics in self.form_metrics.items() 
                         if player in squad_players}
            
            # Sort by form factor (recent avg / overall avg)
            sorted_players = sorted(squad_form.items(), 
                                   key=lambda x: x[1].get('form_factor', 0), 
                                   reverse=True)
            
            # Get top N in-form players
            top_players = sorted_players[:top_n]
            
            # Format results
            in_form_players = []
            for player_name, metrics in top_players:
                in_form_players.append({
                    'name': player_name,
                    'form_factor': metrics.get('form_factor', 1.0),
                    'recent_avg': metrics.get('recent_avg', 0),
                    'overall_avg': metrics.get('overall_avg', 0),
                    'consistency': metrics.get('consistency', 0.5),
                    'matches_played': metrics.get('matches_played', 0),
                    'trend': 'improving' if metrics.get('recent_form_trend', 0) > 0 else 'declining'
                })
            
            return in_form_players
        
        except Exception as e:
            logger.error(f"Error identifying in-form players: {str(e)}")
            return []
    
    def get_player_form_report(self, player_name):
        """
        Get detailed form report for a specific player
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            dict: Detailed form report
        """
        if player_name not in self.form_metrics:
            logger.warning(f"No form data available for player {player_name}")
            return {}
        
        try:
            metrics = self.form_metrics[player_name]
            consistency = self.consistency_scores.get(player_name, {'consistency': 0.5})
            
            # Create form report
            report = {
                'name': player_name,
                'recent_average': metrics.get('recent_avg', 0),
                'overall_average': metrics.get('overall_avg', 0),
                'form_factor': metrics.get('form_factor', 1.0),
                'consistency_score': consistency.get('consistency', 0.5),
                'matches_played': metrics.get('matches_played', 0),
                'recent_matches': metrics.get('recent_matches', 0),
                'form_trend': metrics.get('recent_form_trend', 0),
                'max_points': metrics.get('max_points', 0),
                'min_points': metrics.get('min_points', 0),
                'standard_deviation': metrics.get('std_dev', 0),
                'weighted_recent_average': metrics.get('weighted_recent_avg', 0),
                'lower_bound_estimate': consistency.get('lower_bound', 0)
            }
            
            # Add form status
            if report['form_factor'] >= 1.2:
                report['form_status'] = 'excellent'
            elif report['form_factor'] >= 1.0:
                report['form_status'] = 'good'
            elif report['form_factor'] >= 0.8:
                report['form_status'] = 'average'
            else:
                report['form_status'] = 'poor'
            
            # Add consistency status
            if report['consistency_score'] >= 0.8:
                report['consistency_status'] = 'very consistent'
            elif report['consistency_score'] >= 0.6:
                report['consistency_status'] = 'consistent'
            elif report['consistency_score'] >= 0.4:
                report['consistency_status'] = 'moderately consistent'
            else:
                report['consistency_status'] = 'inconsistent'
            
            # Add trend description
            if report['form_trend'] > 1:
                report['trend_description'] = 'strongly improving'
            elif report['form_trend'] > 0:
                report['trend_description'] = 'slightly improving'
            elif report['form_trend'] < -1:
                report['trend_description'] = 'strongly declining'
            elif report['form_trend'] < 0:
                report['trend_description'] = 'slightly declining'
            else:
                report['trend_description'] = 'stable'
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating player form report: {str(e)}")
            return {}