import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logger = logging.getLogger('matchup_analyzer')

class MatchupAnalyzer:
    """
    Analyzes player performance against specific opponents.
    
    This module identifies favorable player vs. team matchups and analyzes batsman vs. bowler
    historical performance to adjust player predictions based on opposition.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the MatchupAnalyzer
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.matchup_data = None
        self.player_vs_team = {}
        self.batsman_vs_bowler = {}
        
    def load_matchup_data(self, file_path=None):
        """
        Load matchup data from file or create from match data
        
        Args:
            file_path (str, optional): Path to ball-by-ball data file
            
        Returns:
            pd.DataFrame: Loaded matchup data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "IPL_Ball_by_Ball_2008_2022.csv")
            
        try:
            if os.path.exists(file_path):
                # Load ball-by-ball data
                ball_by_ball_df = pd.read_csv(file_path)
                logger.info(f"Loaded ball-by-ball data from {file_path}: {ball_by_ball_df.shape[0]} records")
                
                # Extract player vs team matchups
                self.player_vs_team = self._extract_player_vs_team(ball_by_ball_df)
                
                # Extract batsman vs bowler matchups
                self.batsman_vs_bowler = self._extract_batsman_vs_bowler(ball_by_ball_df)
                
                # Combine into single matchup dataset
                self.matchup_data = self._combine_matchups()
                
                logger.info(f"Extracted {len(self.player_vs_team)} player vs team matchups and "
                           f"{len(self.batsman_vs_bowler)} batsman vs bowler matchups")
                
                return self.matchup_data
            else:
                logger.warning(f"Matchup data file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading matchup data: {str(e)}")
            return None
    
    def _extract_player_vs_team(self, ball_by_ball_df):
        """
        Extract player vs team matchups from ball-by-ball data
        
        Args:
            ball_by_ball_df (pd.DataFrame): Ball-by-ball match data
            
        Returns:
            dict: Dictionary of player vs team matchups with performance metrics
        """
        player_vs_team = {}
        
        try:
            # Get match data with team information
            match_file = os.path.join(self.data_dir, "IPL_Matches_2008_2022.csv")
            if os.path.exists(match_file):
                match_df = pd.read_csv(match_file)
                
                # Merge match data to get team information
                merged_df = pd.merge(ball_by_ball_df, match_df, left_on='ID', right_on='id', how='left')
                
                # Extract batsman vs team matchups
                for (batsman, bowling_team), group in merged_df.groupby(['batsman', 'bowling_team']):
                    # Calculate batting stats
                    runs = group['batsman_runs'].sum()
                    balls = len(group)
                    dismissals = len(group[group['player_dismissed'] == batsman])
                    
                    # Calculate batting average and strike rate
                    avg = runs / dismissals if dismissals > 0 else runs
                    sr = (runs / balls) * 100 if balls > 0 else 0
                    
                    # Create matchup key
                    key = f"{batsman}_vs_{bowling_team}"
                    
                    # Store matchup data
                    player_vs_team[key] = {
                        'player': batsman,
                        'team': bowling_team,
                        'role': 'batsman',
                        'runs': runs,
                        'balls': balls,
                        'dismissals': dismissals,
                        'average': avg,
                        'strike_rate': sr,
                        'matches': len(group['ID'].unique())
                    }
                
                # Extract bowler vs team matchups
                for (bowler, batting_team), group in merged_df.groupby(['bowler', 'batting_team']):
                    # Calculate bowling stats
                    runs = group['total_runs'].sum()
                    balls = len(group)
                    wickets = len(group[group['player_dismissed'].notna()])
                    
                    # Calculate bowling average and economy
                    avg = runs / wickets if wickets > 0 else float('inf')
                    economy = (runs / (balls/6)) if balls > 0 else 0
                    
                    # Create matchup key
                    key = f"{bowler}_vs_{batting_team}"
                    
                    # Store matchup data
                    player_vs_team[key] = {
                        'player': bowler,
                        'team': batting_team,
                        'role': 'bowler',
                        'runs': runs,
                        'balls': balls,
                        'wickets': wickets,
                        'average': avg,
                        'economy': economy,
                        'matches': len(group['ID'].unique())
                    }
        
        except Exception as e:
            logger.error(f"Error extracting player vs team matchups: {str(e)}")
        
        return player_vs_team
    
    def _extract_batsman_vs_bowler(self, ball_by_ball_df):
        """
        Extract batsman vs bowler matchups from ball-by-ball data
        
        Args:
            ball_by_ball_df (pd.DataFrame): Ball-by-ball match data
            
        Returns:
            dict: Dictionary of batsman vs bowler matchups with performance metrics
        """
        batsman_vs_bowler = {}
        
        try:
            # Group by batsman and bowler
            for (batsman, bowler), group in ball_by_ball_df.groupby(['batsman', 'bowler']):
                # Calculate stats
                runs = group['batsman_runs'].sum()
                balls = len(group)
                dismissals = len(group[(group['player_dismissed'] == batsman) & 
                                     (group['bowler'] == bowler)])
                
                # Calculate batting average and strike rate
                avg = runs / dismissals if dismissals > 0 else runs
                sr = (runs / balls) * 100 if balls > 0 else 0
                
                # Create matchup key
                key = f"{batsman}_vs_{bowler}"
                
                # Store matchup data
                batsman_vs_bowler[key] = {
                    'batsman': batsman,
                    'bowler': bowler,
                    'runs': runs,
                    'balls': balls,
                    'dismissals': dismissals,
                    'average': avg,
                    'strike_rate': sr,
                    'matches': len(group['ID'].unique())
                }
        
        except Exception as e:
            logger.error(f"Error extracting batsman vs bowler matchups: {str(e)}")
        
        return batsman_vs_bowler
    
    def _combine_matchups(self):
        """
        Combine player vs team and batsman vs bowler matchups into a single dataset
        
        Returns:
            pd.DataFrame: Combined matchup data
        """
        # Convert dictionaries to DataFrames
        player_team_df = pd.DataFrame(list(self.player_vs_team.values()))
        batsman_bowler_df = pd.DataFrame(list(self.batsman_vs_bowler.values()))
        
        # Return combined data if both are available
        if not player_team_df.empty and not batsman_bowler_df.empty:
            return {
                'player_vs_team': player_team_df,
                'batsman_vs_bowler': batsman_bowler_df
            }
        elif not player_team_df.empty:
            return {'player_vs_team': player_team_df}
        elif not batsman_bowler_df.empty:
            return {'batsman_vs_bowler': batsman_bowler_df}
        else:
            return {}
    
    def get_player_vs_team_matchup(self, player_name, team_name):
        """
        Get player vs team matchup data
        
        Args:
            player_name (str): Name of the player
            team_name (str): Name of the team
            
        Returns:
            dict: Matchup data for player vs team
        """
        key = f"{player_name}_vs_{team_name}"
        return self.player_vs_team.get(key, {})
    
    def get_batsman_vs_bowler_matchup(self, batsman_name, bowler_name):
        """
        Get batsman vs bowler matchup data
        
        Args:
            batsman_name (str): Name of the batsman
            bowler_name (str): Name of the bowler
            
        Returns:
            dict: Matchup data for batsman vs bowler
        """
        key = f"{batsman_name}_vs_{bowler_name}"
        return self.batsman_vs_bowler.get(key, {})
    
    def adjust_player_predictions(self, player_predictions, team1, team2):
        """
        Adjust player predictions based on matchup data
        
        Args:
            player_predictions (dict): Dictionary of player predictions
            team1 (str): First team name
            team2 (str): Second team name
            
        Returns:
            dict: Adjusted player predictions
        """
        # Create a copy of predictions to adjust
        adjusted_predictions = player_predictions.copy()
        
        try:
            for player_name, prediction in player_predictions.items():
                # Get player team
                player_team = prediction.get('team', '')
                
                # Determine opposition team
                opposition_team = team2 if player_team == team1 else team1
                
                # Get player vs team matchup
                matchup = self.get_player_vs_team_matchup(player_name, opposition_team)
                
                if matchup:
                    # Calculate adjustment factor based on matchup data
                    adjustment_factor = self._calculate_matchup_adjustment(matchup, prediction.get('role', ''))
                    
                    # Apply adjustment to prediction
                    if 'predicted_points' in prediction:
                        adjusted_predictions[player_name]['predicted_points'] = \
                            prediction['predicted_points'] * adjustment_factor
                        
                        # Add matchup data to prediction
                        adjusted_predictions[player_name]['matchup_factor'] = adjustment_factor
                        
                        # Log significant adjustments
                        if abs(adjustment_factor - 1.0) > 0.1:
                            logger.info(f"Adjusted {player_name}'s prediction by factor {adjustment_factor:.2f} "
                                      f"based on matchup vs {opposition_team}")
        
        except Exception as e:
            logger.error(f"Error adjusting player predictions: {str(e)}")
        
        return adjusted_predictions
    
    def _calculate_matchup_adjustment(self, matchup, role):
        """
        Calculate matchup adjustment factor
        
        Args:
            matchup (dict): Matchup data
            role (str): Player role (batsman, bowler, all-rounder)
            
        Returns:
            float: Matchup adjustment factor
        """
        # Default adjustment factor
        adjustment = 1.0
        
        try:
            # Minimum matches threshold for reliable adjustment
            min_matches = 3
            
            if matchup.get('matches', 0) < min_matches:
                return adjustment
            
            # Calculate adjustment based on role
            if role in ['batsman', 'BAT', 'WK'] or matchup.get('role') == 'batsman':
                # For batsmen, higher average and strike rate is better
                avg = matchup.get('average', 0)
                sr = matchup.get('strike_rate', 0)
                
                # Baseline values for comparison
                baseline_avg = 25.0
                baseline_sr = 125.0
                
                # Calculate adjustment factor
                avg_factor = min(1.5, max(0.7, avg / baseline_avg))
                sr_factor = min(1.3, max(0.8, sr / baseline_sr))
                
                # Combined adjustment (weighted average)
                adjustment = (avg_factor * 0.6) + (sr_factor * 0.4)
                
            elif role in ['bowler', 'BOWL'] or matchup.get('role') == 'bowler':
                # For bowlers, lower average and economy is better
                avg = matchup.get('average', float('inf'))
                economy = matchup.get('economy', 10.0)
                
                # Baseline values for comparison
                baseline_avg = 25.0
                baseline_economy = 8.0
                
                # Calculate adjustment factor (inverted for bowling)
                avg_factor = min(1.5, max(0.7, baseline_avg / avg if avg > 0 else 1.5))
                economy_factor = min(1.3, max(0.8, baseline_economy / economy if economy > 0 else 1.3))
                
                # Combined adjustment (weighted average)
                adjustment = (avg_factor * 0.6) + (economy_factor * 0.4)
            
            # Limit adjustment range
            adjustment = min(1.5, max(0.7, adjustment))
            
        except Exception as e:
            logger.error(f"Error calculating matchup adjustment: {str(e)}")
            adjustment = 1.0
        
        return adjustment
    
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
        try:
            # Get list of players in the squad
            player_col = next((col for col in ['player_name', 'name', 'Player'] 
                             if col in squad_data.columns), None)
            
            if player_col is None:
                logger.warning("Could not find player name column in squad data")
                return []
            
            squad_players = squad_data[player_col].tolist()
            
            # Get matchups for squad players vs opposition team
            matchups = []
            for player in squad_players:
                matchup = self.get_player_vs_team_matchup(player, opposition_team)
                if matchup and matchup.get('matches', 0) >= 2:  # Minimum 2 matches for reliability
                    # Calculate matchup score
                    role = matchup.get('role', '')
                    if role == 'batsman':
                        score = (matchup.get('average', 0) / 25.0) * (matchup.get('strike_rate', 0) / 125.0)
                    elif role == 'bowler':
                        # Invert for bowling (lower is better)
                        avg = matchup.get('average', float('inf'))
                        economy = matchup.get('economy', 10.0)
                        score = (25.0 / avg if avg > 0 else 2.0) * (8.0 / economy if economy > 0 else 1.5)
                    else:
                        score = 1.0
                    
                    matchups.append({
                        'player': player,
                        'opposition': opposition_team,
                        'role': role,
                        'matches': matchup.get('matches', 0),
                        'score': score,
                        'details': matchup
                    })
            
            # Sort by matchup score
            sorted_matchups = sorted(matchups, key=lambda x: x['score'], reverse=True)
            
            # Get top N favorable matchups
            top_matchups = sorted_matchups[:top_n]
            
            # Format results
            favorable_matchups = []
            for matchup in top_matchups:
                details = matchup['details']
                if matchup['role'] == 'batsman':
                    description = f"{matchup['player']} averages {details.get('average', 0):.1f} with a "
                    description += f"strike rate of {details.get('strike_rate', 0):.1f} against {matchup['opposition']}"
                elif matchup['role'] == 'bowler':
                    description = f"{matchup['player']} has taken {details.get('wickets', 0)} wickets with an "
                    description += f"economy of {details.get('economy', 0):.1f} against {matchup['opposition']}"
                else:
                    description = f"{matchup['player']} has a favorable matchup against {matchup['opposition']}"
                
                favorable_matchups.append({
                    'player': matchup['player'],
                    'opposition': matchup['opposition'],
                    'role': matchup['role'],
                    'score': matchup['score'],
                    'description': description
                })
            
            return favorable_matchups
        
        except Exception as e:
            logger.error(f"Error identifying favorable matchups: {str(e)}")
            return []