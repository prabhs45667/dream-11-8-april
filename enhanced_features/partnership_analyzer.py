import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx

# Configure logging
logger = logging.getLogger('partnership_analyzer')

class PartnershipAnalyzer:
    """
    Analyzes batting and bowling partnerships to optimize team selection.
    
    This module identifies strong batting and bowling partnerships to enhance team selection
    by analyzing historical partnership data between players and calculating partnership
    strength based on statistical metrics.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the PartnershipAnalyzer
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.partnership_data = None
        self.partnership_graph = None
        self.partnership_strength = {}
        self.batting_partnerships = {}
        self.bowling_partnerships = {}
        
    def load_partnership_data(self, file_path=None):
        """
        Load partnership data from file or create from match data
        
        Args:
            file_path (str, optional): Path to partnership data file
            
        Returns:
            pd.DataFrame: Loaded partnership data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "IPL_Ball_by_Ball_2008_2022.csv")
            
        try:
            if os.path.exists(file_path):
                # Load ball-by-ball data
                ball_by_ball_df = pd.read_csv(file_path)
                logger.info(f"Loaded ball-by-ball data from {file_path}: {ball_by_ball_df.shape[0]} records")
                
                # Extract batting partnerships
                self.batting_partnerships = self._extract_batting_partnerships(ball_by_ball_df)
                
                # Extract bowling partnerships
                self.bowling_partnerships = self._extract_bowling_partnerships(ball_by_ball_df)
                
                # Combine into single partnership dataset
                self.partnership_data = self._combine_partnerships()
                
                logger.info(f"Extracted {len(self.batting_partnerships)} batting partnerships and "
                           f"{len(self.bowling_partnerships)} bowling partnerships")
                
                return self.partnership_data
            else:
                logger.warning(f"Partnership data file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading partnership data: {str(e)}")
            return None
    
    def _extract_batting_partnerships(self, ball_by_ball_df):
        """
        Extract batting partnerships from ball-by-ball data
        
        Args:
            ball_by_ball_df (pd.DataFrame): Ball-by-ball match data
            
        Returns:
            dict: Dictionary of batting partnerships with strength metrics
        """
        batting_partnerships = {}
        
        try:
            # Group by match_id and innings to identify partnerships
            for (match_id, innings), group in ball_by_ball_df.groupby(['ID', 'innings']):
                # Get unique batsmen in this innings
                batsmen = group['batsman'].unique()
                
                # Create partnerships for all pairs of batsmen
                for i in range(len(batsmen)):
                    for j in range(i+1, len(batsmen)):
                        batter1 = batsmen[i]
                        batter2 = batsmen[j]
                        
                        # Get balls where both batsmen were at crease
                        partnership_balls = group[
                            (group['batsman'] == batter1) & (group['non_striker'] == batter2) |
                            (group['batsman'] == batter2) & (group['non_striker'] == batter1)
                        ]
                        
                        if len(partnership_balls) > 0:
                            # Calculate partnership runs
                            runs = partnership_balls['batsman_runs'].sum() + partnership_balls['extras'].sum()
                            balls = len(partnership_balls)
                            
                            # Create partnership key
                            key = f"{batter1}_{batter2}"
                            
                            # Update partnership stats
                            if key not in batting_partnerships:
                                batting_partnerships[key] = {
                                    'player1': batter1,
                                    'player2': batter2,
                                    'type': 'batting',
                                    'runs': runs,
                                    'balls': balls,
                                    'matches': 1,
                                    'strike_rate': (runs / balls) * 100 if balls > 0 else 0
                                }
                            else:
                                batting_partnerships[key]['runs'] += runs
                                batting_partnerships[key]['balls'] += balls
                                batting_partnerships[key]['matches'] += 1
                                batting_partnerships[key]['strike_rate'] = \
                                    (batting_partnerships[key]['runs'] / batting_partnerships[key]['balls']) * 100 \
                                    if batting_partnerships[key]['balls'] > 0 else 0
            
            # Calculate partnership strength
            for key, partnership in batting_partnerships.items():
                # Calculate strength based on runs, strike rate and matches
                strength = (partnership['runs'] * 0.5 + 
                           partnership['strike_rate'] * 0.3 + 
                           partnership['matches'] * 0.2) / 100
                
                # Normalize strength to 0-1 range
                partnership['strength'] = min(1.0, max(0.0, strength))
                
                # Categorize strength
                if partnership['strength'] >= 0.8:
                    partnership['strength_category'] = 'strong'
                elif partnership['strength'] >= 0.6:
                    partnership['strength_category'] = 'good'
                elif partnership['strength'] >= 0.4:
                    partnership['strength_category'] = 'neutral'
                else:
                    partnership['strength_category'] = 'poor'
        
        except Exception as e:
            logger.error(f"Error extracting batting partnerships: {str(e)}")
        
        return batting_partnerships
    
    def _extract_bowling_partnerships(self, ball_by_ball_df):
        """
        Extract bowling partnerships from ball-by-ball data
        
        Args:
            ball_by_ball_df (pd.DataFrame): Ball-by-ball match data
            
        Returns:
            dict: Dictionary of bowling partnerships with strength metrics
        """
        bowling_partnerships = {}
        
        try:
            # Group by match_id and innings to identify partnerships
            for (match_id, innings), group in ball_by_ball_df.groupby(['ID', 'innings']):
                # Get unique bowlers in this innings
                bowlers = group['bowler'].unique()
                
                # Create partnerships for all pairs of bowlers
                for i in range(len(bowlers)):
                    for j in range(i+1, len(bowlers)):
                        bowler1 = bowlers[i]
                        bowler2 = bowlers[j]
                        
                        # Get balls bowled by both bowlers
                        bowler1_balls = group[group['bowler'] == bowler1]
                        bowler2_balls = group[group['bowler'] == bowler2]
                        
                        if len(bowler1_balls) > 0 and len(bowler2_balls) > 0:
                            # Calculate partnership stats
                            wickets1 = len(bowler1_balls[bowler1_balls['player_dismissed'].notna()])
                            wickets2 = len(bowler2_balls[bowler2_balls['player_dismissed'].notna()])
                            total_wickets = wickets1 + wickets2
                            
                            runs1 = bowler1_balls['total_runs'].sum()
                            runs2 = bowler2_balls['total_runs'].sum()
                            total_runs = runs1 + runs2
                            
                            balls1 = len(bowler1_balls)
                            balls2 = len(bowler2_balls)
                            total_balls = balls1 + balls2
                            
                            # Create partnership key
                            key = f"{bowler1}_{bowler2}"
                            
                            # Update partnership stats
                            if key not in bowling_partnerships:
                                bowling_partnerships[key] = {
                                    'player1': bowler1,
                                    'player2': bowler2,
                                    'type': 'bowling',
                                    'wickets': total_wickets,
                                    'runs': total_runs,
                                    'balls': total_balls,
                                    'matches': 1,
                                    'economy': (total_runs / (total_balls/6)) if total_balls > 0 else 0
                                }
                            else:
                                bowling_partnerships[key]['wickets'] += total_wickets
                                bowling_partnerships[key]['runs'] += total_runs
                                bowling_partnerships[key]['balls'] += total_balls
                                bowling_partnerships[key]['matches'] += 1
                                bowling_partnerships[key]['economy'] = \
                                    (bowling_partnerships[key]['runs'] / (bowling_partnerships[key]['balls']/6)) \
                                    if bowling_partnerships[key]['balls'] > 0 else 0
            
            # Calculate partnership strength
            for key, partnership in bowling_partnerships.items():
                # Calculate strength based on wickets, economy and matches
                # Lower economy is better, so we invert it
                max_economy = 12.0  # Maximum economy rate to consider
                normalized_economy = 1 - min(1.0, partnership['economy'] / max_economy)
                
                strength = (partnership['wickets'] * 0.5 + 
                           normalized_economy * 0.3 + 
                           partnership['matches'] * 0.2) / 10
                
                # Normalize strength to 0-1 range
                partnership['strength'] = min(1.0, max(0.0, strength))
                
                # Categorize strength
                if partnership['strength'] >= 0.8:
                    partnership['strength_category'] = 'strong'
                elif partnership['strength'] >= 0.6:
                    partnership['strength_category'] = 'good'
                elif partnership['strength'] >= 0.4:
                    partnership['strength_category'] = 'neutral'
                else:
                    partnership['strength_category'] = 'poor'
        
        except Exception as e:
            logger.error(f"Error extracting bowling partnerships: {str(e)}")
        
        return bowling_partnerships
    
    def _combine_partnerships(self):
        """
        Combine batting and bowling partnerships into a single dataset
        
        Returns:
            pd.DataFrame: Combined partnership data
        """
        # Convert dictionaries to DataFrames
        batting_df = pd.DataFrame(list(self.batting_partnerships.values()))
        bowling_df = pd.DataFrame(list(self.bowling_partnerships.values()))
        
        # Combine DataFrames
        if not batting_df.empty and not bowling_df.empty:
            combined_df = pd.concat([batting_df, bowling_df], ignore_index=True)
        elif not batting_df.empty:
            combined_df = batting_df
        elif not bowling_df.empty:
            combined_df = bowling_df
        else:
            combined_df = pd.DataFrame()
        
        return combined_df
    
    def create_partnership_graph(self):
        """
        Create a graph representation of player partnerships
        
        Returns:
            nx.Graph: NetworkX graph of player partnerships
        """
        try:
            # Create graph
            G = nx.Graph()
            
            # Add partnerships as edges
            if self.partnership_data is not None and not self.partnership_data.empty:
                for _, row in self.partnership_data.iterrows():
                    player1 = row['player1']
                    player2 = row['player2']
                    strength = row['strength']
                    partnership_type = row['type']
                    
                    # Add nodes if they don't exist
                    if not G.has_node(player1):
                        G.add_node(player1)
                    if not G.has_node(player2):
                        G.add_node(player2)
                    
                    # Add edge with attributes
                    G.add_edge(player1, player2, 
                              weight=strength, 
                              type=partnership_type,
                              strength_category=row['strength_category'])
            
            self.partnership_graph = G
            logger.info(f"Created partnership graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            return G
        except Exception as e:
            logger.error(f"Error creating partnership graph: {str(e)}")
            return nx.Graph()
    
    def adjust_player_predictions(self, player_predictions, squad_data):
        """
        Adjust player predictions based on partnership strength
        
        Args:
            player_predictions (dict): Dictionary of player predictions
            squad_data (pd.DataFrame): Squad data with player information
            
        Returns:
            dict: Adjusted player predictions
        """
        if self.partnership_graph is None:
            # Create graph if not already created
            self.create_partnership_graph()
            
        if self.partnership_graph is None or self.partnership_graph.number_of_nodes() == 0:
            logger.warning("Partnership graph is empty, cannot adjust predictions")
            return player_predictions
        
        # Create a copy of predictions to adjust
        adjusted_predictions = player_predictions.copy()
        
        try:
            # Get list of players in the squad
            squad_players = squad_data['name'].tolist() if 'name' in squad_data.columns else []
            
            # Adjust predictions based on partnerships
            for player_name, prediction in player_predictions.items():
                if player_name in self.partnership_graph:
                    # Get player's partnerships
                    partnerships = self.partnership_graph[player_name]
                    
                    # Calculate adjustment factor based on partnerships with other squad players
                    adjustment_factor = 1.0
                    strong_partnerships = 0
                    
                    for partner, attrs in partnerships.items():
                        if partner in squad_players:
                            # Adjust based on partnership strength
                            strength = attrs.get('weight', 0.5)
                            
                            # Strong partnerships increase adjustment factor
                            if strength >= 0.8:
                                adjustment_factor += 0.15
                                strong_partnerships += 1
                            elif strength >= 0.6:
                                adjustment_factor += 0.1
                                strong_partnerships += 1
                            elif strength >= 0.4:
                                adjustment_factor += 0.05
                    
                    # Cap adjustment factor
                    adjustment_factor = min(1.5, adjustment_factor)
                    
                    # Apply adjustment to prediction
                    if 'predicted_points' in prediction:
                        adjusted_predictions[player_name]['predicted_points'] = \
                            prediction['predicted_points'] * adjustment_factor
                        
                        # Log significant adjustments
                        if strong_partnerships > 0:
                            logger.info(f"Adjusted {player_name}'s prediction by factor {adjustment_factor:.2f} "
                                      f"based on {strong_partnerships} strong partnerships")
        
        except Exception as e:
            logger.error(f"Error adjusting player predictions: {str(e)}")
        
        return adjusted_predictions
    
    def recommend_partnerships(self, squad_data, top_n=5):
        """
        Recommend strong partnerships to include in team selection
        
        Args:
            squad_data (pd.DataFrame): Squad data with player information
            top_n (int): Number of top partnerships to recommend
            
        Returns:
            list: List of recommended partnerships
        """
        if self.partnership_data is None or self.partnership_data.empty:
            logger.warning("Partnership data is empty, cannot recommend partnerships")
            return []
        
        try:
            # Get list of players in the squad
            squad_players = squad_data['name'].tolist() if 'name' in squad_data.columns else []
            
            # Filter partnerships to include only squad players
            squad_partnerships = self.partnership_data[
                (self.partnership_data['player1'].isin(squad_players)) & 
                (self.partnership_data['player2'].isin(squad_players))
            ]
            
            # Sort by strength
            sorted_partnerships = squad_partnerships.sort_values('strength', ascending=False)
            
            # Get top N partnerships
            top_partnerships = sorted_partnerships.head(top_n)
            
            # Format recommendations
            recommendations = []
            for _, row in top_partnerships.iterrows():
                recommendations.append({
                    'player1': row['player1'],
                    'player2': row['player2'],
                    'type': row['type'],
                    'strength': row['strength'],
                    'strength_category': row['strength_category'],
                    'description': f"{row['player1']} and {row['player2']} have a {row['strength_category']} "
                                 f"{row['type']} partnership (strength: {row['strength']:.2f})"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error recommending partnerships: {str(e)}")
            return []