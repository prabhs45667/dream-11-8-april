import pandas as pd
import numpy as np
import logging
import traceback
import copy
from typing import Dict, List, Union, Optional, Tuple

# Import local modules
from anti_fragile_strategy import AntiFragileStrategy

# Configure logging
logger = logging.getLogger('monte_carlo_simulation')

class MonteCarloSimulation:
    """
    Enhanced Monte Carlo simulation for Dream11 team selection.
    
    Uses advanced variance modeling and strategies to produce robust teams.
    """
    
    def __init__(self, team_optimizer=None):
        """
        Initialize the MonteCarloSimulation
        
        Args:
            team_optimizer: Reference to TeamOptimizer instance
        """
        self.team_optimizer = team_optimizer
        self.anti_fragile_strategy = AntiFragileStrategy()
        self.default_iterations = 15
        self.default_variance_factor = 0.2
        self.sim_results = {
            'teams': [],
            'scores': [],
            'player_frequencies': {},
            'role_distributions': [],
            'team_distributions': []
        }
        self.core_players = []
        self.flex_players = []
        self.variance_analysis = {}
    
    def run_simulation(self, problem: Dict, 
                      num_iterations: int = None, 
                      variance_factor: float = None, 
                      pitch_type: str = 'balanced',
                      use_anti_fragile: bool = True) -> Dict:
        """
        Run Monte Carlo simulation to generate multiple team variations
        
        Args:
            problem (dict): Optimization problem structure
            num_iterations (int, optional): Number of Monte Carlo iterations
            variance_factor (float, optional): Controls the amount of random variation
            pitch_type (str): Pitch type for team balance adjustments
            use_anti_fragile (bool): Whether to apply anti-fragile strategies
            
        Returns:
            dict: Result containing best team and analysis
        """
        # Set default values if not provided
        if num_iterations is None:
            num_iterations = self.default_iterations
        if variance_factor is None:
            variance_factor = self.default_variance_factor
            
        logger.info(f"Running Monte Carlo simulation with {num_iterations} iterations, "
                  f"variance factor {variance_factor}, pitch type '{pitch_type}'")
                  
        try:
            # Extract team names
            team1 = problem.get('team1', 'Team1')
            team2 = problem.get('team2', 'Team2')
            
            # Validate problem structure
            if not isinstance(problem, dict) or 'players' not in problem:
                logger.error("Invalid problem structure for Monte Carlo simulation")
                return None
                
            # Initialize results
            self.sim_results = {
                'teams': [],
                'scores': [],
                'player_frequencies': {},
                'role_distributions': [],
                'team_distributions': []
            }
            successful_iterations = 0
            
            # Apply anti-fragile strategy if requested
            if use_anti_fragile and 'players' in problem:
                # Convert players dict to DataFrame for anti-fragile processing
                player_df = pd.DataFrame([
                    {**{'name': name}, **player_data} 
                    for name, player_data in problem['players'].items()
                ])
                
                # Apply anti-fragile scores and adjust player values
                player_df = self.anti_fragile_strategy.calculate_anti_fragile_score(
                    player_df, team1, team2, 
                    strategy=self._map_pitch_to_strategy(pitch_type)
                )
                
                # Adjust player values (points)
                if 'anti_fragile_score' in player_df.columns:
                    player_df = self.anti_fragile_strategy.adjust_player_values(player_df, 'points')
                    
                    # Update problem with adjusted points
                    if 'adjusted_points' in player_df.columns:
                        for _, row in player_df.iterrows():
                            player_name = row['name']
                            if player_name in problem['players']:
                                problem['players'][player_name]['original_points'] = problem['players'][player_name]['points']
                                problem['players'][player_name]['points'] = row['adjusted_points']
                
                # Update role requirements based on pitch type
                if 'role_requirements' in problem:
                    role_constraints = self.anti_fragile_strategy.get_role_constraints(pitch_type)
                    problem['role_requirements'] = role_constraints
            
            # Run multiple iterations
            for i in range(num_iterations):
                try:
                    # Create problem variation
                    mc_problem = self._create_problem_variation(problem, variance_factor)
                    
                    # Run optimization
                    logger.info(f"Running Monte Carlo iteration {i+1}/{num_iterations}")
                    if self.team_optimizer:
                        mc_result = self.team_optimizer.optimize_team(mc_problem)
                    else:
                        logger.error("Team optimizer not available. Cannot run optimization.")
                        return None
                    
                    # Process result
                    if mc_result and 'players' in mc_result and mc_result['players']:
                        # Convert to DataFrame for easier analysis
                        team = pd.DataFrame(mc_result['players'])
                        team_score = mc_result.get('total_points', 0)
                        
                        # Add to results
                        self.sim_results['teams'].append(team)
                        self.sim_results['scores'].append(team_score)
                        successful_iterations += 1
                        
                        # Track role distribution
                        role_dist = team['role'].value_counts().to_dict()
                        self.sim_results['role_distributions'].append(role_dist)
                        
                        # Track team distribution
                        team_dist = team['team'].value_counts().to_dict()
                        self.sim_results['team_distributions'].append(team_dist)
                        
                        # Track player selection frequency
                        for _, player in team.iterrows():
                            player_name = self._get_player_name(player)
                            if player_name in self.sim_results['player_frequencies']:
                                self.sim_results['player_frequencies'][player_name] += 1
                            else:
                                self.sim_results['player_frequencies'][player_name] = 1
                    else:
                        logger.warning(f"Monte Carlo iteration {i+1} failed to produce a valid team")
                        
                except Exception as e:
                    logger.error(f"Error in Monte Carlo iteration {i+1}: {str(e)}")
                    traceback.print_exc()
            
            # Check if we have any successful iterations
            if successful_iterations == 0:
                logger.warning("All Monte Carlo iterations failed. Attempting standard optimization.")
                if self.team_optimizer:
                    return self.team_optimizer.optimize_team(problem)
                else:
                    return None
            
            # Analyze results
            self._analyze_simulation_results(successful_iterations)
            
            # Get best team (highest score)
            best_team_idx = self.sim_results['scores'].index(max(self.sim_results['scores']))
            best_team = self.sim_results['teams'][best_team_idx]
            
            # Prepare result
            result = {
                'players': best_team.to_dict('records'),
                'total_points': self.sim_results['scores'][best_team_idx],
                'variance_analysis': self.variance_analysis,
                'is_monte_carlo': True
            }
            
            # Log summary
            role_counts = best_team['role'].value_counts().to_dict()
            logger.info(f"Monte Carlo simulation completed with {successful_iterations} successful iterations")
            logger.info(f"Best team role distribution: {role_counts}")
            logger.info(f"Core players (>70% selection): {self.core_players}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            traceback.print_exc()
            # Attempt standard optimization
            if self.team_optimizer:
                return self.team_optimizer.optimize_team(problem)
            else:
                return None
    
    def _create_problem_variation(self, problem, variance_factor=0.2):
        """Create a variation of the problem with randomized point predictions"""
        # Create deep copy
        mc_problem = copy.deepcopy(problem)
        
        # Add variance to player points
        if 'players' in mc_problem:
            for player_name, player_data in mc_problem['players'].items():
                if 'points' in player_data:
                    # Calculate random variance
                    if 'variance_factor' in player_data:
                        # Use player-specific variance if available
                        player_variance = player_data['variance_factor'] * variance_factor
                    else:
                        player_variance = variance_factor
                    
                    # Generate random multiplier
                    # Using triangular distribution for more realistic variance
                    multiplier = np.random.triangular(
                        1 - player_variance,  # Lower bound
                        1,                    # Mode (most likely)
                        1 + player_variance   # Upper bound
                    )
                    
                    # Store original points if not already saved
                    if 'original_points' not in player_data:
                        player_data['original_points'] = player_data['points']
                    
                    # Apply variance to points
                    player_data['points'] = player_data['original_points'] * multiplier
        
        return mc_problem
        
    def _analyze_simulation_results(self, successful_iterations):
        """Analyze simulation results to extract insights"""
        if successful_iterations == 0:
            return
            
        # Calculate player selection probabilities
        self.variance_analysis['player_selection_probabilities'] = {
            player: count / successful_iterations 
            for player, count in self.sim_results['player_frequencies'].items()
        }
        
        # Identify core players (selected in >70% of teams)
        self.core_players = [
            player for player, prob in self.variance_analysis['player_selection_probabilities'].items() 
            if prob > 0.7
        ]
        self.variance_analysis['core_players'] = self.core_players
        
        # Identify flex players (selected in 30-70% of teams)
        self.flex_players = [
            player for player, prob in self.variance_analysis['player_selection_probabilities'].items() 
            if 0.3 <= prob <= 0.7
        ]
        self.variance_analysis['flex_players'] = self.flex_players
        
        # Calculate team distribution metrics
        if self.sim_results['team_distributions']:
            # Get all possible teams
            all_teams = set()
            for team_dist in self.sim_results['team_distributions']:
                all_teams.update(team_dist.keys())
                
            # Calculate average player count per team
            avg_team_counts = {}
            for team in all_teams:
                team_counts = [
                    team_dist.get(team, 0) for team_dist in self.sim_results['team_distributions']
                ]
                avg_team_counts[team] = sum(team_counts) / len(team_counts)
                
            self.variance_analysis['avg_team_distribution'] = avg_team_counts
        
        # Calculate role distribution metrics
        if self.sim_results['role_distributions']:
            # Get all possible roles
            all_roles = set()
            for role_dist in self.sim_results['role_distributions']:
                all_roles.update(role_dist.keys())
                
            # Calculate average player count per role
            avg_role_counts = {}
            for role in all_roles:
                role_counts = [
                    role_dist.get(role, 0) for role_dist in self.sim_results['role_distributions']
                ]
                avg_role_counts[role] = sum(role_counts) / len(role_counts)
                
            self.variance_analysis['avg_role_distribution'] = avg_role_counts
        
        # Calculate general statistics
        self.variance_analysis['total_iterations'] = successful_iterations
        self.variance_analysis['average_team_score'] = sum(self.sim_results['scores']) / len(self.sim_results['scores'])
        self.variance_analysis['score_variance'] = np.var(self.sim_results['scores']) if len(self.sim_results['scores']) > 1 else 0
        
    def _get_player_name(self, player_row):
        """Extract player name from a row, handling different column names"""
        for name_col in ['Player Name', 'player_name', 'name', 'Name']:
            if name_col in player_row:
                return player_row[name_col]
        return 'Unknown'
    
    def _map_pitch_to_strategy(self, pitch_type):
        """Map pitch type to role strategy"""
        pitch_to_strategy = {
            'batting_friendly': 'batting_heavy',
            'bowling_friendly': 'bowling_heavy',
            'balanced': 'balanced'
        }
        return pitch_to_strategy.get(pitch_type, 'balanced')
    
    def get_stats_for_core_players(self):
        """Get detailed statistics for core players"""
        if not self.core_players:
            return {}
            
        # Extract teams with core players
        core_player_teams = []
        for team in self.sim_results['teams']:
            core_in_team = []
            for _, player in team.iterrows():
                player_name = self._get_player_name(player)
                if player_name in self.core_players:
                    core_in_team.append(player_name)
            if core_in_team:
                core_player_teams.append((team, core_in_team))
        
        # Calculate average points for each core player
        core_player_stats = {}
        for player in self.core_players:
            points = []
            roles = []
            for team, core_players in core_player_teams:
                if player in core_players:
                    # Find player in team
                    for _, row in team.iterrows():
                        if self._get_player_name(row) == player:
                            # Extract points and role
                            points_val = 0
                            for points_col in ['predicted_points', 'points', 'Points']:
                                if points_col in row:
                                    points_val = row[points_col]
                                    break
                            
                            role_val = 'Unknown'
                            for role_col in ['role', 'Role']:
                                if role_col in row:
                                    role_val = row[role_col]
                                    break
                            
                            points.append(points_val)
                            roles.append(role_val)
                            break
            
            # Calculate statistics
            if points:
                core_player_stats[player] = {
                    'avg_points': sum(points) / len(points),
                    'min_points': min(points),
                    'max_points': max(points),
                    'std_points': np.std(points) if len(points) > 1 else 0,
                    'most_common_role': max(set(roles), key=roles.count) if roles else 'Unknown',
                    'selection_frequency': self.variance_analysis['player_selection_probabilities'].get(player, 0)
                }
        
        return core_player_stats
    
    def recommend_captain_vice_captain(self):
        """Recommend captain and vice-captain based on simulation results"""
        if not self.sim_results['teams']:
            return None, None
            
        # Get player stats across all teams
        player_stats = {}
        for i, team in enumerate(self.sim_results['teams']):
            for _, player in team.iterrows():
                player_name = self._get_player_name(player)
                
                # Extract points
                points_val = 0
                for points_col in ['predicted_points', 'points', 'Points']:
                    if points_col in player:
                        points_val = player[points_col]
                        break
                
                # Extract role
                role_val = 'Unknown'
                for role_col in ['role', 'Role']:
                    if role_col in player:
                        role_val = player[role_col]
                        break
                
                # Update stats
                if player_name not in player_stats:
                    player_stats[player_name] = {
                        'points': [points_val],
                        'roles': [role_val],
                        'teams': [i]
                    }
                else:
                    player_stats[player_name]['points'].append(points_val)
                    player_stats[player_name]['roles'].append(role_val)
                    player_stats[player_name]['teams'].append(i)
        
        # Calculate average points and selection frequency
        for player, stats in player_stats.items():
            stats['avg_points'] = sum(stats['points']) / len(stats['points'])
            stats['selection_freq'] = len(stats['teams']) / len(self.sim_results['teams'])
            stats['most_common_role'] = max(set(stats['roles']), key=stats['roles'].count)
        
        # Calculate captain score (considering points and selection frequency)
        for player, stats in player_stats.items():
            # Base score on average points
            captain_score = stats['avg_points']
            
            # Boost if frequently selected (core player)
            if stats['selection_freq'] > 0.7:
                captain_score *= 1.2
            elif stats['selection_freq'] > 0.5:
                captain_score *= 1.1
            
            # Boost based on role (AR and BAT typically better captains)
            role_boosts = {
                'AR': 1.2,
                'BAT': 1.15,
                'BOWL': 1.05,
                'WK': 1.1
            }
            
            role = stats['most_common_role']
            if role in role_boosts:
                captain_score *= role_boosts[role]
            
            stats['captain_score'] = captain_score
        
        # Rank by captain score
        ranked_players = sorted(
            [(player, stats['captain_score']) for player, stats in player_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top two as captain and vice-captain
        if len(ranked_players) >= 2:
            return ranked_players[0][0], ranked_players[1][0]
        elif len(ranked_players) == 1:
            return ranked_players[0][0], None
        else:
            return None, None 