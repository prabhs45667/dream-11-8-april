import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pulp import *

class TeamOptimizer:
    def __init__(self, max_credits: int = 100):
        self.max_credits = max_credits
        self.required_roles = {
            'WK': 1,
            'BAT': 3,
            'AR': 3,
            'BOWL': 4
        }
        
    def create_optimization_problem(self, players: pd.DataFrame, predicted_points: np.ndarray) -> LpProblem:
        """Create and solve the team optimization problem"""
        prob = LpProblem("Dream11_Team_Selection", LpMaximize)
        
        # Create binary variables for each player
        player_vars = LpVariable.dicts("Player",
                                     players.index,
                                     cat='Binary')
        
        # Objective: Maximize predicted points
        prob += lpSum([predicted_points[i] * player_vars[i] for i in players.index])
        
        # Constraint 1: Total players = 11
        prob += lpSum([player_vars[i] for i in players.index]) == 11
        
        # Constraint 2: Total credits <= 100
        prob += lpSum([players.loc[i, 'credits'] * player_vars[i] for i in players.index]) <= self.max_credits
        
        # Constraint 3: Role requirements
        for role, count in self.required_roles.items():
            prob += lpSum([player_vars[i] for i in players.index if players.loc[i, 'role'] == role]) == count
            
        return prob, player_vars
    
    def optimize_team(self, players: pd.DataFrame, predicted_points: np.ndarray) -> Tuple[List[str], float]:
        """Optimize team selection using linear programming"""
        prob, player_vars = self.create_optimization_problem(players, predicted_points)
        
        # Solve the optimization problem
        prob.solve()
        
        # Get selected players
        selected_players = []
        total_points = 0
        
        for i in players.index:
            if player_vars[i].value() == 1:
                selected_players.append(i)
                total_points += predicted_points[i]
                
        return selected_players, total_points
    
    def create_backup_team(self, players: pd.DataFrame, predicted_points: np.ndarray, 
                          main_team: List[str], max_backups: int = 4) -> List[str]:
        """Create backup team for each role"""
        backup_team = []
        role_backups = {role: [] for role in self.required_roles.keys()}
        
        # Sort players by predicted points within each role
        for role in self.required_roles.keys():
            role_players = players[players['role'] == role]
            role_points = predicted_points[role_players.index]
            
            # Sort players by points (excluding main team players)
            available_players = role_players[~role_players.index.isin(main_team)]
            available_points = predicted_points[available_players.index]
            
            sorted_indices = np.argsort(available_points)[::-1]
            role_backups[role] = available_players.index[sorted_indices].tolist()
        
        # Select top backup for each role
        for role, backups in role_backups.items():
            if backups and len(backup_team) < max_backups:
                backup_team.append(backups[0])
                
        return backup_team
    
    def format_team_output(self, main_team: List[str], backup_team: List[str], 
                          players: pd.DataFrame, predicted_points: np.ndarray) -> Dict:
        """Format team output with player details"""
        team_output = {
            'main_team': [],
            'backup_team': [],
            'total_predicted_points': 0,
            'total_credits': 0
        }
        
        # Format main team
        for player in main_team:
            player_info = {
                'name': player,
                'role': players.loc[player, 'role'],
                'credits': players.loc[player, 'credits'],
                'predicted_points': predicted_points[players.index.get_loc(player)]
            }
            team_output['main_team'].append(player_info)
            team_output['total_predicted_points'] += player_info['predicted_points']
            team_output['total_credits'] += player_info['credits']
            
        # Format backup team
        for player in backup_team:
            player_info = {
                'name': player,
                'role': players.loc[player, 'role'],
                'credits': players.loc[player, 'credits'],
                'predicted_points': predicted_points[players.index.get_loc(player)]
            }
            team_output['backup_team'].append(player_info)
            
        return team_output 