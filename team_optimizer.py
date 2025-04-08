import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pulp import *
import traceback
from role_mapper import RoleMapper

class TeamOptimizer:
    def __init__(self):
        """Initialize the team optimizer"""
        self.selected_teams = []
        self.role_mapper = RoleMapper()
        
    def standardize_role(self, role):
        """Standardize player role to one of: WK, BAT, AR, BOWL"""
        return self.role_mapper.standardize_role(role)
        
    def create_optimization_problem(self, squad_data):
        """Create optimization problem for team selection"""
        try:
            # Create a copy to avoid modifying the original
            squad_data = squad_data.copy()
            
            # Ensure roles are standardized
            if 'Player Type' in squad_data.columns:
                squad_data['role'] = squad_data['Player Type'].apply(self.standardize_role)
            elif 'role' in squad_data.columns:
                squad_data['role'] = squad_data['role'].apply(self.standardize_role)
            else:
                raise ValueError("No role column found in squad data")
                
            # Validate roles
            role_counts = squad_data['role'].value_counts().to_dict()
            print(f"Available players by role: {role_counts}")
            
            # Get adjusted role requirements based on available players
            role_requirements = self.role_mapper.get_adjusted_role_requirements(role_counts)
            print(f"Adjusted role requirements: {role_requirements}")
            
            # Create binary variables for each player
            player_vars = {}
            for idx, player in squad_data.iterrows():
                player_vars[idx] = {
                    'name': player['Player Name'],
                    'role': player['role'],
                    'credits': float(player['Credits']),
                    'points': float(player['predicted_points']),
                    'team': player['Team']
                }
                
            return {
                'players': player_vars,
                'role_requirements': role_requirements,
                'max_credits': 100,
                'max_players': 11
            }
            
        except Exception as e:
            print(f"Error creating optimization problem: {str(e)}")
            traceback.print_exc()
            return None
            
    def solve_optimization_problem(self, problem):
        """Solve the team optimization problem"""
        try:
            if problem is None:
                return None
                
            # Get players and role requirements
            players = problem['players']
            role_requirements = problem['role_requirements']
            max_credits = problem['max_credits']
            max_players = problem['max_players']
            
            # Try using PuLP solver
            try:
                # Create optimization model
                model = LpProblem("Dream11_Team_Selection", LpMaximize)
                
                # Create binary variables for each player
                player_vars = LpVariable.dicts("player",
                                             list(players.keys()),
                                             cat='Binary')
                
                # Objective: Maximize predicted points
                model += lpSum([players[p]['points'] * player_vars[p] for p in players])
                
                # Constraints
                # 1. Total players = 11
                model += lpSum([player_vars[p] for p in players]) == 11
                
                # 2. Max credits = 100
                model += lpSum([players[p]['credits'] * player_vars[p] for p in players]) <= max_credits
                
                # 3. Role constraints
                for role, (min_players, max_players) in role_requirements.items():
                    role_indices = [p for p in players if players[p]['role'] == role]
                    if role_indices:
                        if min_players > 0:
                            model += lpSum([player_vars[p] for p in role_indices]) >= min_players
                        model += lpSum([player_vars[p] for p in role_indices]) <= max_players
                
                # 4. Team balance constraints - ensure at least 4 players from each team
                # First, identify the teams in the data
                teams = set(players[p]['team'] for p in players)
                
                # Only enforce team balance if we have more than one team
                if len(teams) > 1:
                    for team in teams:
                        team_indices = [p for p in players if players[p]['team'] == team]
                        # Ensure at least 4 players from each team
                        model += lpSum([player_vars[p] for p in team_indices]) >= 4
                
                # Solve the problem
                model.solve(PULP_CBC_CMD(msg=True))
                
                if LpStatus[model.status] == 'Optimal':
                    # Get selected players
                    selected_team = []
                    for p in players:
                        if player_vars[p].value() > 0.5:
                            selected_team.append(players[p])
                    
                    # Check team balance in selected players
                    team_counts = {}
                    for player in selected_team:
                        team = player['team']
                        team_counts[team] = team_counts.get(team, 0) + 1
                    
                    print(f"Team distribution in selected players: {team_counts}")
                    
                    return selected_team
                else:
                    print(f"No optimal solution found. Status: {LpStatus[model.status]}")
                    return None
                    
            except Exception as e:
                print(f"Error using PuLP solver: {str(e)}")
                traceback.print_exc()
                
                # Fallback to simple greedy algorithm with team balance
                return self._greedy_team_selection_with_balance(problem)
                
        except Exception as e:
            print(f"Error solving optimization problem: {str(e)}")
            traceback.print_exc()
            return None
    
    def _greedy_team_selection_with_balance(self, problem):
        """Simple greedy algorithm for team selection with team balance"""
        try:
            players = problem['players']
            role_requirements = problem['role_requirements']
            max_credits = problem['max_credits']
            max_team_size = problem['max_players']
            
            # Determine unique teams
            teams = set(player['team'] for _, player in players.items())
            print(f"Teams in data: {teams}")
            
            # Count available players by role and team
            available_players_by_role = {}
            available_players_by_team = {}
            
            for player_id, player in players.items():
                # Group by role
                role = player['role']
                if role not in available_players_by_role:
                    available_players_by_role[role] = []
                available_players_by_role[role].append(player)
                
                # Group by team
                team = player['team']
                if team not in available_players_by_team:
                    available_players_by_team[team] = []
                available_players_by_team[team].append(player)
            
            # Sort players by predicted points within each role and team
            for role in available_players_by_role:
                available_players_by_role[role] = sorted(
                    available_players_by_role[role],
                    key=lambda x: x['points'],
                    reverse=True
                )
                
            for team in available_players_by_team:
                available_players_by_team[team] = sorted(
                    available_players_by_team[team],
                    key=lambda x: x['points'],
                    reverse=True
                )
            
            # Initialize selected team
            selected_team = []
            selected_credits = 0
            role_counts = {role: 0 for role in role_requirements}
            team_counts = {team: 0 for team in teams}
            
            # First, ensure we have the minimum required players from each role
            for role, (min_req, _) in role_requirements.items():
                if role in available_players_by_role and min_req > 0:
                    # Get the top min_req players for this role
                    top_players = available_players_by_role[role][:min_req]
                    for player in top_players:
                        if selected_credits + player['credits'] <= max_credits:
                            selected_team.append(player)
                            selected_credits += player['credits']
                            role_counts[role] += 1
                            team_counts[player['team']] += 1
                            
                            # Remove from available players
                            available_players_by_role[role].remove(player)
                            if player in available_players_by_team[player['team']]:
                                available_players_by_team[player['team']].remove(player)
            
            # Next, ensure we have at least 4 players from each team
            min_team_players = 4
            for team in teams:
                # If we don't have enough players from this team yet
                if team_counts[team] < min_team_players:
                    # Get top players from this team
                    team_players = available_players_by_team[team]
                    needed = min_team_players - team_counts[team]
                    
                    for player in team_players[:needed]:
                        if len(selected_team) < max_team_size and selected_credits + player['credits'] <= max_credits:
                            selected_team.append(player)
                            selected_credits += player['credits']
                            role_counts[player['role']] += 1
                            team_counts[player['team']] += 1
                            
                            # Remove from available players
                            if player in available_players_by_role[player['role']]:
                                available_players_by_role[player['role']].remove(player)
                            available_players_by_team[player['team']].remove(player)
            
            # Create a flat list of all remaining players sorted by points
            remaining_players = []
            for role in available_players_by_role:
                remaining_players.extend(available_players_by_role[role])
            remaining_players.sort(key=lambda x: x['points'], reverse=True)
            
            # Fill the rest of the team with the best players
            while len(selected_team) < max_team_size and remaining_players:
                # Get next best player
                player = remaining_players[0]
                
                # Check if adding this player would exceed constraints
                role_max = role_requirements[player['role']][1]
                if role_counts[player['role']] < role_max and selected_credits + player['credits'] <= max_credits:
                    selected_team.append(player)
                    selected_credits += player['credits']
                    role_counts[player['role']] += 1
                    team_counts[player['team']] += 1
                
                # Remove this player from consideration
                remaining_players.pop(0)
            
            print(f"Selected main team with {len(selected_team)} players")
            print(f"Team distribution: {team_counts}")
            print(f"Role distribution: {role_counts}")
            
            return selected_team
            
        except Exception as e:
            print(f"Error in greedy team selection: {str(e)}")
            traceback.print_exc()
            return None
    
    def select_impact_players(self, players, main_team):
        """Select 4 impact players (substitutes) from available players"""
        try:
            print("Selecting impact players...")
            
            # Ensure we have the correct column names
            player_id_col = 'name'  # Use 'name' as the player identifier column
            
            # Filter out players already in main team
            available_players = players[~players[player_id_col].isin(main_team[player_id_col])].copy()
            
            if len(available_players) == 0:
                print("No available players for impact selection")
                return pd.DataFrame()
            
            # Get role distribution
            role_counts = available_players['role'].value_counts()
            print(f"Available players by role: {role_counts.to_dict()}")
            
            # Select one player from each role if possible
            impact_players = []
            roles = ['WK', 'BAT', 'AR', 'BOWL']
            
            for role in roles:
                role_players = available_players[available_players['role'] == role]
                if len(role_players) > 0:
                    # Sort by predicted points
                    role_players = role_players.sort_values('predicted_points', ascending=False)
                    # Take the best player
                    impact_players.append(role_players.iloc[0])
                    # Remove from available players
                    available_players = available_players[~available_players[player_id_col].isin([role_players.iloc[0][player_id_col]])]
                    print(f"Selected {role_players.iloc[0][player_id_col]} as impact player for role {role}")
                else:
                    print(f"No players available for role {role}")
            
            # If we don't have 4 players yet, add more from any role
            remaining = 4 - len(impact_players)
            if remaining > 0 and len(available_players) > 0:
                print(f"Need {remaining} more impact players. Selecting from any role...")
                available_players = available_players.sort_values('predicted_points', ascending=False)
                for i in range(min(remaining, len(available_players))):
                    impact_players.append(available_players.iloc[i])
                    print(f"Selected additional impact player {available_players.iloc[i][player_id_col]} from role {available_players.iloc[i]['role']}")
            
            # Create DataFrame for return
            if impact_players:
                impact_df = pd.DataFrame(impact_players)
                print(f"Selected {len(impact_df)} impact players")
                return impact_df
            else:
                print("No impact players selected")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error selecting impact players: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def optimize_team(self, players, team1, team2, role_requirements=None):
        """Optimize the Dream11 team selection and impact players"""
        try:
            # Set selected teams
            self.selected_teams = [team1, team2]
            
            # Create a new optimization model
            self.model = Model("Dream11")
            
            # Use default role requirements if none provided
            if role_requirements is None:
                role_requirements = self.role_mapper.get_adjusted_role_requirements(players['role'].value_counts().to_dict())
                
            # Create the optimization problem and solve
            player_vars, available_players = self.create_optimization_problem(players)
            
            # Optimize
            self.model.optimize()
            
            # Check if optimal solution found
            if self.model.status == GRB.OPTIMAL:
                # Extract the selected team
                main_team = available_players[
                    [player_vars[idx].x > 0.5 for idx in available_players.index]
                ]
                
                if len(main_team) != 11:
                    print(f"Warning: Optimization returned {len(main_team)} players instead of 11")
                
                print(f"Selected main team with {len(main_team)} players")
                
                # Select impact players
                impact_players = self.select_impact_players(players, main_team)
                
                # Return combined result
                result = {
                    'main_team': main_team,
                    'impact_players': impact_players
                }
                
                return result
            else:
                print(f"Optimization failed with status {self.model.status}")
                return None
                
        except Exception as e:
            print(f"Error in team optimization: {e}")
            traceback.print_exc()
            
            # Try with PuLP as a fallback
            try:
                print("Trying fallback optimization with PuLP...")
                return self.optimize_team_with_pulp(players, team1, team2)
            except Exception as fallback_error:
                print(f"Fallback optimization failed: {fallback_error}")
                return None
                
    def optimize_team_with_pulp(self, players, team1, team2):
        """Fallback method using PuLP instead of Gurobi"""
        try:
            # Create PuLP problem
            prob = LpProblem("Dream11", LpMaximize)
            
            # Filter players for the selected teams
            team_col = 'Team' if 'Team' in players.columns else 'team'
            available_players = players[players[team_col].isin([team1, team2])].copy()
            
            if len(available_players) == 0:
                raise ValueError(f"No players found for teams {team1} and {team2}")
            
            # Standardize roles if needed
            if 'standardized_role' not in available_players.columns:
                role_col = 'Player Type' if 'Player Type' in available_players.columns else 'role'
                available_players['standardized_role'] = available_players[role_col].apply(self.standardize_role)
            
            # Create variables
            player_vars = LpVariable.dicts("player", 
                                          available_players.index, 
                                          cat=LpBinary)
            
            # Set objective to maximize predicted points
            points_col = 'predicted_points' if 'predicted_points' in available_players.columns else 'fantasy_points'
            prob += lpSum([player_vars[i] * available_players.loc[i, points_col] for i in available_players.index])
            
            # Total players constraint
            prob += lpSum([player_vars[i] for i in available_players.index]) == 11
            
            # Role constraints
            role_requirements = self.role_mapper.get_adjusted_role_requirements(available_players['standardized_role'].value_counts().to_dict())
            for role, (min_players, max_players) in role_requirements.items():
                role_indices = available_players[available_players['standardized_role'] == role].index
                if len(role_indices) > 0:
                    prob += lpSum([player_vars[i] for i in role_indices]) >= min_players
                    prob += lpSum([player_vars[i] for i in role_indices]) <= max_players
            
            # Credits constraint
            credits_col = 'Credits' if 'Credits' in available_players.columns else 'credits'
            if credits_col in available_players.columns:
                prob += lpSum([player_vars[i] * available_players.loc[i, credits_col] 
                           for i in available_players.index]) <= 100
            
            # Team balance constraints
            team1_indices = available_players[available_players[team_col] == team1].index
            team2_indices = available_players[available_players[team_col] == team2].index
            
            if len(team1_indices) > 0:
                prob += lpSum([player_vars[i] for i in team1_indices]) >= 4
            if len(team2_indices) > 0:
                prob += lpSum([player_vars[i] for i in team2_indices]) >= 4
            
            # Solve the problem
            prob.solve(PULP_CBC_CMD(msg=True))
            
            if prob.status == LpStatusOptimal:
                # Extract selected team
                selected_indices = [i for i in available_players.index if player_vars[i].value() > 0.5]
                main_team = available_players.loc[selected_indices].copy()
                
                print(f"PuLP optimization successful, selected {len(main_team)} players")
                
                # Select impact players
                impact_players = self.select_impact_players(players, main_team)
                
                # Return combined result
                result = {
                    'main_team': main_team,
                    'impact_players': impact_players
                }
                
                return result
            else:
                print(f"PuLP optimization failed with status {LpStatus[prob.status]}")
                return None
                
        except Exception as e:
            print(f"Error in PuLP optimization: {e}")
            traceback.print_exc()
            return None
    
    def create_backup_team(self, players: pd.DataFrame, predicted_points: np.ndarray, 
                          main_team: List[str], max_backups: int = 4) -> List[str]:
        """Create backup team for each role"""
        backup_team = []
        role_backups = {role: [] for role in self.role_mapper.get_adjusted_role_requirements(players['role'].value_counts().to_dict()).keys()}
        
        # Sort players by predicted points within each role
        for role in self.role_mapper.get_adjusted_role_requirements(players['role'].value_counts().to_dict()).keys():
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