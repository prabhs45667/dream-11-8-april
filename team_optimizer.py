import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pulp import *
import traceback
from role_mapper import RoleMapper
import random

class TeamOptimizer:
    def __init__(self):
        """Initialize the team optimizer"""
        self.selected_teams = []
        self.role_mapper = RoleMapper()
        
        # Role mapping dictionary for standardizing roles
        self.role_mapping = {
            'WK': 'WK', 'WICKET-KEEPER': 'WK', 'Wicket Keeper': 'WK', 'KEEPER': 'WK',
            'BAT': 'BAT', 'BATSMAN': 'BAT', 'Batsman': 'BAT', 'BATTER': 'BAT',
            'AR': 'AR', 'ALL-ROUNDER': 'AR', 'All Rounder': 'AR', 'ALL': 'AR', 'ALL ROUNDER': 'AR',
            'BOWL': 'BOWL', 'BOWLER': 'BOWL', 'Bowler': 'BOWL'
        }
        self.default_role_requirements = {
            'WK': {'min': 1, 'max': 4},
            'BAT': {'min': 3, 'max': 6},
            'AR': {'min': 1, 'max': 4},
            'BOWL': {'min': 3, 'max': 6}
        }
        
    def standardize_role(self, role):
        """Standardize player role to one of: WK, BAT, AR, BOWL"""
        return self.role_mapper.standardize_role(role)
        
    def create_optimization_problem(self, squad_data):
        """Create optimization problem for team selection"""
        try:
            # Create a copy to avoid modifying the original
            squad_data = squad_data.copy()
            
            # Ensure predicted_points exist
            if 'predicted_points' not in squad_data.columns:
                print("WARNING: predicted_points column not found, creating based on Credits")
                if 'Credits' in squad_data.columns:
                    squad_data['predicted_points'] = squad_data['Credits'].astype(float) * 100
                elif 'credits' in squad_data.columns:
                    squad_data['predicted_points'] = squad_data['credits'].astype(float) * 100
                else:
                    # Default value if no Credits column
                    squad_data['predicted_points'] = 500
            
            # Cap player points to a realistic maximum (150)
            if 'predicted_points' in squad_data.columns:
                print("Capping player points to realistic values (max 150)")
                # Find the 90th percentile of predicted points
                p90 = squad_data['predicted_points'].quantile(0.9)
                # Apply point capping: min of original points, 150, or 2*p90
                max_allowed = min(150, 2 * p90)
                squad_data['original_points'] = squad_data['predicted_points'].copy()
                squad_data['predicted_points'] = squad_data['predicted_points'].clip(upper=max_allowed)
                # Log scaling to compress extremely high values
                high_values = squad_data['predicted_points'] > 100
                if high_values.any():
                    squad_data.loc[high_values, 'predicted_points'] = 100 + 10 * np.log(squad_data.loc[high_values, 'predicted_points'] - 99)
                print(f"Points capped at {max_allowed}. Mean: {squad_data['predicted_points'].mean():.2f}, Max: {squad_data['predicted_points'].max():.2f}")
                # Ensure total points are less than 1000
                if squad_data['predicted_points'].sum() > 1000:
                    scaling_factor = 1000 / squad_data['predicted_points'].sum()
                    squad_data['predicted_points'] = squad_data['predicted_points'] * scaling_factor
                    print(f"Scaling player points by {scaling_factor:.2f} to keep total under 1000")
            
            # Ensure roles are standardized
            if 'Player Type' in squad_data.columns and 'role' not in squad_data.columns:
                squad_data['role'] = squad_data['Player Type'].apply(self.standardize_role)
            elif 'role' in squad_data.columns:
                squad_data['role'] = squad_data['role'].apply(self.standardize_role)
            else:
                print("WARNING: No role column found in squad data. Adding default roles")
                squad_data['role'] = 'BAT'  # Default role
                
            # Validate roles
            role_counts = squad_data['role'].value_counts().to_dict()
            print(f"Available players by role: {role_counts}")
            
            # Get adjusted role requirements based on available players
            role_requirements = self.role_mapper.get_adjusted_role_requirements(role_counts)
            print(f"Adjusted role requirements: {role_requirements}")
            
            # Create binary variables for each player
            player_vars = {}
            for i, player in squad_data.iterrows():
                player_vars[i] = LpVariable(f"player_{i}", cat='Binary')
                
            # Create optimization problem (maximize predicted points)
            prob = LpProblem("Dream11_Team_Selection", LpMaximize)
            
            # Objective function: maximize total predicted points
            prob += lpSum([player_vars[i] * squad_data.loc[i, 'predicted_points'] for i in player_vars])
            
            # Constraint: exactly 11 players
            prob += lpSum([player_vars[i] for i in player_vars]) == 11
            
            # Constraint: role requirements
            for role, (min_players, max_players) in role_requirements.items():
                role_indices = squad_data[squad_data['role'] == role].index
                if len(role_indices) > 0:
                    prob += lpSum([player_vars[i] for i in role_indices]) >= min_players
                    prob += lpSum([player_vars[i] for i in role_indices]) <= max_players
            
            # Constraint: credits limit (100)
            credits_col = None
            if 'Credits' in squad_data.columns:
                credits_col = 'Credits'
            elif 'credits' in squad_data.columns:
                credits_col = 'credits'
                
            if credits_col is not None:
                prob += lpSum([player_vars[i] * squad_data.loc[i, credits_col] for i in player_vars]) <= 100
            else:
                print("WARNING: No credits column found. Skipping credits constraint")
            
            # NEW CONSTRAINT: maximum 4 players from any team
            team_col = None
            if 'team' in squad_data.columns:
                team_col = 'team'
            elif 'Team' in squad_data.columns:
                team_col = 'Team'
            elif 'team_code' in squad_data.columns:
                team_col = 'team_code'
                
            if team_col:
                teams = squad_data[team_col].unique()
                for team in teams:
                    team_indices = squad_data[squad_data[team_col] == team].index
                    prob += lpSum([player_vars[i] for i in team_indices]) <= 4
                    print(f"Added constraint: max 4 players from team {team}")
            else:
                print("Warning: No team column found for team limits constraint")
            
            # NEW CONSTRAINT: Minimum 2 death overs specialists
            # Add death over specialist designation
            death_overs_specialists = []
            # Look for known death over bowlers or batsmen who perform well in final overs
            player_col = None
            if 'Player Name' in squad_data.columns:
                player_col = 'Player Name'
            elif 'name' in squad_data.columns:
                player_col = 'name'
            elif 'player' in squad_data.columns:
                player_col = 'player'
                
            if player_col:
                # List of known death over specialists
                death_specialists = [
                    'Jasprit Bumrah', 'Matheesha Pathirana', 'Arshdeep Singh', 'Trent Boult',
                    'Mohammad Shami', 'Kagiso Rabada', 'Andre Russell', 'Hardik Pandya',
                    'Bhuvneshwar Kumar', 'Harshal Patel', 'Kieron Pollard', 'MS Dhoni',
                    'Suryakumar Yadav', 'Rishabh Pant', 'Dinesh Karthik', 'Rashid Khan'
                ]
                
                # Find indices of death over specialists
                for player_name in death_specialists:
                    try:
                        matches = squad_data[squad_data[player_col].str.contains(player_name, case=False, na=False, regex=False)]
                        if not matches.empty:
                            death_overs_specialists.extend(matches.index.tolist())
                    except Exception as e:
                        print(f"Error finding specialist {player_name}: {str(e)}")
                
                if death_overs_specialists:
                    # Need at least 2 death overs specialists
                    prob += lpSum([player_vars[i] for i in death_overs_specialists]) >= 2
                    print(f"Added constraint: minimum 2 death overs specialists from {len(death_overs_specialists)} options")
                else:
                    print("Warning: No death overs specialists identified in the squad data")
            
            return prob, player_vars, squad_data
            
        except Exception as e:
            print(f"Error creating optimization problem: {str(e)}")
            traceback.print_exc()
            return None, None, squad_data
            
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
    
    def select_impact_players(self, all_players_df, selected_team_df, num_substitutes=4):
        """
        Select impact substitute players based on the main team.
        
        Args:
            all_players_df (pd.DataFrame): DataFrame containing all available players
            selected_team_df (pd.DataFrame): DataFrame containing the selected main team
            num_substitutes (int): Number of substitute players to select (default: 4)
            
        Returns:
            pd.DataFrame: DataFrame containing the selected impact substitute players
        """
        try:
            # Ensure columns are standardized
            if all_players_df is None or all_players_df.empty:
                return pd.DataFrame()
            
            # Make a copy to avoid modifying original
            all_players_df = all_players_df.copy()
            
            # Standardize column names
            if 'Player Name' in all_players_df.columns:
                all_players_df = all_players_df.rename(columns={
                    'Player Name': 'name',
                    'Player Type': 'role',
                    'Team': 'team',
                    'Credits': 'credits'
                })
            
            # Convert selected_team_df to standardized format if needed
            if selected_team_df is None or selected_team_df.empty:
                return pd.DataFrame()
                
            # Make a copy of selected_team_df
            selected_team_df = selected_team_df.copy()
            
            if 'player' in selected_team_df.columns and 'name' not in selected_team_df.columns:
                selected_team_df = selected_team_df.rename(columns={'player': 'name'})
            
            # Get the list of already selected players to exclude them
            selected_players = set()
            if 'name' in selected_team_df.columns:
                selected_players = set(selected_team_df['name'].str.lower().tolist())
            
            # Reset index to ensure no duplicate indices
            all_players_df = all_players_df.reset_index(drop=True)
            
            # Filter out already selected players
            available_players = pd.DataFrame()
            if 'name' in all_players_df.columns:
                # Use mask to filter rows to avoid reindexing errors
                mask = ~all_players_df['name'].str.lower().isin(selected_players)
                available_players = all_players_df[mask].copy().reset_index(drop=True)
            else:
                available_players = all_players_df.copy()
            
            if available_players.empty:
                return pd.DataFrame()
            
            # Standardize roles in available_players
            if 'role' in available_players.columns:
                # Use a safer approach without applying function to Series
                roles = []
                for r in available_players['role']:
                    if isinstance(r, str):
                        roles.append(self.standardize_role(r))
                    else:
                        roles.append('BAT')
                available_players['role'] = roles
            
            # We want to select one player for each role (WK, BAT, AR, BOWL)
            roles_to_select = ['WK', 'BAT', 'AR', 'BOWL']
            impact_players = []
            
            # Get the teams to ensure balance
            team_col = 'team' if 'team' in selected_team_df.columns else 'Team'
            teams = []
            if team_col in selected_team_df.columns:
                teams = selected_team_df[team_col].unique().tolist()
            
            # Limit to max 2 players per team
            team_count = {team: 0 for team in teams}
            
            # Try to select one player from each role
            for role in roles_to_select:
                if 'role' not in available_players.columns:
                    continue
                
                # Safer approach: Create mask and use it to filter
                role_players_indices = []
                for idx, r in enumerate(available_players['role']):
                    if r == role:
                        role_players_indices.append(idx)
                
                if role_players_indices:
                    role_players = available_players.iloc[role_players_indices].copy()
                    
                    if not role_players.empty:
                        # Sort by predicted_points or credits
                        if 'predicted_points' in role_players.columns:
                            role_players = role_players.sort_values('predicted_points', ascending=False)
                        elif 'fantasy_points' in role_players.columns:
                            role_players = role_players.sort_values('fantasy_points', ascending=False)
                        else:
                            # If no points available, sort by credits with some randomness
                            role_players['random_factor'] = np.random.rand(len(role_players))
                            role_players = role_players.sort_values(['credits', 'random_factor'], ascending=[False, True])
                        
                        # Prioritize players from less represented teams
                        team_col = 'team' if 'team' in role_players.columns else 'Team'
                        selected_player = None
                        
                        for idx, player in role_players.iterrows():
                            player_team = None
                            if team_col in role_players.columns:
                                player_team = player[team_col]
                                
                            if player_team and team_count.get(player_team, 0) < 2:  # Limit to max 2 per team
                                selected_player = player.to_dict()
                                team_count[player_team] = team_count.get(player_team, 0) + 1
                                impact_players.append(selected_player)
                                break
            
            # If we don't have enough players, add more from any role
            remaining_slots = num_substitutes - len(impact_players)
            if remaining_slots > 0:
                # Build a list of names already selected as impact players
                selected_impact_names = [p.get('name', '') for p in impact_players]
                
                # Create a copy of available players excluding already selected impact players
                remaining_available = available_players.copy()
                name_col = 'name' if 'name' in remaining_available.columns else 'Player Name'
                
                if name_col in remaining_available.columns:
                    # Use list comprehension to create mask safely
                    mask_indices = []
                    for idx, name in enumerate(remaining_available[name_col]):
                        if name not in selected_impact_names:
                            mask_indices.append(idx)
                    
                    if mask_indices:
                        remaining_available = remaining_available.iloc[mask_indices].copy().reset_index(drop=True)
                
                if not remaining_available.empty:
                    # Add a random factor for diversity
                    remaining_available.loc[:, 'random'] = np.random.rand(len(remaining_available))
                    
                    # Sort by a combination of points and random factor
                    if 'predicted_points' in remaining_available.columns:
                        remaining_available = remaining_available.sort_values(['predicted_points', 'random'], ascending=[False, True])
                    elif 'fantasy_points' in remaining_available.columns:
                        remaining_available = remaining_available.sort_values(['fantasy_points', 'random'], ascending=[False, True])
                    else:
                        credits_col = 'credits' if 'credits' in remaining_available.columns else 'Credits'
                        if credits_col in remaining_available.columns:
                            remaining_available = remaining_available.sort_values([credits_col, 'random'], ascending=[False, True])
                        else:
                            remaining_available = remaining_available.sort_values('random')
                    
                    # Add remaining players considering team balance
                    team_col = 'team' if 'team' in remaining_available.columns else 'Team'
                    for idx, player in remaining_available.iterrows():
                        if len(impact_players) >= num_substitutes:
                            break
                        
                        player_team = None
                        if team_col in remaining_available.columns:
                            player_team = player[team_col]
                            
                        if player_team and team_count.get(player_team, 0) < 2:  # Maintain team limit
                            impact_players.append(player.to_dict())
                            team_count[player_team] = team_count.get(player_team, 0) + 1
            
            if not impact_players:
                return pd.DataFrame()
                
            # Convert to DataFrame from list of dictionaries
            impact_df = pd.DataFrame(impact_players)
            
            # If no predicted_points column, create a synthetic one based on credits and role
            if 'predicted_points' not in impact_df.columns and len(impact_df) > 0:
                # Assign points proportional to credits with some randomness
                base_points = {
                    'WK': 30, 'BAT': 35, 'AR': 40, 'BOWL': 30
                }
                role_col = 'role' if 'role' in impact_df.columns else 'Player Type'
                credits_col = 'credits' if 'credits' in impact_df.columns else 'Credits'
                
                if role_col in impact_df.columns and credits_col in impact_df.columns:
                    points = []
                    for idx, row in impact_df.iterrows():
                        role = row[role_col]
                        credits = float(row[credits_col])
                        base = base_points.get(role, 35)
                        points.append((credits * 4 + base) * (0.9 + 0.2 * random.random()))
                    impact_df['predicted_points'] = points
                else:
                    # Default points if columns are missing
                    impact_df['predicted_points'] = [75 * random.uniform(0.8, 1.2) for _ in range(len(impact_df))]
            
            # Limit to requested number of substitutes
            return impact_df.head(num_substitutes)
            
        except Exception as e:
            print(f"Error selecting impact players: {str(e)}")
            traceback.print_exc()
            # Use a simpler approach as fallback
            return self._select_impact_players_fallback(all_players_df, selected_team_df, num_substitutes)
    
    def _select_impact_players_fallback(self, all_players_df, selected_team_df, num_substitutes=4):
        """Fallback method for selecting impact players when the main method fails"""
        try:
            # Make a copy to avoid modifying original
            if all_players_df is None or all_players_df.empty:
                return pd.DataFrame()
                
            all_players_df = all_players_df.copy().reset_index(drop=True)
            
            # Basic standardization of column names
            if 'Player Name' in all_players_df.columns and 'name' not in all_players_df.columns:
                all_players_df['name'] = all_players_df['Player Name']
            
            if 'Player Type' in all_players_df.columns and 'role' not in all_players_df.columns:
                all_players_df['role'] = all_players_df['Player Type']
                
            if 'Team' in all_players_df.columns and 'team' not in all_players_df.columns:
                all_players_df['team'] = all_players_df['Team']
                
            if 'Credits' in all_players_df.columns and 'credits' not in all_players_df.columns:
                all_players_df['credits'] = all_players_df['Credits']
                
            if 'predicted_points' not in all_players_df.columns and 'Credits' in all_players_df.columns:
                all_players_df['predicted_points'] = all_players_df['Credits'] * 10  # Simple proxy
            
            # Get names of already selected players
            selected_names = []
            if selected_team_df is not None and not selected_team_df.empty:
                selected_team_df = selected_team_df.copy()
                if 'player' in selected_team_df.columns:
                    selected_names = selected_team_df['player'].tolist()
                elif 'name' in selected_team_df.columns:
                    selected_names = selected_team_df['name'].tolist()
                elif 'Player Name' in selected_team_df.columns:
                    selected_names = selected_team_df['Player Name'].tolist()
            
            # Filter out already selected players
            available = all_players_df.copy()
            if selected_names:
                # Create a list to store indices of rows to keep
                mask_indices = []
                name_col = None
                
                # Find the correct name column
                if 'name' in available.columns:
                    name_col = 'name'
                elif 'Player Name' in available.columns:
                    name_col = 'Player Name'
                
                if name_col:
                    for idx, name in enumerate(available[name_col]):
                        if name not in selected_names:
                            mask_indices.append(idx)
                    
                    if mask_indices:
                        available = available.iloc[mask_indices].copy()
                    
            if available.empty:
                return pd.DataFrame()
            
            # Reset index to ensure no duplicate indices
            available = available.reset_index(drop=True)
            
            # Add random column for diversity in selection
            available.loc[:, 'random'] = np.random.rand(len(available))
            
            # Sort by predicted points and random factor
            if 'predicted_points' in available.columns:
                available = available.sort_values(['predicted_points', 'random'], ascending=[False, True])
            elif 'Credits' in available.columns:
                available = available.sort_values(['Credits', 'random'], ascending=[False, True])
            else:
                available = available.sort_values('random')
            
            # Get a player from each role if possible
            roles = ['WK', 'BAT', 'AR', 'BOWL']
            selected_players = []
            
            # First try to get one player from each role
            for role in roles:
                if len(selected_players) < num_substitutes:
                    # Create indices list for safer filtering
                    role_player_indices = []
                    
                    # Find correct role column
                    role_col = None
                    if 'role' in available.columns:
                        role_col = 'role'
                    elif 'Role' in available.columns:
                        role_col = 'Role'
                    elif 'Player Type' in available.columns:
                        role_col = 'Player Type'
                    
                    if role_col:
                        for idx, r in enumerate(available[role_col]):
                            if isinstance(r, str) and (r == role or r.lower() == role.lower()):
                                role_player_indices.append(idx)
                    
                    if role_player_indices:
                        # Get players with this role
                        role_players = available.iloc[role_player_indices].copy()
                        
                        if not role_players.empty:
                            # Get the first player
                            selected_player = role_players.iloc[0].to_dict()
                            selected_players.append(selected_player)
                            
                            # Remove selected player from available pool by index
                            player_idx = role_players.index[0]
                            available = available.drop(player_idx).reset_index(drop=True)
            
            # Fill remaining slots with best available players
            remaining_slots = num_substitutes - len(selected_players)
            if remaining_slots > 0 and not available.empty:
                for i in range(min(remaining_slots, len(available))):
                    if i < len(available):
                        selected_players.append(available.iloc[i].to_dict())
            
            # Convert list of players to DataFrame
            if not selected_players:
                return pd.DataFrame()
                
            impact_df = pd.DataFrame(selected_players)
            
            # Ensure we have predicted_points
            if 'predicted_points' not in impact_df.columns and len(impact_df) > 0:
                # Base points by role
                base_points = {'WK': 30, 'BAT': 35, 'AR': 40, 'BOWL': 30}
                # Get credits column - either 'credits' or 'Credits'
                credits_col = 'credits' if 'credits' in impact_df.columns else 'Credits'
                # Get role column - either 'role' or 'Role' 
                role_col = 'role' if 'role' in impact_df.columns else 'Role'
                
                if credits_col in impact_df.columns and role_col in impact_df.columns:
                    # Create points list manually
                    points = []
                    for _, row in impact_df.iterrows():
                        role = row[role_col]
                        base = base_points.get(role, 35)
                        credits = float(row[credits_col])
                        points.append((credits * 4 + base) * (0.9 + 0.2 * random.random()))
                    impact_df['predicted_points'] = points
                else:
                    # Default points if missing columns
                    impact_df['predicted_points'] = [75 * random.uniform(0.8, 1.2) for _ in range(len(impact_df))]
            
            # Return the impact players
            return impact_df.head(num_substitutes)
            
        except Exception as e:
            print(f"Error in fallback impact player selection: {str(e)}")
            traceback.print_exc()
            # Last resort fallback - return empty DataFrame
            return pd.DataFrame()
    
    def optimize_team(self, players, team1, team2, role_requirements=None):
        """Optimize team selection using PuLP"""
        try:
            print("Using TeamOptimizer for team selection...")
            
            # Create the optimization problem
            prob, player_vars, updated_players = self.create_optimization_problem(players)
            
            if prob is None:
                print("Failed to create optimization problem. Using fallback method.")
                return self._greedy_team_selection(players, team1, team2, role_requirements)
            
            # Solve the optimization problem
            prob.solve(PULP_CBC_CMD(msg=True))
            
            # Check if a solution was found
            if LpStatus[prob.status] != "Optimal":
                print(f"Optimization problem could not be solved optimally. Status: {LpStatus[prob.status]}")
                print("Using fallback method.")
                return self._greedy_team_selection(players, team1, team2, role_requirements)
            
            # Extract the selected team
            selected_indices = [i for i in player_vars if value(player_vars[i]) > 0.5]
            
            if not selected_indices:
                print("No players selected by optimizer. Using fallback method.")
                return self._greedy_team_selection(players, team1, team2, role_requirements)
            
            # Create dataframe of selected players
            selected_team = updated_players.loc[selected_indices].copy()
            
            print(f"Selected main team with {len(selected_team)} players")
            
            # Calculate team distribution
            if 'Team' in selected_team.columns:
                team_counts = selected_team['Team'].value_counts().to_dict()
                print(f"Team distribution in selected players: {team_counts}")
            elif 'team' in selected_team.columns:
                team_counts = selected_team['team'].value_counts().to_dict()
                print(f"Team distribution in selected players: {team_counts}")
                
            # Select captain and vice-captain
            final_team = self.select_captain_vice_captain(selected_team)
            
            # Format output in the expected structure
            team_output = {
                'players': final_team.reset_index().to_dict('records'),
                'total_points': final_team['predicted_points'].sum(),
                'total_credits': final_team['Credits'].sum() if 'Credits' in final_team else 0
            }
            
            return team_output
            
        except Exception as e:
            print(f"Error in team optimization: {str(e)}")
            traceback.print_exc()
            return self._greedy_team_selection(players, team1, team2, role_requirements)
    
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

    def select_captain_vice_captain(self, selected_team):
        """Select captain and vice-captain from the team"""
        try:
            # Ensure we have required columns
            if not isinstance(selected_team, pd.DataFrame) or selected_team.empty:
                print("WARNING: Cannot select captain and vice-captain from empty team")
                return selected_team
                
            # Ensure we have predicted_points column
            if 'predicted_points' not in selected_team.columns:
                print("WARNING: No predicted_points column, creating based on Credits")
                if 'Credits' in selected_team.columns:
                    selected_team['predicted_points'] = selected_team['Credits'].astype(float) * 100
                else:
                    # Default value if no Credits column
                    selected_team['predicted_points'] = 500
                    
            # Ensure we have role column
            if 'role' not in selected_team.columns:
                if 'Player Type' in selected_team.columns:
                    selected_team['role'] = selected_team['Player Type']
                else:
                    # Default roles if no role column
                    selected_team['role'] = 'BAT'
            
            # Sort by predicted points
            sorted_team = selected_team.sort_values('predicted_points', ascending=False)
            
            # Select top player as captain
            captain = sorted_team.iloc[0].copy()
            # Add (C) to role without using string methods that might fail
            captain_role = str(captain.get('role', ''))
            captain['role'] = captain_role + " (C)" if captain_role else "BAT (C)"
            captain['predicted_points'] *= 2  # Captain gets 2x points
            
            # Select second-best player as vice-captain from top 3
            # This ensures captain is among top 3 predicted players
            if len(sorted_team) >= 3:
                # Randomly choose from top 3 (excluding captain)
                top_three = sorted_team.iloc[1:3]
                vice_captain = top_three.sample(1).iloc[0].copy()
            else:
                # Fallback to second player if team has fewer than 3 players
                vice_captain = sorted_team.iloc[1].copy() if len(sorted_team) >= 2 else sorted_team.iloc[0].copy()
                
            # Add (VC) to role without using string methods that might fail
            vc_role = str(vice_captain.get('role', ''))
            vice_captain['role'] = vc_role + " (VC)" if vc_role else "BAT (VC)"
            vice_captain['predicted_points'] *= 1.5  # Vice-captain gets 1.5x points
            
            # Remove captain and vice-captain from the team
            captain_index = sorted_team.index.get_loc(captain.name)
            vc_index = sorted_team.index.get_loc(vice_captain.name)
            
            team_without_captain_vc = sorted_team.drop([captain.name, vice_captain.name])
            
            # Create final team with captain and vice-captain at the top
            final_team = pd.concat([
                pd.DataFrame([captain]), 
                pd.DataFrame([vice_captain]), 
                team_without_captain_vc
            ])
            
            return final_team
            
        except Exception as e:
            print(f"Error selecting captain and vice-captain: {str(e)}")
            traceback.print_exc()
            # Return the input team as fallback
            return selected_team

    def _greedy_team_selection(self, players, team1, team2, role_requirements=None):
        """Fallback method using greedy algorithm when optimization fails"""
        try:
            print("Using greedy algorithm for team selection")
            
            # Create a copy to avoid modifying the original
            squad_data = players.copy()
            
            # Ensure roles are standardized
            if 'Player Type' in squad_data.columns and 'role' not in squad_data.columns:
                squad_data['role'] = squad_data['Player Type'].apply(self.standardize_role)
            elif 'role' in squad_data.columns:
                squad_data['role'] = squad_data['role'].apply(self.standardize_role)
            
            # Make sure we have predicted_points
            if 'predicted_points' not in squad_data.columns:
                print("WARNING: predicted_points column not found, creating based on Credits")
                if 'Credits' in squad_data.columns:
                    squad_data['predicted_points'] = squad_data['Credits'].astype(float) * 100
                else:
                    # Default value if no Credits column
                    squad_data['predicted_points'] = 500
            
            # Ensure we have team column
            team_col = 'Team' if 'Team' in squad_data.columns else 'team'
            
            # Get role requirements if not provided
            if role_requirements is None:
                role_counts = squad_data['role'].value_counts().to_dict()
                role_requirements = self.role_mapper.get_adjusted_role_requirements(role_counts)
            
            # Prepare for selection
            selected_players = []
            credits_remaining = 100
            team1_count = 0
            team2_count = 0
            role_counts = {role: 0 for role in role_requirements}
            
            # Cap player points if they're too high
            if 'predicted_points' in squad_data.columns:
                max_allowed = min(150, squad_data['predicted_points'].max() * 0.9)
                squad_data['predicted_points'] = squad_data['predicted_points'].clip(upper=max_allowed)
            
            # First select death over specialists (at least 2)
            death_specialists = ['Jasprit Bumrah', 'Matheesha Pathirana', 'Arshdeep Singh', 'Trent Boult', 
                               'Mohammad Shami', 'Kagiso Rabada', 'Andre Russell', 'Hardik Pandya']
            
            specialists_selected = 0
            for name in death_specialists:
                # Safely check for name matches with proper error handling
                try:
                    if 'Player Name' in squad_data.columns:
                        matches = squad_data[squad_data['Player Name'].str.contains(name, case=False, na=False, regex=False)]
                    else:
                        matches = pd.DataFrame()  # Empty DataFrame if column doesn't exist
                except Exception as e:
                    print(f"Error matching specialist {name}: {str(e)}")
                    matches = pd.DataFrame()  # Empty DataFrame on error
                
                if not matches.empty and specialists_selected < 2:
                    player = matches.iloc[0]
                    if credits_remaining >= player['Credits']:
                        selected_players.append(player)
                        squad_data = squad_data.drop(player.name)
                        credits_remaining -= player['Credits']
                        role_counts[player['role']] += 1
                        
                        # Update team counts
                        if player[team_col] in [team1] + [t for t in squad_data[team_col].unique() if team1 in t]:
                            team1_count += 1
                        else:
                            team2_count += 1
                            
                        specialists_selected += 1
            
            # Get minimum players required from each role
            min_role_requirements = {role: req[0] for role, req in role_requirements.items()}
            
            # First, ensure minimum requirements for each role
            for role, min_required in min_role_requirements.items():
                # Adjust for already selected specialists
                if role_counts[role] >= min_required:
                    continue
                    
                # Calculate how many more players we need for this role
                needed = min_required - role_counts[role]
                
                # Get all players of this role, sorted by points
                # Ensure the predicted_points column exists, use Credits as fallback
                sort_col = 'predicted_points' if 'predicted_points' in squad_data.columns else 'Credits'
                role_players = squad_data[squad_data['role'] == role].sort_values(sort_col, ascending=False)
                
                # Select top players by points
                for _, player in role_players.iterrows():
                    if credits_remaining >= player['Credits'] and needed > 0:
                        # Check team balance
                        if player[team_col] in [team1] + [t for t in squad_data[team_col].unique() if team1 in t]:
                            if team1_count >= 7:  # Maximum 7 players from one team
                                continue
                            team1_count += 1
                        else:
                            if team2_count >= 7:  # Maximum 7 players from one team
                                continue
                            team2_count += 1
                            
                        selected_players.append(player)
                        squad_data = squad_data.drop(player.name)
                        credits_remaining -= player['Credits']
                        role_counts[player['role']] += 1
                        needed -= 1
                        
                        if len(selected_players) >= 11:
                            break
            
            # Fill remaining slots with best players by points
            remaining_slots = 11 - len(selected_players)
            if remaining_slots > 0:
                # Get max players allowed from each role
                max_role_requirements = {role: req[1] for role, req in role_requirements.items()}
                
                # Sort available players by points
                # Ensure the predicted_points column exists, use Credits as fallback
                sort_col = 'predicted_points' if 'predicted_points' in squad_data.columns else 'Credits'
                available_players = squad_data.sort_values(sort_col, ascending=False)
                
                for _, player in available_players.iterrows():
                    role = player['role']
                    
                    # Skip if we've reached the maximum for this role
                    if role_counts[role] >= max_role_requirements[role]:
                        continue
                        
                    # Check credits
                    if credits_remaining < player['Credits']:
                        continue
                        
                    # Check team balance
                    if player[team_col] in [team1] + [t for t in squad_data[team_col].unique() if team1 in t]:
                        if team1_count >= 7:  # Maximum 7 players from one team
                            continue
                        team1_count += 1
                    else:
                        if team2_count >= 7:  # Maximum 7 players from one team
                            continue
                        team2_count += 1
                        
                    # Add player
                    selected_players.append(player)
                    squad_data = squad_data.drop(player.name)
                    credits_remaining -= player['Credits']
                    role_counts[role] += 1
                    
                    if len(selected_players) >= 11:
                        break
            
            # Create DataFrame from selected players
            if len(selected_players) < 11:
                print(f"WARNING: Could only select {len(selected_players)} players within constraints")
                # Relax constraints and try again with remaining credits
                remaining_slots = 11 - len(selected_players)
                available_players = squad_data.sort_values('predicted_points', ascending=False)
                
                for _, player in available_players.iterrows():
                    if credits_remaining >= player['Credits'] and remaining_slots > 0:
                        selected_players.append(player)
                        credits_remaining -= player['Credits']
                        remaining_slots -= 1
                        
                        if remaining_slots == 0:
                            break
            
            final_team = pd.DataFrame(selected_players)
            
            # Select captain and vice-captain
            if not final_team.empty:
                final_team = self.select_captain_vice_captain(final_team)
            
            # Format output
            team_output = {
                'players': final_team.reset_index().to_dict('records'),
                'total_points': final_team['predicted_points'].sum(),
                'total_credits': final_team['Credits'].sum() if 'Credits' in final_team else 0
            }
            
            return team_output
            
        except Exception as e:
            print(f"Error in greedy team selection: {str(e)}")
            traceback.print_exc()
            
            # Emergency fallback - return any valid 11 players
            try:
                # Ensure we have a column to sort by
                sort_col = 'predicted_points' if 'predicted_points' in players.columns else 'Credits'
                
                # Add predicted_points if it doesn't exist
                if 'predicted_points' not in players.columns:
                    players['predicted_points'] = players['Credits'] * 10
                
                emergency_players = players.sort_values(sort_col, ascending=False).head(11)
                emergency_players = self.select_captain_vice_captain(emergency_players)
                
                return {
                    'players': emergency_players.reset_index().to_dict('records'),
                    'total_points': emergency_players['predicted_points'].sum(), 
                    'total_credits': emergency_players['Credits'].sum() if 'Credits' in emergency_players else 0
                }
            except Exception as emergency_error:
                print(f"Emergency fallback failed: {str(emergency_error)}")
                return {
                    'players': [],
                    'total_points': 0,
                    'total_credits': 0
                } 

    def manual_select_impact_players(self, selected_11_players, num_impact_players=4):
        """
        Fallback method to manually select impact players based on credits and role distribution
        """
        try:
            self.logger.info("Using fallback method to select impact players")
            
            # Filter out already selected players
            available = self.df[~self.df['Player_Name'].isin(selected_11_players['Player_Name'])]
            
            if available.empty:
                return pd.DataFrame()
            
            # Reset index to ensure no duplicate indices
            available = available.reset_index(drop=True)
            
            # Add random column for diversity in selection
            available = available.copy()  # Make a copy to avoid SettingWithCopyWarning
            available.loc[:, 'random'] = np.random.rand(len(available))
            
            # First try to select players by role with some randomness
            selected_players = []
            selected_player_names = []
            
            # Try to get one player from each role
            role_priorities = ['WK', 'BAT', 'AR', 'BOWL']
            
            # Apply standardize role to ensure consistent role representation
            available['Role'] = available['Role'].apply(self.standardize_role)
            
            # Ensure we don't have more than 2 players from the same team
            available_teams = available['Team'].unique()
            team_counts = {}
            
            for role in role_priorities:
                if len(selected_players) >= num_impact_players:
                    break
                    
                role_players = available[available['Role'] == role]
                
                if not role_players.empty:
                    # Sort by a combination of predicted points and randomness
                    role_players['combined_score'] = role_players['PredPoints'] * 0.7 + role_players['random'] * 0.3
                    role_players = role_players.sort_values('combined_score', ascending=False)
                    
                    # Get the top player not from already well-represented teams
                    for _, player in role_players.iterrows():
                        team = player['Team']
                        
                        # Initialize team count if not seen before
                        if team not in team_counts:
                            team_counts[team] = 0
                            
                        # Skip if we already have 2 players from this team
                        if team_counts[team] >= 2:
                            continue
                            
                        if player['Player_Name'] not in selected_player_names:
                            selected_players.append(player)
                            selected_player_names.append(player['Player_Name'])
                            team_counts[team] = team_counts.get(team, 0) + 1
                            break
            
            # If we need more players, fill in with best available regardless of role
            remaining_spots = num_impact_players - len(selected_players)
            
            if remaining_spots > 0:
                remaining_available = available[~available['Player_Name'].isin(selected_player_names)]
                
                if not remaining_available.empty:
                    # Sort by predicted points and some randomness
                    remaining_available['combined_score'] = remaining_available['PredPoints'] * 0.7 + remaining_available['random'] * 0.3
                    remaining_available = remaining_available.sort_values('combined_score', ascending=False)
                    
                    for _, player in remaining_available.iterrows():
                        if len(selected_players) >= num_impact_players:
                            break
                            
                        team = player['Team']
                        
                        # Skip if we already have 2 players from this team
                        if team_counts.get(team, 0) >= 2:
                            continue
                            
                        if player['Player_Name'] not in selected_player_names:
                            selected_players.append(player)
                            selected_player_names.append(player['Player_Name'])
                            team_counts[team] = team_counts.get(team, 0) + 1
            
            # Convert selected players to DataFrame
            if not selected_players:
                return pd.DataFrame()
                
            # Extract consistent lists from selected players for DataFrame creation
            player_names = [p['Player_Name'] for p in selected_players]
            teams = [p['Team'] for p in selected_players]
            roles = [p['Role'] for p in selected_players]
            credits = [p['Credits'] for p in selected_players]
            pred_points = [p['PredPoints'] for p in selected_players]
            
            # Create DataFrame with consistent lists
            impact_players = pd.DataFrame({
                'Player_Name': player_names,
                'Team': teams,
                'Role': roles,
                'Credits': credits,
                'PredPoints': pred_points
            })
            
            return impact_players
            
        except Exception as e:
            self.logger.error(f"Error in manual impact player selection: {e}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame() 