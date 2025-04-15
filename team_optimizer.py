import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pulp import *
import traceback
from role_mapper import RoleMapper
import random
import logging
import warnings

class TeamOptimizer:
    def __init__(self):
        """Initialize the team optimizer"""
        self.selected_teams = []
        self.role_mapper = RoleMapper()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings from optimization
        warnings.filterwarnings("ignore")
        
        # Initialize required variables
        self.df = None  # Will be set when data is loaded
        
        # Role mapping dictionary for standardizing roles
        self.role_mapping = {
            'WK': ['WK', 'Wicket Keeper', 'wicket-keeper', 'wicketkeeper', 'keeper'],
            'BAT': ['BAT', 'Batsman', 'batsman', 'batter'],
            'BOWL': ['BOWL', 'Bowler', 'bowler'],
            'AR': ['AR', 'All-Rounder', 'All Rounder', 'allrounder', 'all-rounder']
        }
        self.default_role_requirements = {
            'WK': {'min': 1, 'max': 4},
            'BAT': {'min': 3, 'max': 6},
            'AR': {'min': 1, 'max': 4},
            'BOWL': {'min': 3, 'max': 6}
        }
        
        # Default role standardization function
        self.standardize_role = lambda role: next((std for std, variants in self.role_mapping.items() 
                                                for variant in variants if str(role).lower() == variant.lower()), 'BAT')
    
    def standardize_credits_column(self, df):
        """
        Standardize the credits column name, ensuring it is lowercase 'credits'.
        
        Args:
            df (pd.DataFrame): The DataFrame to standardize
            
        Returns:
            pd.DataFrame: DataFrame with standardized credits column
            str: The name of the credits column used ('credits')
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df, None
            
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Check for either 'credits' or 'Credits' column
        credits_col = None
        if 'credits' in df.columns:
            credits_col = 'credits'
            # Make sure we also have 'Credits' for backward compatibility
            if 'Credits' not in df.columns:
                df['Credits'] = df['credits']
        elif 'Credits' in df.columns:
            # Copy to lowercase for consistency
            df['credits'] = df['Credits']
            credits_col = 'credits'
        
        return df, credits_col
        
    def standardize_role(self, role):
        """Standardize player role to one of: WK, BAT, AR, BOWL"""
        return self.role_mapper.standardize_role(role)
        
    def create_optimization_problem(self, squad_data):
        """Create optimization problem for team selection"""
        try:
            # Debug information to help troubleshoot
            print(f"create_optimization_problem received data of type: {type(squad_data)}")
            
            # Check if squad_data is a dictionary (problem format) and convert to DataFrame
            if isinstance(squad_data, dict):
                if 'players' in squad_data:
                    print("Converting problem dictionary to DataFrame for optimization")
                    players_dict = squad_data['players']
                    # Extract player data from the dictionary format
                    player_records = []
                    for player_id, player_data in players_dict.items():
                        player_record = player_data.copy()
                        player_record['player_id'] = player_id
                        player_records.append(player_record)
                    
                    # Create DataFrame from player records
                    squad_data = pd.DataFrame(player_records)
                    
                    # Ensure required columns exist
                    if 'points' in squad_data.columns and 'predicted_points' not in squad_data.columns:
                        squad_data['predicted_points'] = squad_data['points']
                    
                    if 'name' in squad_data.columns and 'Player Name' not in squad_data.columns:
                        squad_data['Player Name'] = squad_data['name']
                    
                    print(f"Converted {len(squad_data)} players to DataFrame format")
                else:
                    print("ERROR: Received a dictionary without 'players' key")
                    return None, None, pd.DataFrame()
            
            if not isinstance(squad_data, pd.DataFrame):
                print(f"ERROR: After conversion, squad_data is not a DataFrame, but {type(squad_data)}")
                return None, None, pd.DataFrame()
            
            # Create a copy to avoid modifying the original
            squad_data = squad_data.copy()
            
            # Standardize the credits column
            squad_data, credits_col = self.standardize_credits_column(squad_data)
            
            # Now it's safe to access .columns because we've verified it's a DataFrame
            # Ensure predicted_points exist
            if 'predicted_points' not in squad_data.columns:
                print("WARNING: predicted_points column not found, creating based on credits")
                if credits_col is not None:
                    # Use the standardized credits column
                    squad_data['predicted_points'] = squad_data[credits_col].astype(float) * 100
                else:
                    # Default value if no credits column
                    squad_data['predicted_points'] = 500
                    print("WARNING: No credits column found, using default predicted points")
            
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
            if credits_col is not None:
                prob += lpSum([player_vars[i] * squad_data.loc[i, credits_col] for i in player_vars]) <= 100
            else:
                print("WARNING: No credits column found. Skipping credits constraint")
            
            # NEW CONSTRAINT: maximum 7 players from any team
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
                    prob += lpSum([player_vars[i] for i in team_indices]) <= 7
                    print(f"Added constraint: max 7 players from team {team}")
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
                    # Only add this constraint if we have at least 2 specialists
                    if len(death_overs_specialists) >= 2:
                        prob += lpSum([player_vars[i] for i in death_overs_specialists]) >= 2
                        print(f"Added constraint: minimum 2 death overs specialists from {len(death_overs_specialists)} options")
                    else:
                        print(f"Only {len(death_overs_specialists)} death over specialists found, skipping this constraint")
                else:
                    print("Warning: No death overs specialists identified in the squad data")
            
            return prob, player_vars, squad_data
            
        except Exception as e:
            print(f"Error creating optimization problem: {str(e)}")
            traceback.print_exc()
            return None, None, pd.DataFrame()
            
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
                
                # 4. Team balance constraints (removed mandatory minimum per team)
                # First, identify the teams in the data
                teams = set(players[p]['team'] for p in players)
                
                # Only enforce maximum per team if we have more than one team
                if len(teams) > 1:
                    for team in teams:
                        team_indices = [p for p in players if players[p]['team'] == team]
                        # Maximum 7 players from any team
                        model += lpSum([player_vars[p] for p in team_indices]) <= 7
                
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
    
    def _greedy_team_selection(self, players, team1, team2, role_requirements=None):
        """Fallback method using greedy algorithm when optimization fails"""
        try:
            print("Using greedy algorithm for team selection")
            
            # Debug information to help troubleshoot
            print(f"_greedy_team_selection received data of type: {type(players)}")
            
            # If players is already a DataFrame, just use it
            if isinstance(players, pd.DataFrame):
                squad_data = players.copy()
                print(f"Using provided DataFrame with {len(squad_data)} rows")
            # Otherwise, check if it's a dictionary with 'players' key and convert
            elif isinstance(players, dict):
                if 'players' in players:
                    print("Converting problem dictionary to DataFrame for greedy selection")
                    players_dict = players['players']
                    # Extract player data from the dictionary format
                    player_records = []
                    for player_id, player_data in players_dict.items():
                        player_record = player_data.copy()
                        player_record['player_id'] = player_id
                        player_records.append(player_record)
                    
                    # Create DataFrame from player records
                    squad_data = pd.DataFrame(player_records)
                    
                    # Ensure required columns exist
                    if 'points' in squad_data.columns and 'predicted_points' not in squad_data.columns:
                        squad_data['predicted_points'] = squad_data['points']
                    
                    if 'name' in squad_data.columns and 'Player Name' not in squad_data.columns:
                        squad_data['Player Name'] = squad_data['name']
                    
                    # Extract role requirements if present
                    if role_requirements is None and 'role_requirements' in players:
                        role_requirements = players['role_requirements']
                    
                    print(f"Converted {len(squad_data)} players to DataFrame format")
                else:
                    print("ERROR: Received a dictionary without 'players' key")
                    # Return an empty result to avoid crash
                    return {
                        'players': [],
                        'total_points': 0,
                        'total_credits': 0
                    }
            else:
                print(f"ERROR: Unsupported data type {type(players)} for greedy selection")
                return {
                    'players': [],
                    'total_points': 0,
                    'total_credits': 0
                }
            
            if not isinstance(squad_data, pd.DataFrame):
                print(f"ERROR: After conversion, squad_data is not a DataFrame, but {type(squad_data)}")
                return {
                    'players': [],
                    'total_points': 0,
                    'total_credits': 0
                }
            
            # Create a copy to avoid modifying original
            squad_data = squad_data.copy()
            
            # Standardize the credits column
            squad_data, credits_col = self.standardize_credits_column(squad_data)
            
            # Ensure roles are standardized
            if 'Player Type' in squad_data.columns and 'role' not in squad_data.columns:
                squad_data['role'] = squad_data['Player Type'].apply(self.standardize_role)
            elif 'role' in squad_data.columns:
                squad_data['role'] = squad_data['role'].apply(self.standardize_role)
            
            # Make sure we have predicted_points
            if 'predicted_points' not in squad_data.columns:
                print("WARNING: predicted_points column not found, creating based on Credits")
                if credits_col is not None:
                    squad_data['predicted_points'] = squad_data[credits_col].astype(float) * 10
                elif 'Credits' in squad_data.columns:
                    squad_data['predicted_points'] = squad_data['Credits'].astype(float) * 10
                else:
                    # Default value if no Credits column
                    squad_data['predicted_points'] = 50
            
            # Ensure we have team column
            team_col = None
            if 'Team' in squad_data.columns:
                team_col = 'Team'
            elif 'team' in squad_data.columns:
                team_col = 'team'
            
            if team_col is None:
                print("WARNING: No team column found, adding default team")
                # Assign half to each team to ensure balance
                squad_data['team'] = np.where(
                    np.arange(len(squad_data)) % 2 == 0, 
                    team1, 
                    team2
                )
                team_col = 'team'
            
            # Get role requirements if not provided
            if role_requirements is None:
                role_counts = squad_data['role'].value_counts().to_dict()
                role_requirements = self.role_mapper.get_adjusted_role_requirements(role_counts)
                print(f"Using default role requirements: {role_requirements}")
            
            # Prepare for selection
            selected_players = []
            credits_remaining = 100
            team1_count = 0
            team2_count = 0
            role_counts = {role: 0 for role in role_requirements}
            
            # Ensure we have credits column - the credits_col should already be set from standardize_credits_column
            if credits_col is None:
                print("WARNING: No credits column found, adding default credits")
                # Assign random credits between 7 and 10
                squad_data['credits'] = np.random.uniform(7, 10, len(squad_data))
                credits_col = 'credits'
            
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
                    elif 'name' in squad_data.columns:
                        matches = squad_data[squad_data['name'].str.contains(name, case=False, na=False, regex=False)]
                    else:
                        matches = pd.DataFrame()  # Empty DataFrame if column doesn't exist
                except Exception as e:
                    print(f"Error matching specialist {name}: {str(e)}")
                    matches = pd.DataFrame()  # Empty DataFrame on error
                
                if not matches.empty and specialists_selected < 2:
                    player = matches.iloc[0]
                    player_credits = player[credits_col]
                    
                    # Always select specialists even if credits are tight
                    selected_players.append(player)
                    squad_data = squad_data.drop(player.name)
                    credits_remaining -= player_credits
                    role_counts[player['role']] += 1
                    
                    # Update team counts
                    if player[team_col] == team1 or team1 in str(player[team_col]):
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
                sort_col = 'predicted_points' if 'predicted_points' in squad_data.columns else credits_col
                role_players = squad_data[squad_data['role'] == role].sort_values(sort_col, ascending=False)
                
                # Select top players by points
                for _, player in role_players.iterrows():
                    if len(selected_players) >= 11:
                        break
                        
                    # Get player credits
                    player_credits = player[credits_col]
                    
                    # Skip if we don't have enough credits, unless this would leave us with < 11 players
                    if credits_remaining < player_credits and len(selected_players) < 8:
                        # Only skip if we still have many more options
                        if len(role_players) > needed * 2:
                            continue
                    
                    # Skip if adding this player would exceed team limit (maximum 7 players from one team)
                    player_team = player[team_col]
                    player_team_count = 0
                    if player_team == team1 or team1 in str(player_team):
                        player_team_count = team1_count
                    else:
                        player_team_count = team2_count
                        
                    if player_team_count >= 7:
                        continue
                        
                    # Add player to selected players
                    selected_players.append(player)
                    squad_data = squad_data.drop(player.name)
                    
                    # Update remaining credits and role count
                    credits_remaining -= player_credits
                    role_counts[player['role']] += 1
                    
                    # Update team counts
                    if player[team_col] == team1 or team1 in str(player[team_col]):
                        team1_count += 1
                    else:
                        team2_count += 1
                        
                    # Check if we have met our minimum requirements for this role
                    if role_counts[role] >= min_required:
                        break
            
            # Now, fill remaining spots with best available players regardless of role
            remaining_spots = 11 - len(selected_players)
            if remaining_spots > 0:
                # Get all remaining players sorted by predicted points, but still meet max role requirements
                sort_col = 'predicted_points' if 'predicted_points' in squad_data.columns else credits_col
                remaining_players = squad_data.sort_values(sort_col, ascending=False)
                
                for _, player in remaining_players.iterrows():
                    if len(selected_players) >= 11:
                        break
                        
                    # Get player credits
                    player_credits = player[credits_col]
                    
                    # Check credits, but be more lenient as we get closer to 11 players
                    if credits_remaining < player_credits:
                        # If we have very few players, be even more lenient
                        if len(selected_players) < 9:
                            continue
                        # For the last few spots, allow slightly exceeding credit limits if we can't find suitable players
                        elif len(selected_players) < 10 and player_credits > credits_remaining * 1.1:
                            continue
                        
                    # Check if adding this player would exceed max role requirements
                    role = player['role']
                    role_max = role_requirements[role][1]
                    if role_counts[role] >= role_max:
                        continue
                        
                    # Check if adding this player would exceed team limit (maximum 7 players from one team)
                    player_team = player[team_col]
                    player_team_count = 0
                    if player_team == team1 or team1 in str(player_team):
                        player_team_count = team1_count
                    else:
                        player_team_count = team2_count
                        
                    if player_team_count >= 7:
                        continue
                        
                    # Add player to selected players
                    selected_players.append(player)
                    squad_data = squad_data.drop(player.name)
                    
                    # Update remaining credits and role count
                    credits_remaining -= player_credits
                    role_counts[player['role']] += 1
                    
                    # Update team counts
                    if player[team_col] == team1 or team1 in str(player[team_col]):
                        team1_count += 1
                    else:
                        team2_count += 1
            
            # Sort by predicted points or credits
            sort_col = 'predicted_points' if 'predicted_points' in remaining_players.columns else credits_col
            
            final_team = pd.DataFrame(selected_players)
            
            # Select captain and vice-captain
            if not final_team.empty:
                final_team = self.select_captain_vice_captain(final_team)
            
            # Format output
            team_output = {
                'players': final_team.to_dict('records'),
                'total_points': final_team['predicted_points'].sum() if 'predicted_points' in final_team.columns else 0,
                'total_credits': final_team[credits_col].sum() if credits_col in final_team.columns else 0
            }
            
            return team_output
            
        except Exception as e:
            print(f"Error in greedy team selection: {str(e)}")
            traceback.print_exc()
            
            # Emergency fallback - return any valid 11 players
            try:
                # Convert players to DataFrame if it's not already
                if not isinstance(players, pd.DataFrame):
                    if isinstance(players, dict) and 'players' in players:
                        players_dict = players['players']
                        player_records = []
                        for player_id, player_data in players_dict.items():
                            player_record = player_data.copy()
                            player_record['player_id'] = player_id
                            player_records.append(player_record)
                        players = pd.DataFrame(player_records)
                
                # Apply absolute fallback - just take 11 players regardless of any constraints
                print("EMERGENCY FALLBACK: Selecting any 11 players without constraints")
                
                # Find a column to sort by
                sort_col = None
                for col in ['predicted_points', 'points', 'Credits', 'credits']:
                    if col in players.columns:
                        sort_col = col
                        break
                
                if sort_col:
                    players = players.sort_values(sort_col, ascending=False)
                
                # Just take first 11 rows
                emergency_players = players.head(11)
                
                # Ensure required columns exist
                if 'predicted_points' not in emergency_players.columns:
                    emergency_players['predicted_points'] = 50
                
                if 'credits' not in emergency_players.columns and 'Credits' in emergency_players.columns:
                    emergency_players['credits'] = emergency_players['Credits']
                elif 'credits' not in emergency_players.columns:
                    emergency_players['credits'] = 9.0
                
                if 'role' not in emergency_players.columns:
                    # Assign roles to make a valid team
                    emergency_players['role'] = 'BAT'  # Default
                    emergency_players.iloc[0, emergency_players.columns.get_loc('role')] = 'WK'
                    emergency_players.iloc[1:4, emergency_players.columns.get_loc('role')] = 'BOWL'
                    emergency_players.iloc[4:6, emergency_players.columns.get_loc('role')] = 'AR'
                
                # Select captain and vice captain
                emergency_players = self.select_captain_vice_captain(emergency_players)
                
                return {
                    'players': emergency_players.to_dict('records'),
                    'total_points': emergency_players['predicted_points'].sum(),
                    'total_credits': emergency_players['credits'].sum()
                }
            except Exception as emergency_error:
                print(f"Emergency fallback failed: {str(emergency_error)}")
                
                # Last resort - create 11 dummy players
                dummy_players = []
                for i in range(11):
                    role = 'BAT'
                    if i == 0:
                        role = 'WK'
                    elif i < 4:
                        role = 'BOWL'
                    elif i < 6:
                        role = 'AR'
                    
                    dummy_players.append({
                        'name': f'Player {i+1}',
                        'role': role,
                        'team': team1 if i < 6 else team2,
                        'credits': 9.0,
                        'predicted_points': 50.0,
                        'is_captain': True if i == 0 else False,
                        'is_vice_captain': True if i == 1 else False,
                        'multiplier': 2.0 if i == 0 else (1.5 if i == 1 else 1.0)
                    })
                
                return {
                    'players': dummy_players,
                    'total_points': sum(p['predicted_points'] for p in dummy_players),
                    'total_credits': sum(p['credits'] for p in dummy_players)
                }
    
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
            
            # Ensure available_players has a unique index before looping
            available_players = available_players.reset_index(drop=True)

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
                
                # Ensure remaining_available has unique index before iterating
                remaining_available = remaining_available.reset_index(drop=True)

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
                
            # Start with a clean index
            available = all_players_df.copy().reset_index(drop=True)
            
            # Basic standardization of column names
            if 'Player Name' in all_players_df.columns and 'name' not in all_players_df.columns:
                all_players_df['name'] = all_players_df['Player Name']
            
            if 'Player Type' in all_players_df.columns and 'role' not in all_players_df.columns:
                all_players_df['role'] = all_players_df['Player Type']
                
            if 'Team' in all_players_df.columns and 'team' not in all_players_df.columns:
                all_players_df['team'] = all_players_df['Team']
                
            # Standardize the credits column    
            available, credits_col = self.standardize_credits_column(available)
            
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
            elif credits_col is not None:
                available = available.sort_values([credits_col, 'random'], ascending=[False, True])
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
            
            # Standardize the credits column of impact_df
            impact_df, impact_credits_col = self.standardize_credits_column(impact_df)
            
            # Ensure we have predicted_points
            if 'predicted_points' not in impact_df.columns and len(impact_df) > 0:
                # Base points by role
                base_points = {'WK': 30, 'BAT': 35, 'AR': 40, 'BOWL': 30}
                # Get role column - either 'role' or 'Role' 
                role_col = 'role' if 'role' in impact_df.columns else 'Role'
                
                if impact_credits_col is not None and role_col in impact_df.columns:
                    # Create points list manually
                    points = []
                    for _, row in impact_df.iterrows():
                        role = row[role_col]
                        base = base_points.get(role, 35)
                        credits = float(row[impact_credits_col])
                        points.append((credits * 4 + base) * (0.9 + 0.2 * random.random()))
                    impact_df['predicted_points'] = points
                else:
                    # Default points if missing columns
                    impact_df['predicted_points'] = [75 * random.uniform(0.8, 1.2) for _ in range(len(impact_df))]
                    print("WARNING: Missing credits or role columns, using random predicted points")
            
            # Return the impact players
            return impact_df.head(num_substitutes)
            
        except Exception as e:
            print(f"Error in fallback impact player selection: {str(e)}")
            traceback.print_exc()
            # Last resort fallback - return empty DataFrame
            return pd.DataFrame()
    
    def optimize_team(self, players_or_problem, team1=None, team2=None, role_requirements=None):
        """Optimize team selection using PuLP"""
        try:
            print("Using TeamOptimizer for team selection...")
            
            # Debug information to help troubleshoot
            print(f"optimize_team received data of type: {type(players_or_problem)}")
            
            # More detailed logging of the structure
            if isinstance(players_or_problem, dict):
                print(f"Players_or_problem keys: {list(players_or_problem.keys())}")
                # Check each key in detail
                for key in players_or_problem.keys():
                    print(f"  Key '{key}' has type: {type(players_or_problem[key])}")
                    if key == 'players' and isinstance(players_or_problem[key], dict):
                        print(f"  'players' has {len(players_or_problem[key])} entries")
                        if len(players_or_problem[key]) > 0:
                            first_player_id = next(iter(players_or_problem[key]))
                            print(f"  First player ID: {first_player_id}")
                            first_player = players_or_problem[key][first_player_id]
                            print(f"  First player keys: {list(first_player.keys())}")
            
            # Variable to store the team names
            extract_team1 = team1
            extract_team2 = team2
            
            # Check if first parameter is a problem dictionary
            if isinstance(players_or_problem, dict) and 'players' in players_or_problem:
                print(f"Found players key with {len(players_or_problem['players'])} entries")
                print("Converting problem dictionary to DataFrame for optimization")
                players_dict = players_or_problem['players']
                
                # Extract player data from the dictionary format
                player_records = []
                for player_id, player_data in players_dict.items():
                    player_record = player_data.copy()
                    player_record['player_id'] = player_id
                    player_records.append(player_record)
                
                # Create DataFrame from player records
                players_df = pd.DataFrame(player_records)
                
                # Ensure required columns exist
                if 'points' in players_df.columns and 'predicted_points' not in players_df.columns:
                    players_df['predicted_points'] = players_df['points']
                
                if 'name' in players_df.columns and 'Player Name' not in players_df.columns:
                    players_df['Player Name'] = players_df['name']
                
                print(f"Converted {len(players_df)} players to DataFrame format")
                
                # Try to extract team names from the problem
                if extract_team1 is None or extract_team2 is None:
                    # Get unique team names from player data
                    team_names = set()
                    for player_id, player_data in players_dict.items():
                        if 'team' in player_data:
                            team_names.add(player_data['team'])
                    
                    team_names = list(team_names)
                    if len(team_names) >= 2:
                        extract_team1 = extract_team1 or team_names[0]
                        extract_team2 = extract_team2 or team_names[1]
                    else:
                        extract_team1 = extract_team1 or 'Team1'
                        extract_team2 = extract_team2 or 'Team2'
                
                # Extract role requirements if present
                if role_requirements is None and 'role_requirements' in players_or_problem:
                    role_requirements = players_or_problem['role_requirements']
                
                # Use the DataFrame for optimization
                squad_data = players_df
            else:
                # Use the parameters as they are
                squad_data = players_or_problem
                extract_team1 = team1 or 'Team1'
                extract_team2 = team2 or 'Team2'
            
            # Create the optimization problem using the DataFrame
            prob, player_vars, updated_players = self.create_optimization_problem(squad_data)
            
            if prob is None:
                print("Failed to create optimization problem. Using fallback method.")
                # Use the DataFrame for the fallback method
                return self._greedy_team_selection(squad_data, extract_team1, extract_team2, role_requirements)
            
            # Solve the optimization problem
            prob.solve(PULP_CBC_CMD(msg=True))
            
            # Check if a solution was found
            if LpStatus[prob.status] != "Optimal":
                print(f"Optimization problem could not be solved optimally. Status: {LpStatus[prob.status]}")
                print("Using fallback method.")
                return self._greedy_team_selection(squad_data, extract_team1, extract_team2, role_requirements)
            
            # Extract the selected team
            selected_indices = [i for i in player_vars if value(player_vars[i]) > 0.5]
            
            if not selected_indices:
                print("No players selected by optimizer. Using fallback method.")
                return self._greedy_team_selection(squad_data, extract_team1, extract_team2, role_requirements)
            
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
                'total_credits': final_team['credits'].sum() if 'credits' in final_team.columns else (final_team['Credits'].sum() if 'Credits' in final_team.columns else 0)
            }
            
            return team_output
            
        except Exception as e:
            print(f"Error in team optimization: {str(e)}")
            traceback.print_exc()
            if isinstance(squad_data, pd.DataFrame):
                return self._greedy_team_selection(squad_data, extract_team1, extract_team2, role_requirements)
            else:
                # Create an empty result if we can't even create a valid DataFrame
                return {
                    'players': [],
                    'total_points': 0,
                    'total_credits': 0
                }
    
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
        """
        Selects the captain (2x points) and vice captain (1.5x points) from the team.
        
        Args:
            selected_team (DataFrame): The team to select captain from
            
        Returns:
            DataFrame: The team with captain and vice captain at the top
        """
        try:
            if selected_team.empty:
                print("ERROR: No players in selected team for captain selection")
                return selected_team
                
            # Create a copy to avoid modifying the original
            selected_team = selected_team.copy()
            
            # Standardize the credits column
            selected_team, credits_col = self.standardize_credits_column(selected_team)
            
            # Make sure we have the columns we need
            if not all(col in selected_team.columns for col in ['predicted_points', 'role']):
                print("WARNING: Missing required columns for captain selection")
                
            # Create predicted_points if it doesn't exist
            if 'predicted_points' not in selected_team.columns:
                print("WARNING: No predicted_points column, creating based on credits")
                if credits_col is not None:
                    # Use standardized credits column
                    selected_team['predicted_points'] = selected_team[credits_col].astype(float) * 100
                else:
                    # Default value if no credits column exists
                    selected_team['predicted_points'] = 500
                    print("WARNING: No credits column found, using default predicted points")
                
            # Create role if it doesn't exist
            if 'role' not in selected_team.columns:
                selected_team['role'] = 'BAT'  # Default role
            
            # Sort by predicted points
            sorted_team = selected_team.sort_values('predicted_points', ascending=False)
            
            if len(sorted_team) < 3:
                print(f"WARNING: Not enough players ({len(sorted_team)}) to select captain and vice captain properly")
                if len(sorted_team) < 1:
                    return selected_team
                    
            # Select captain (top player)
            captain = sorted_team.iloc[0]
            captain_series = captain.copy()
            captain_series['is_captain'] = True
            captain_series['is_vice_captain'] = False
            captain_series['multiplier'] = 2.0  # Captain gets 2x points
            
            # Select vice captain (randomly from next 2 top players or if not enough, from remaining)
            if len(sorted_team) >= 3:
                vice_captain_options = sorted_team.iloc[1:3]
                vice_captain_idx = np.random.choice(vice_captain_options.index)
                vice_captain = sorted_team.loc[vice_captain_idx]
            else:
                # If only 1 or 2 players, select randomly or the second player
                remaining_players = sorted_team.iloc[1:] if len(sorted_team) > 1 else sorted_team
                vice_captain_idx = np.random.choice(remaining_players.index)
                vice_captain = sorted_team.loc[vice_captain_idx]
                
            vice_captain_series = vice_captain.copy()
            vice_captain_series['is_captain'] = False
            vice_captain_series['is_vice_captain'] = True
            vice_captain_series['multiplier'] = 1.5  # Vice captain gets 1.5x points
            
            # Remove captain and vice captain from regular team
            remaining_team = sorted_team.drop([captain.name, vice_captain.name])
            for player in remaining_team.itertuples():
                remaining_team.at[player.Index, 'is_captain'] = False
                remaining_team.at[player.Index, 'is_vice_captain'] = False
                remaining_team.at[player.Index, 'multiplier'] = 1.0
                
            # Combine to create final team with captain and vice captain at top
            final_team = pd.concat([
                pd.DataFrame([captain_series, vice_captain_series]),
                remaining_team
            ])
            
            return final_team
            
        except Exception as e:
            print(f"ERROR in captain selection: {str(e)}")
            
            # Fallback method - just mark someone as captain
            try:
                if not selected_team.empty:
                    selected_team['is_captain'] = False
                    selected_team['is_vice_captain'] = False
                    selected_team['multiplier'] = 1.0
                    
                    # Mark first player as captain
                    selected_team.iloc[0, selected_team.columns.get_loc('is_captain')] = True
                    selected_team.iloc[0, selected_team.columns.get_loc('multiplier')] = 2.0
                    
                    # Mark second player as vice captain if possible
                    if len(selected_team) > 1:
                        selected_team.iloc[1, selected_team.columns.get_loc('is_vice_captain')] = True
                        selected_team.iloc[1, selected_team.columns.get_loc('multiplier')] = 1.5
            except Exception as fallback_error:
                print(f"ERROR in captain fallback: {str(fallback_error)}")
                
            return selected_team

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
    
    # --- New Monte Carlo Team Selection Method ---
    def monte_carlo_team_selection(self, problem, num_iterations=5, variance_factor=0.2):
        """
        Run Monte Carlo simulation to generate multiple team variations
        
        Args:
            problem (dict): Optimization problem structure
            num_iterations (int): Number of Monte Carlo iterations
            variance_factor (float): Controls the amount of random variation (0.0-1.0)
            
        Returns:
            dict: Result containing:
                - best_team: The highest expected value team
                - team_variations: List of alternative teams
                - variance_analysis: Analysis of player selection frequency
        """
        self.logger.info(f"Running Monte Carlo team selection with {num_iterations} iterations")
        
        # Debug print to log the structure of the problem
        print(f"monte_carlo_team_selection received problem of type: {type(problem)}")
        print(f"Problem keys: {list(problem.keys()) if isinstance(problem, dict) else 'Not a dict'}")
        if isinstance(problem, dict):
            for key in problem.keys():
                print(f"  Key '{key}' has type: {type(problem[key])}")
        
        try:
            # Validate problem structure
            if 'players' not in problem or not problem['players']:
                self.logger.error("Invalid or empty problem structure (missing players)")
                return self.optimize_team(problem, 'Team1', 'Team2')  # Fall back to standard optimization
            
            # Extract team names if present in the problem
            team1 = problem.get('team1', 'Team1')
            team2 = problem.get('team2', 'Team2')
            
            # Limit iterations to a reasonable number
            if num_iterations > 50:
                self.logger.warning(f"Requested {num_iterations} iterations is too high, limiting to 50")
                num_iterations = 50
        
            # Store all generated teams
            all_teams = []
            team_scores = []
            player_selection_counts = {}
            successful_iterations = 0
            
            # Create multiple variations of the problem with random point fluctuations
            for i in range(num_iterations):
                try:
                    # Create a copy of the original problem with random variations
                    mc_problem = self._create_monte_carlo_problem_variation(problem, variance_factor)
                    
                    # Debug print to verify mc_problem structure
                    print(f"After _create_monte_carlo_problem_variation, mc_problem has keys: {list(mc_problem.keys()) if isinstance(mc_problem, dict) else 'Not a dict'}")
                    if isinstance(mc_problem, dict) and 'players' in mc_problem:
                        print(f"mc_problem['players'] has {len(mc_problem['players'])} entries")
                    
                    # Pass team1 and team2 to optimize_team
                    self.logger.info(f"Running Monte Carlo iteration {i+1}/{num_iterations}")
                    mc_result = self.optimize_team(mc_problem, team1, team2)
                    
                    if mc_result and 'players' in mc_result and mc_result['players']:
                        team = pd.DataFrame(mc_result['players'])
                        team_score = mc_result.get('total_points', 0)
                        
                        # Add to collection of teams
                        all_teams.append(team)
                        team_scores.append(team_score)
                        successful_iterations += 1
                        
                        # Track player selection frequency
                        for _, player in team.iterrows():
                            player_name = player.get('Player Name', player.get('player_name', player.get('name', 'Unknown')))
                            if player_name in player_selection_counts:
                                player_selection_counts[player_name] += 1
                            else:
                                player_selection_counts[player_name] = 1
                    else:
                        self.logger.warning(f"Monte Carlo iteration {i+1} failed to produce a valid team")
                except Exception as e:
                    self.logger.error(f"Error in Monte Carlo iteration {i+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            if not all_teams:
                self.logger.warning("Monte Carlo simulation failed to generate any valid teams")
                return self.optimize_team(problem, team1, team2)  # Fall back to standard optimization
            
            # Find the best team (highest expected value)
            best_team_idx = team_scores.index(max(team_scores))
            best_team = all_teams[best_team_idx]
            
            # Calculate selection probability for each player
            total_iterations = successful_iterations
            player_selection_probabilities = {
                player: count / total_iterations 
                for player, count in player_selection_counts.items()
            }
            
            # Identify core players (selected in >70% of teams)
            core_players = [player for player, prob in player_selection_probabilities.items() 
                          if prob > 0.7]
            
            # Identify flex players (selected in 30-70% of teams)
            flex_players = [player for player, prob in player_selection_probabilities.items() 
                          if 0.3 <= prob <= 0.7]
            
            # Prepare variance analysis
            variance_analysis = {
                'player_selection_probabilities': player_selection_probabilities,
                'core_players': core_players,
                'flex_players': flex_players,
                'total_iterations': total_iterations,
                'average_team_score': sum(team_scores) / len(team_scores) if team_scores else 0,
                'score_variance': np.var(team_scores) if len(team_scores) > 1 else 0
            }
            
            # Convert best team to dict for return
            best_team_dict = best_team.to_dict('records')
            
            # Prepare result with best team and alternatives
            result = {
                'players': best_team_dict,
                'total_points': team_scores[best_team_idx],
                'variance_analysis': variance_analysis,
                'team_variations': [team.to_dict('records') for team in all_teams],
                'is_monte_carlo': True
            }
            
            # Add up team role counts to verify balanced team
            role_counts = {}
            for player in best_team_dict:
                role = player.get('role', 'unknown').split()[0]  # Remove (C)/(VC) if present
                role_counts[role] = role_counts.get(role, 0) + 1
                
            self.logger.info(f"Monte Carlo result role distribution: {role_counts}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo team selection: {str(e)}")
            traceback.print_exc()
            # Extract team names if present in the problem
            team1 = problem.get('team1', 'Team1')
            team2 = problem.get('team2', 'Team2')
            # Fall back to standard optimization with team names
            return self.optimize_team(problem, team1, team2)
    
    def _create_monte_carlo_problem_variation(self, problem, variance_factor=0.2):
        """Create a variation of the problem with randomized point predictions"""
        # Debug print
        print(f"_create_monte_carlo_problem_variation received problem of type: {type(problem)}")
        print(f"Problem keys: {list(problem.keys()) if isinstance(problem, dict) else 'Not a dict'}")
        
        # Create a deep copy of the problem to avoid modifying the original
        import copy
        mc_problem = copy.deepcopy(problem)
        
        # Debug print after copy
        print(f"After copy, mc_problem has keys: {list(mc_problem.keys()) if isinstance(mc_problem, dict) else 'Not a dict'}")
        
        # Check if 'players' key exists
        if not isinstance(mc_problem, dict) or 'players' not in mc_problem:
            print("Error: mc_problem is not a dict or doesn't have 'players' key")
            # Create a minimal structure
            mc_problem = {
                'players': {},
                'role_requirements': {
                    'WK': (1, 4),
                    'BAT': (3, 6),
                    'AR': (1, 4),
                    'BOWL': (3, 6)
                }
            }
            return mc_problem
        
        # Add random variations to player points
        try:
            for player_id, player in mc_problem['players'].items():
                # Calculate random variation factor (higher variance_factor = more randomness)
                random_factor = 1.0 + (np.random.random() - 0.5) * 2 * variance_factor
                
                # Apply random variation to points
                if 'points' in player:
                    original_points = player['points']
                    player['points'] = original_points * random_factor
                    
                    # Also track the original points and the variation applied
                    player['original_points'] = original_points
                    player['variance_factor'] = random_factor
                else:
                    print(f"Warning: Player {player_id} doesn't have 'points' key")
        except Exception as e:
            print(f"Error adding random variations: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Debug print before returning
        print(f"Before returning, mc_problem has keys: {list(mc_problem.keys()) if isinstance(mc_problem, dict) else 'Not a dict'}")
        if 'players' in mc_problem:
            print(f"mc_problem['players'] has {len(mc_problem['players'])} entries")
        
        return mc_problem
    # --- End Monte Carlo Implementation ---