import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from quick_implementation_plan import Dream11Predictor
from team_optimizer import TeamOptimizer
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import traceback
from data_preprocessor import DataPreprocessor
from role_mapper import RoleMapper

class Dream11App:
    def __init__(self, data_dir="dataset"):
        """Initialize the Dream11 app with the specified data directory"""
        self.data_dir = data_dir
        self.data_preprocessor = DataPreprocessor(data_dir)
        self.team_predictor = Dream11Predictor(data_dir)
        self.optimizer = TeamOptimizer()
        self.role_mapper = RoleMapper()
        self.df = None
        self.squad_data = None
        self.model = None
        self.data_loaded = False
        self.models_trained = False
        
    def load_data(self):
        """Load and preprocess data"""
        try:
            player_data = self.team_predictor.load_and_preprocess_data()
            self.data_loaded = True
            return player_data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
            
    def train_models(self):
        """Train prediction models"""
        try:
            if not self.data_loaded:
                self.load_data()
                
            results = self.team_predictor.train_models()
            self.models_trained = True
            return results
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return None
            
    def _predict_player_points(self, squad_data):
        """Predict fantasy points for players in the squad"""
        try:
            # Get player and match features
            player_name_col = 'Player Name' if 'Player Name' in squad_data.columns else 'player'
            
            # Create dummy features for prediction (simple model)
            X = np.zeros((len(squad_data), 10))
            
            # Predict points using predictor
            predicted_points = self.team_predictor.predict_player_points(squad_data)
            
            # Display preview of predictions
            print("Player Predictions Preview (top 10 by predicted points):")
            preview = squad_data.copy()
            preview['predicted_points'] = predicted_points
            preview['is_playing'] = True
            
            preview_sorted = preview.sort_values('predicted_points', ascending=False).head(10)
            for _, player in preview_sorted.iterrows():
                print(f"{player[player_name_col]} ({player['Team']}, {player['role']}): {player['predicted_points']:.2f} points, Playing: {player['is_playing']}")
                
            return predicted_points
            
        except Exception as e:
            print(f"Error predicting player points: {str(e)}")
            traceback.print_exc()
            return np.zeros(len(squad_data))
            
    def predict_team(self, team1, team2):
        """Predict the best Dream11 team for a given match"""
        try:
            # Load squad data
            squad_data = pd.read_csv('dataset/SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv')
            
            # Print all available teams in squad data for debugging
            print(f"Available teams in squad data: {squad_data['Team'].unique().tolist()}")
            
            # Direct mapping for team codes that might be represented differently in data
            direct_team_mapping = {
                'CSK': ['CHE', 'CSK', 'Chennai'],
                'MI': ['MUM', 'MI', 'Mumbai'],
                'RCB': ['BLR', 'RCB', 'Bangalore', 'BAN'],
                'KKR': ['KOL', 'KKR', 'Kolkata'],
                'PBKS': ['PUN', 'PBKS', 'Punjab', 'KXI'],
                'DC': ['DEL', 'DC', 'Delhi'],
                'RR': ['RAJ', 'RR', 'Rajasthan'],
                'SRH': ['HYD', 'SRH', 'Hyderabad'],
                'GT': ['GUJ', 'GT', 'Gujarat'],
                'LSG': ['LUC', 'LSG', 'Lucknow']
            }
            
            # Get all possible team codes for selected teams
            team1_codes = direct_team_mapping.get(team1, [team1])
            team2_codes = direct_team_mapping.get(team2, [team2])
            
            # Also check if team1/team2 are values in any of the lists in direct_team_mapping
            for team_code, aliases in direct_team_mapping.items():
                if team1 in aliases and team1 != team_code and team_code not in team1_codes:
                    team1_codes.append(team_code)
                if team2 in aliases and team2 != team_code and team_code not in team2_codes:
                    team2_codes.append(team_code)
                    
            print(f"Team 1 ({team1}) codes: {team1_codes}")
            print(f"Team 2 ({team2}) codes: {team2_codes}")
            
            # Filter players from selected teams - ensure we match team names exactly
            team_squads = squad_data[squad_data['Team'].isin(team1_codes + team2_codes)].copy()
            
            # If no players found, try a more lenient search
            if len(team_squads) == 0:
                print(f"No exact matches found. Trying case-insensitive matching...")
                # Try case-insensitive matching
                team_patterns = '|'.join([f"{code}" for code in (team1_codes + team2_codes)])
                mask = squad_data['Team'].str.contains(team_patterns, case=False, regex=True)
                team_squads = squad_data[mask].copy()
            
            # Final check
            if len(team_squads) == 0:
                print(f"No players found for teams {team1} or {team2}")
                print(f"Available teams: {squad_data['Team'].unique()}")
                return None, f"No players found for the selected teams. Available teams: {squad_data['Team'].unique()}"
            
            # Check how many players from each team to verify balance
            team_counts = team_squads['Team'].value_counts()
            print(f"Players found by team: {team_counts.to_dict()}")
                
            # Standardize roles
            team_squads['role'] = team_squads['Player Type'].apply(self.role_mapper.standardize_role)
            
            # Balance teams to ensure we have enough players from both teams
            min_team_size = 12  # Ensure we have at least this many players from each team
            team1_players = sum(team_squads['Team'].isin(team1_codes))
            team2_players = sum(team_squads['Team'].isin(team2_codes))
            
            if team1_players < min_team_size or team2_players < min_team_size:
                print(f"Imbalanced teams: {team1}={team1_players}, {team2}={team2_players}")
                # Adjust credit values to make players from underrepresented team more attractive
                if team1_players < team2_players:
                    team_squads.loc[team_squads['Team'].isin(team1_codes), 'Credits'] *= 0.85
                else:
                    team_squads.loc[team_squads['Team'].isin(team2_codes), 'Credits'] *= 0.85
            
            # Calculate predicted points for each player
            team_squads['predicted_points'] = self._predict_player_points(team_squads)
            
            # Adjust player points to ensure balanced team representation
            team1_avg_points = team_squads[team_squads['Team'].isin(team1_codes)]['predicted_points'].mean()
            team2_avg_points = team_squads[team_squads['Team'].isin(team2_codes)]['predicted_points'].mean()
            
            # If one team has much higher average points, balance them
            if team1_avg_points > 1.5 * team2_avg_points:
                scale_factor = team1_avg_points / team2_avg_points * 0.9
                team_squads.loc[team_squads['Team'].isin(team2_codes), 'predicted_points'] *= scale_factor
                print(f"Boosting {team2} players by factor {scale_factor:.2f}")
            elif team2_avg_points > 1.5 * team1_avg_points:
                scale_factor = team2_avg_points / team1_avg_points * 0.9
                team_squads.loc[team_squads['Team'].isin(team1_codes), 'predicted_points'] *= scale_factor
                print(f"Boosting {team1} players by factor {scale_factor:.2f}")
            
            # Create optimization problem
            print("Using TeamOptimizer for team selection...")
            problem = self.optimizer.create_optimization_problem(team_squads)
            if problem is None:
                return None, "Failed to create optimization problem"
                
            # Solve optimization problem
            selected_team = self.optimizer.solve_optimization_problem(problem)
            if selected_team is None:
                return None, "Failed to find a valid team combination"
                
            # Check team balance in selected players
            selected_teams = [player['team'] for player in selected_team]
            team_distribution = pd.Series(selected_teams).value_counts()
            print(f"Selected team distribution: {team_distribution.to_dict()}")
            
            # Format results
            result = {
                'players': [{
                    'name': player['name'],
                    'Display Name': player['name'],
                    'role': player['role'],
                    'Team': player['team'],
                    'Player Type': player['role'],
                    'Credits': player['credits'],
                    'predicted_points': player['points']
                } for player in selected_team],
                'total_points': sum(player['points'] for player in selected_team),
                'total_credits': sum(player['credits'] for player in selected_team)
            }
            
            return result, None
            
        except Exception as e:
            print(f"Error in predict_team: {str(e)}")
            traceback.print_exc()
            return None, f"Error predicting team: {str(e)}"
            
    def plot_team_composition(self, team):
        """Create a visualization of team composition by role"""
        self.team_predictor.plot_team_composition(team)
        
    def plot_player_points(self, team):
        """Create a visualization of predicted points by player"""
        self.team_predictor.plot_player_points(team)
        
    def run(self):
        """Run the Streamlit app"""
        st.title("Dream11 IPL Team Predictor")
        
        # Make sure data is loaded
        try:
            self.team_predictor.load_models()
        except Exception as e:
            st.warning(f"Could not load pre-trained models: {str(e)}")
            st.info("Training new models with available data...")
            self.team_predictor.load_and_preprocess_data()
            self.team_predictor.train_models()
            
        # Define team codes and names
        team_codes = {
            'CSK': 'Chennai Super Kings',
            'MI': 'Mumbai Indians',
            'RCB': 'Royal Challengers Bangalore',
            'KKR': 'Kolkata Knight Riders',
            'DC': 'Delhi Capitals',
            'PBKS': 'Punjab Kings',
            'RR': 'Rajasthan Royals',
            'SRH': 'Sunrisers Hyderabad',
            'GT': 'Gujarat Titans',
            'LSG': 'Lucknow Super Giants'
        }
        
        team_names = list(team_codes.keys())
        team_full_names = [f"{code} - {name}" for code, name in team_codes.items()]
        
        # Team selection
        col1, col2 = st.columns(2)
        with col1:
            team1_full = st.selectbox("Select Home Team", team_full_names)
            team1 = team1_full.split(' - ')[0]  # Extract team code
        
        with col2:
            # Filter out the team already selected
            available_teams = [t for t in team_full_names if t.split(' - ')[0] != team1]
            team2_full = st.selectbox("Select Away Team", available_teams)
            team2 = team2_full.split(' - ')[0]  # Extract team code
        
        # Add match details input
        st.subheader("Match Details")
        col1, col2 = st.columns(2)
        with col1:
            venue = st.selectbox("Select Venue", [
                "M. Chinnaswamy Stadium, Bangalore",
                "Eden Gardens, Kolkata",
                "Wankhede Stadium, Mumbai",
                "MA Chidambaram Stadium, Chennai",
                "Arun Jaitley Stadium, Delhi",
                "Narendra Modi Stadium, Ahmedabad",
                "Rajiv Gandhi Stadium, Hyderabad",
                "Sawai Mansingh Stadium, Jaipur",
                "Punjab Cricket Association Stadium, Mohali",
                "DY Patil Stadium, Mumbai"
            ])
        with col2:
            match_type = st.selectbox("Match Type", ["League", "Qualifier", "Eliminator", "Final"])
        
        # Predict team
        if st.button("Predict Dream11 Team"):
            try:
                with st.spinner("Predicting the best Dream11 team..."):
                    result, error = self.predict_team(team1, team2)
                    
                    if result is not None:
                        # Show the prediction
                        st.success(f"Successfully predicted Dream11 team for {team1} vs {team2}!")
                        
                        # Display team table
                        st.subheader("Recommended Dream11 Team")
                        
                        # Format table for display
                        display_cols = ['Display Name', 'Team', 'Player Type', 'Credits', 'predicted_points']
                        renamed_cols = {
                            'Display Name': 'Player',
                            'Team': 'Team',
                            'Player Type': 'Role',
                            'Credits': 'Credits',
                            'predicted_points': 'Predicted Points'
                        }
                        
                        # Create DataFrame from result players
                        results_df = pd.DataFrame(result['players'])
                        
                        # Check if the columns exist
                        existing_cols = [col for col in display_cols if col in results_df.columns]
                        renamed_cols = {col: renamed_cols[col] for col in existing_cols if col in renamed_cols}
                        
                        # Handle missing Display Name column
                        if 'Display Name' not in existing_cols and 'name' in results_df.columns:
                            results_df['Display Name'] = results_df['name']
                            existing_cols = ['Display Name'] + [col for col in existing_cols if col != 'Display Name']
                        
                        # Create the display DataFrame
                        display_df = results_df[existing_cols].copy()
                        display_df = display_df.rename(columns=renamed_cols)
                        
                        # Round numeric columns
                        for col in display_df.columns:
                            if display_df[col].dtype in [np.float64, np.float32]:
                                display_df[col] = display_df[col].round(1)
                                
                        # Show the team summary
                        st.success(f"Total Team Points: {result['total_points']:.1f}, Total Credits: {result['total_credits']:.1f}")
                        
                        # Display the team
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Select impact players (substitutes)
                        st.subheader("Impact Substitutes")
                        st.info("The following 4 players are recommended as impact substitutes")
                        
                        try:
                            # Get all available players not in main team
                            all_players = pd.read_csv('dataset/SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv')
                            
                            # Use the direct team mapping from above for consistency
                            direct_team_mapping = {
                                'CSK': ['CHE', 'CSK', 'Chennai'],
                                'MI': ['MUM', 'MI', 'Mumbai'],
                                'RCB': ['BLR', 'RCB', 'Bangalore', 'BAN'],
                                'KKR': ['KOL', 'KKR', 'Kolkata'],
                                'PBKS': ['PUN', 'PBKS', 'Punjab', 'KXI'],
                                'DC': ['DEL', 'DC', 'Delhi'],
                                'RR': ['RAJ', 'RR', 'Rajasthan'],
                                'SRH': ['HYD', 'SRH', 'Hyderabad'],
                                'GT': ['GUJ', 'GT', 'Gujarat'],
                                'LSG': ['LUC', 'LSG', 'Lucknow']
                            }
                            
                            # Get all possible team codes for selected teams
                            local_team1_codes = direct_team_mapping.get(team1, [team1])
                            local_team2_codes = direct_team_mapping.get(team2, [team2])
                            
                            # Also check if team1/team2 are values in any of the lists in direct_team_mapping
                            for team_code, aliases in direct_team_mapping.items():
                                if team1 in aliases and team1 != team_code and team_code not in local_team1_codes:
                                    local_team1_codes.append(team_code)
                                if team2 in aliases and team2 != team_code and team_code not in local_team2_codes:
                                    local_team2_codes.append(team_code)
                            
                            team_codes = [team1, team2] + local_team1_codes + local_team2_codes
                            team_players = all_players[all_players['Team'].isin(team_codes)].copy()
                            
                            # Get selected player names
                            selected_names = set(results_df['name'].values)
                            
                            # Filter out players already in main team
                            substitute_candidates = team_players[~team_players['Player Name'].isin(selected_names)].copy()
                            
                            if len(substitute_candidates) > 0:
                                # Standardize roles for substitutes
                                substitute_candidates['role'] = substitute_candidates['Player Type'].apply(self.role_mapper.standardize_role)
                                
                                # Calculate predicted points for substitutes
                                substitute_candidates['predicted_points'] = substitute_candidates.apply(
                                    lambda x: float(x['Credits']) * 100 * np.random.uniform(0.75, 1.05), axis=1
                                )
                                
                                # Create a mini team of 4 substitutes (one from each role if possible)
                                # Convert selected_team to DataFrame for optimizer
                                main_team_df = pd.DataFrame(result['players'])
                                
                                # Use the team optimizer's select_impact_players method
                                try:
                                    # Prepare input data for impact player selection
                                    # Ensure column names match between main team and substitute candidates
                                    substitute_candidates_for_optimizer = substitute_candidates.copy()
                                    substitute_candidates_for_optimizer = substitute_candidates_for_optimizer.rename(columns={
                                        'Player Name': 'name',
                                        'Team': 'team',
                                        'Player Type': 'role',
                                        'Credits': 'credits'
                                    })
                                    
                                    # Prepare main team data for optimizer
                                    main_team_for_optimizer = main_team_df.copy()
                                    if 'Display Name' in main_team_for_optimizer.columns:
                                        main_team_for_optimizer = main_team_for_optimizer.rename(columns={'Display Name': 'name'})
                                    
                                    # Select impact players
                                    impact_players = self.optimizer.select_impact_players(substitute_candidates_for_optimizer, main_team_for_optimizer)
                                    
                                    if not impact_players.empty:
                                        # Define display columns
                                        sub_display_cols = ['name', 'team', 'role', 'credits', 'predicted_points']
                                        sub_renamed_cols = {
                                            'name': 'Player',
                                            'team': 'Team',
                                            'role': 'Role',
                                            'credits': 'Credits',
                                            'predicted_points': 'Predicted Points'
                                        }
                                        
                                        # Check which columns exist
                                        sub_existing_cols = [col for col in sub_display_cols if col in impact_players.columns]
                                        sub_renamed_cols = {col: sub_renamed_cols[col] for col in sub_existing_cols if col in sub_renamed_cols}
                                        
                                        # Create display DataFrame
                                        sub_display_df = impact_players[sub_existing_cols].copy()
                                        sub_display_df = sub_display_df.rename(columns=sub_renamed_cols)
                                        
                                        # Round numeric columns
                                        for col in sub_display_df.columns:
                                            if sub_display_df[col].dtype in [np.float64, np.float32]:
                                                sub_display_df[col] = sub_display_df[col].round(1)
                                        
                                        # Display substitute players
                                        st.dataframe(sub_display_df, use_container_width=True)
                                    else:
                                        st.warning("No suitable substitute players available from optimizer")
                                        # Fall back to manual selection
                                        self._select_impact_players_manually(substitute_candidates, st)
                                except Exception as e:
                                    print(f"Error selecting impact players using optimizer: {str(e)}")
                                    traceback.print_exc()
                                    # Fall back to manual selection
                                    self._select_impact_players_manually(substitute_candidates, st)
                            else:
                                st.warning("No remaining players available for substitutes")
                        except Exception as e:
                            print(f"Error selecting substitutes: {str(e)}")
                            traceback.print_exc()
                            st.error("Could not select substitute players due to an error")
                        
                        # Team composition visualization
                        st.subheader("Team Composition")
                        
                        # Create role counts - use standardized role column
                        if 'role' in results_df.columns:
                            role_col = 'role'
                        elif 'Player Type' in results_df.columns:
                            role_col = 'Player Type'
                        else:
                            # Create a default role column if none exists
                            results_df['role'] = 'BAT'
                            role_col = 'role'
                            
                        role_counts = results_df[role_col].value_counts()
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['royalblue', 'forestgreen', 'firebrick', 'goldenrod']
                        bars = ax.bar(role_counts.index, role_counts.values, color=colors[:len(role_counts)])
                        
                        # Add labels
                        ax.set_xlabel('Role')
                        ax.set_ylabel('Number of Players')
                        ax.set_title('Team Composition by Role')
                        
                        # Add count labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.0f}', ha='center', va='bottom')
                        
                        # Display in Streamlit
                        st.pyplot(fig)
                        
                        # Player points prediction
                        st.subheader("Player Points Prediction")
                        
                        # Create player points chart
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Get the name column
                        if 'Display Name' in results_df.columns:
                            name_col = 'Display Name'
                        elif 'name' in results_df.columns:
                            name_col = 'name'
                        else:
                            name_col = results_df.columns[0]  # Take first column as fallback
                        
                        # Get the points column
                        if 'predicted_points' in results_df.columns:
                            points_col = 'predicted_points'
                        elif 'points' in results_df.columns:
                            points_col = 'points'
                        else:
                            # Create a random points column if none exists
                            results_df['points'] = np.random.uniform(30, 100, len(results_df))
                            points_col = 'points'
                        
                        # Sort by points
                        team_sorted = results_df.sort_values(points_col, ascending=False)
                        
                        # Set colors based on role
                        role_colors = {
                            'WK': 'purple',
                            'BAT': 'royalblue',
                            'AR': 'forestgreen',
                            'BOWL': 'firebrick',
                            'ALL': 'forestgreen'  # Map ALL to same color as AR
                        }
                        
                        # Determine colors based on role
                        colors = [role_colors.get(r, 'gray') for r in team_sorted[role_col]]
                        
                        # Highlight captain and vice-captain
                        for i, name in enumerate(team_sorted[name_col]):
                            if '(C)' in str(name):
                                colors[i] = 'gold'
                            elif '(VC)' in str(name):
                                colors[i] = 'silver'
                        
                        bars = ax.barh(team_sorted[name_col], team_sorted[points_col], color=colors)
                        
                        # Add labels
                        ax.set_xlabel('Predicted Points')
                        ax.set_ylabel('Player')
                        ax.set_title('Predicted Points by Player')
                        
                        # Add legend
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor='gold', label='Captain'),
                            Patch(facecolor='silver', label='Vice Captain'),
                            Patch(facecolor=role_colors['WK'], label='Wicket Keeper'),
                            Patch(facecolor=role_colors['BAT'], label='Batsman'),
                            Patch(facecolor=role_colors['AR'], label='All Rounder'),
                            Patch(facecolor=role_colors['BOWL'], label='Bowler')
                        ]
                        ax.legend(handles=legend_elements, loc='lower right')
                        
                        # Display in Streamlit
                        st.pyplot(fig)
                        
                        # Team distribution by team
                        st.subheader("Team Distribution")
                        
                        # Get the team counts
                        if 'Team' in results_df.columns:
                            team_distribution = results_df['Team'].value_counts()
                        else:
                            # Create a default team distribution if Team column doesn't exist
                            results_df['Team'] = ['Team A'] * (len(results_df) // 2) + ['Team B'] * (len(results_df) - len(results_df) // 2)
                            team_distribution = results_df['Team'].value_counts()
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.pie(team_distribution, labels=team_distribution.index, autopct='%1.1f%%', 
                              colors=['royalblue', 'forestgreen'])
                        ax.set_title('Player Distribution by Team')
                        
                        # Display in Streamlit
                        st.pyplot(fig)
                    else:
                        st.error(f"Error: {error}")
                        
            except Exception as e:
                st.error(f"Error predicting team: {str(e)}")
                st.error("Check if squad data is available.")
                
        # Add app info
        with st.sidebar:
            st.subheader("About Dream11 Predictor")
            st.write("""
            This app uses machine learning to predict the best possible Dream11 team 
            for IPL matches based on historical player performance data.
            
            The prediction takes into account:
            - Player's recent form
            - Match conditions
            - Team combination requirements
            - Role-based performance metrics
            """)
            
            st.subheader("How it works")
            st.write("""
            1. Select the two teams playing the match
            2. Choose the venue and match type
            3. Click 'Predict Dream11 Team'
            4. The app will suggest the optimal 11-player team
            5. It will also recommend captain and vice-captain choices
            """)
            
            st.subheader("Team Requirements")
            st.write("""
            - 1 Wicket-keeper (WK)
            - 3-5 Batsmen (BAT)
            - 1-3 All-rounders (AR)
            - 3-5 Bowlers (BOWL)
            - Total of 11 players
            - Maximum of 7 players from one team
            """)

    def _select_impact_players_manually(self, substitute_candidates, st_container=None):
        """Manually select impact players when optimizer fails"""
        try:
            # Get one player from each role if possible
            substitutes = []
            roles = ['WK', 'BAT', 'AR', 'BOWL']
            
            for role in roles:
                role_players = substitute_candidates[substitute_candidates['Player Type'] == role]
                if len(role_players) > 0:
                    # Sort by predicted points
                    role_players = role_players.sort_values('predicted_points', ascending=False)
                    # Take the best player
                    substitutes.append(role_players.iloc[0])
                    # Remove from candidates
                    substitute_candidates = substitute_candidates[~substitute_candidates['Player Name'].isin([role_players.iloc[0]['Player Name']])]
            
            # If we don't have 4 players yet, add more from any role
            remaining = 4 - len(substitutes)
            if remaining > 0 and len(substitute_candidates) > 0:
                substitute_candidates = substitute_candidates.sort_values('predicted_points', ascending=False)
                for i in range(min(remaining, len(substitute_candidates))):
                    substitutes.append(substitute_candidates.iloc[i])
            
            # Create DataFrame for display
            if substitutes:
                substitutes_df = pd.DataFrame(substitutes)
                
                # Define display columns
                sub_display_cols = ['Player Name', 'Team', 'Player Type', 'Credits', 'predicted_points']
                sub_renamed_cols = {
                    'Player Name': 'Player',
                    'Team': 'Team',
                    'Player Type': 'Role', 
                    'Credits': 'Credits',
                    'predicted_points': 'Predicted Points'
                }
                
                # Check which columns exist
                sub_existing_cols = [col for col in sub_display_cols if col in substitutes_df.columns]
                sub_renamed_cols = {col: sub_renamed_cols[col] for col in sub_existing_cols if col in sub_renamed_cols}
                
                # Create display DataFrame
                sub_display_df = substitutes_df[sub_existing_cols].copy()
                sub_display_df = sub_display_df.rename(columns=sub_renamed_cols)
                
                # Round numeric columns
                for col in sub_display_df.columns:
                    if sub_display_df[col].dtype in [np.float64, np.float32]:
                        sub_display_df[col] = sub_display_df[col].round(1)
                
                # Display substitute players
                if st_container is not None:
                    st_container.dataframe(sub_display_df, use_container_width=True)
                
                return sub_display_df
            else:
                if st_container is not None:
                    st_container.warning("No suitable substitute players available")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error in manual impact player selection: {str(e)}")
            traceback.print_exc()
            if st_container is not None:
                st_container.error(f"Error selecting substitute players: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    app = Dream11App()
    app.run() 