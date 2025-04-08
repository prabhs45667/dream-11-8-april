import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from team_optimizer import TeamOptimizer
import plotly.graph_objects as go
import plotly.express as px

class Dream11App:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.team_optimizer = TeamOptimizer()
        
    def load_data(self):
        """Load and preprocess data"""
        with st.spinner('Loading and preprocessing data...'):
            features, target = self.preprocessor.prepare_training_data()
            return features, target
            
    def train_models(self, features, target):
        """Train prediction models"""
        with st.spinner('Training models...'):
            models = self.model_trainer.train_models(features, target)
            return models
            
    def predict_team(self, team1, team2, models):
        """Predict best team for given match"""
        # Get players from both teams
        players = self.preprocessor.get_match_players(team1, team2)
        
        # Get feature columns for prediction
        feature_columns = [col for col in players.columns if col not in ['role', 'credits', 'team']]
        
        # Predict points for each player
        X = self.model_trainer.scaler.transform(players[feature_columns])
        predicted_points = models['best_model'].predict(X)
        
        # Create a Series with predicted points aligned to player indices
        predicted_points_series = pd.Series(predicted_points, index=players.index)
        
        # Optimize team selection
        main_team, total_points = self.team_optimizer.optimize_team(players, predicted_points_series)
        backup_team = self.team_optimizer.create_backup_team(players, predicted_points_series, main_team)
        
        # Format output
        team_output = self.team_optimizer.format_team_output(
            main_team, backup_team, players, predicted_points_series
        )
        
        return team_output
        
    def plot_team_composition(self, team_output):
        """Create team composition visualization"""
        roles = ['WK', 'BAT', 'AR', 'BOWL', 'ALL']
        role_counts = {role: 0 for role in roles}
        
        for player in team_output['main_team']:
            role = player['role']
            if role in role_counts:
                role_counts[role] += 1
            else:
                # Handle any unexpected roles
                role_counts['ALL'] += 1
            
        fig = go.Figure(data=[
            go.Bar(
                x=list(role_counts.keys()),
                y=list(role_counts.values()),
                text=list(role_counts.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Team Composition by Role',
            xaxis_title='Role',
            yaxis_title='Number of Players',
            showlegend=False
        )
        
        return fig
        
    def plot_player_points(self, team_output):
        """Create player points visualization"""
        players = [p['name'] for p in team_output['main_team']]
        points = [p['predicted_points'] for p in team_output['main_team']]
        
        fig = px.bar(
            x=players,
            y=points,
            title='Predicted Points by Player',
            labels={'x': 'Player', 'y': 'Predicted Points'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
        
    def select_captain_vice_captain(self, team_output):
        """Select captain and vice-captain based on predicted points"""
        # Sort players by predicted points in descending order
        sorted_players = sorted(
            team_output['main_team'], 
            key=lambda x: x['predicted_points'], 
            reverse=True
        )
        
        # Select captain (2x points) and vice-captain (1.5x points)
        captain = sorted_players[0]
        vice_captain = sorted_players[1]
        
        # Update total points to include captain and vice-captain bonuses
        captain_bonus = captain['predicted_points']  # Additional 100% points
        vice_captain_bonus = vice_captain['predicted_points'] * 0.5  # Additional 50% points
        
        team_output['total_predicted_points'] += captain_bonus + vice_captain_bonus
        
        # Add captain and vice-captain information to the output
        team_output['captain'] = captain
        team_output['vice_captain'] = vice_captain
        
        return team_output
        
    def run(self):
        """Run the Streamlit app"""
        st.title('Dream11 Team Predictor')
        st.write('Predict the best Dream11 team for any IPL match!')
        
        # Sidebar for team selection
        st.sidebar.header('Match Details')
        team1 = st.sidebar.selectbox('Select Team 1', ['CHE', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR'])
        team2 = st.sidebar.selectbox('Select Team 2', ['CHE', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR'])
        
        if team1 == team2:
            st.error('Please select different teams!')
            return
            
        # Load and process data
        features, target = self.load_data()
        
        # Train models
        models = self.train_models(features, target)
        
        # Predict team
        team_output = self.predict_team(team1, team2, models)
        
        # Select captain and vice-captain
        team_output = self.select_captain_vice_captain(team_output)
        
        # Display results
        st.header('Predicted Dream11 Team')
        
        # Captain and Vice-Captain
        st.subheader('Captain and Vice-Captain')
        captain = team_output['captain']
        vice_captain = team_output['vice_captain']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                'Captain', 
                f"{captain['name']} ({captain['role']})", 
                f"{captain['predicted_points']:.1f} pts"
            )
        with col2:
            st.metric(
                'Vice-Captain', 
                f"{vice_captain['name']} ({vice_captain['role']})", 
                f"{vice_captain['predicted_points']:.1f} pts"
            )
        
        # Main team
        st.subheader('Main Team')
        main_team_df = pd.DataFrame(team_output['main_team'])
        st.dataframe(main_team_df)
        
        # Backup team
        st.subheader('Backup Team')
        backup_team_df = pd.DataFrame(team_output['backup_team'])
        st.dataframe(backup_team_df)
        
        # Team statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Total Predicted Points', f"{team_output['total_predicted_points']:.2f}")
        with col2:
            st.metric('Total Credits Used', f"{team_output['total_credits']:.1f}")
            
        # Visualizations
        st.plotly_chart(self.plot_team_composition(team_output))
        st.plotly_chart(self.plot_player_points(team_output))
        
        # Download button for team
        team_df = pd.concat([main_team_df, backup_team_df])
        csv = team_df.to_csv(index=False)
        st.download_button(
            label="Download Team as CSV",
            data=csv,
            file_name=f"dream11_team_{team1}_vs_{team2}.csv",
            mime="text/csv"
        )

if __name__ == '__main__':
    app = Dream11App()
    app.run() 