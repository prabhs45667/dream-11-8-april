import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from quick_implementation_plan import Dream11Predictor

def main():
    st.set_page_config(page_title="Dream11 Team Predictor", layout="wide")
    st.title("Dream11 IPL Team Predictor")
    st.markdown("### Predict the best Dream11 team for any IPL match")
    
    # Initialize predictor
    @st.cache_resource
    def load_predictor():
        try:
            predictor = Dream11Predictor()
            
            # Check data directory
            if not os.path.exists('dataset'):
                st.error("Error: 'dataset' directory not found!")
                st.info("Please ensure you have the required dataset files in a 'dataset' folder")
                return None
                
            # Check for required files
            required_files = ['IPL_Ball_by_Ball_2008_2022.csv', 
                             'SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv']
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join('dataset', file)):
                    missing_files.append(file)
            
            if missing_files:
                st.error(f"Error: The following required files are missing: {', '.join(missing_files)}")
                st.info("Please ensure all required data files are in the 'dataset' folder")
                return None
            
            # Check if models already exist
            if os.path.exists('models/scaler.joblib'):
                st.info("Loading saved models...")
                try:
                    predictor.load_models()
                except Exception as e:
                    st.error(f"Error loading models: {e}")
                    st.info("Creating new models...")
                    with st.spinner("Loading and preprocessing data..."):
                        predictor.load_and_preprocess_data()
                    
                    with st.spinner("Training prediction models (this may take a few minutes)..."):
                        predictor.train_models()
                        predictor.save_models()
                        predictor.plot_feature_importance()
            else:
                with st.spinner("First-time setup: Loading and preprocessing data..."):
                    predictor.load_and_preprocess_data()
                
                with st.spinner("Training prediction models (this may take a few minutes)..."):
                    predictor.train_models()
                    predictor.save_models()
                    predictor.plot_feature_importance()
            
            return predictor
        except Exception as e:
            st.error(f"Error initializing predictor: {e}")
            return None
    
    # Load the predictor with error handling
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Could not initialize the prediction model. Please check the error messages above.")
        return
    
    # Team selection sidebar
    st.sidebar.header("Select Match")
    teams = ['CSK', 'MI', 'RCB', 'KKR', 'PBKS', 'DC', 'RR', 'SRH', 'GT', 'LSG']
    team1 = st.sidebar.selectbox("Team 1", teams, index=0)
    team2 = st.sidebar.selectbox("Team 2", teams, index=1)
    
    # Check if teams are different
    if team1 == team2:
        st.error("Please select different teams!")
        return
    
    # Button to predict team
    if st.sidebar.button("Predict Dream11 Team"):
        try:
            with st.spinner(f"Predicting Dream11 team for {team1} vs {team2}..."):
                predicted_team = predictor.predict_team(team1, team2)
                
                if predicted_team is None:
                    st.error("Error predicting team. Check if squad data is available.")
                    return
                
                # Display the predicted team
                st.subheader(f"Predicted Dream11 Team for {team1} vs {team2}")
                
                # Find captain and vice-captain
                captain_rows = predicted_team[predicted_team['role'].str.contains('(C)')].index
                vice_captain_rows = predicted_team[predicted_team['role'].str.contains('(VC)')].index
                
                if len(captain_rows) > 0 and len(vice_captain_rows) > 0:
                    captain = predicted_team.loc[captain_rows[0]]
                    vice_captain = predicted_team.loc[vice_captain_rows[0]]
                    
                    # Display captain and vice-captain
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Captain")
                        st.markdown(f"**{captain['player']}** ({captain['team']})")
                        st.markdown(f"Role: {captain['role'].replace(' (C)', '')}")
                        st.markdown(f"Predicted Points: {captain['predicted_points']:.2f}")
                        
                    with col2:
                        st.markdown("### Vice-Captain")
                        st.markdown(f"**{vice_captain['player']}** ({vice_captain['team']})")
                        st.markdown(f"Role: {vice_captain['role'].replace(' (VC)', '')}")
                        st.markdown(f"Predicted Points: {vice_captain['predicted_points']:.2f}")
                else:
                    st.warning("Could not find captain and vice-captain in the predicted team.")
                
                # Team composition
                st.subheader("Team Composition")
                role_counts = predicted_team['role'].str.replace(' \\(C\\)| \\(VC\\)', '', regex=True).value_counts()
                
                # Create a pie chart for team composition
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = sns.color_palette('viridis', len(role_counts))
                ax.pie(role_counts, labels=role_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')
                st.pyplot(fig)
                
                # Team by team
                team1_players = predicted_team[predicted_team['team'] == team1]
                team2_players = predicted_team[predicted_team['team'] == team2]
                
                st.markdown(f"**{team1}**: {len(team1_players)} players")
                st.markdown(f"**{team2}**: {len(team2_players)} players")
                
                # Full team table
                st.subheader("Full Team")
                # Clean up roles
                display_team = predicted_team.copy()
                display_team['role'] = display_team['role'].str.replace(' \\(C\\)| \\(VC\\)', '', regex=True)
                
                # Add columns to indicate captain and vice-captain
                display_team['Selection'] = ''
                if len(captain_rows) > 0:
                    display_team.loc[captain_rows[0], 'Selection'] = 'Captain'
                if len(vice_captain_rows) > 0:
                    display_team.loc[vice_captain_rows[0], 'Selection'] = 'Vice-Captain'
                
                # Sort by predicted points
                display_team = display_team.sort_values('predicted_points', ascending=False)
                
                # Format points for display
                display_team['predicted_points'] = display_team['predicted_points'].round(2)
                
                # Reset index
                display_team = display_team.reset_index(drop=True)
                
                # Display dataframe
                st.dataframe(display_team[['player', 'team', 'role', 'credits', 'predicted_points', 'Selection']])
                
                # Download as CSV
                csv = display_team.to_csv(index=False)
                st.download_button(
                    label="Download Team as CSV",
                    data=csv,
                    file_name=f"dream11_team_{team1}_vs_{team2}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error while predicting team: {e}")
    
    # Feature importance section
    st.sidebar.header("Model Insights")
    if os.path.exists('feature_importance.png') and st.sidebar.checkbox("Show Feature Importance"):
        st.subheader("Feature Importance")
        st.image('feature_importance.png')
        st.markdown("This plot shows the most important features used by the model to predict player performance.")

if __name__ == "__main__":
    main() 