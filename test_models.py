import pandas as pd
import numpy as np
import os
import sys
import time
import argparse
from pitch_specific_models import PitchSpecificModelTrainer
from model_evaluator import ModelEvaluator
from model_integration import Dream11ModelIntegrator

# Attempt to import tabulate for pretty printing, but proceed without if unavailable
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("'tabulate' library not found. Output will be less formatted.")
    print("Install using: pip install tabulate")

def print_table(data, headers='firstrow', tablefmt='grid', **kwargs):
    """Helper function to print tables using tabulate if available, otherwise basic print."""
    if HAS_TABULATE:
        print(tabulate(data, headers=headers, tablefmt=tablefmt, **kwargs))
    elif isinstance(data, pd.DataFrame):
        print(data.to_string(**kwargs))
    elif isinstance(data, list) and isinstance(data[0], list):
        # Basic print for list of lists
        if headers == 'firstrow':
             print("\t".join(map(str, data[0])))
             print("---" * len(data[0]))
             for row in data[1:]:
                 print("\t".join(map(str, row)))
        else:
             print("\t".join(map(str, headers)))
             print("---" * len(headers))
             for row in data:
                 print("\t".join(map(str, row)))
    else:
        print(data) # Fallback to basic print

def train_and_evaluate_models():
    """Train and evaluate pitch-specific and fallback models"""
    print("\n=== Training Pitch-Specific and Fallback Models ===")
    
    # Initialize model trainer
    trainer = PitchSpecificModelTrainer()
    
    # Start timer
    start_time = time.time()
    
    # Train models (including fallback)
    results = trainer.train_models()
    
    # End timer
    training_time = time.time() - start_time
    
    # Print training results
    if results:
        print("\n--- Training Results ---")
        rows = []
        headers = ["Model Key", "Model Type", "Train R²", "Train MAE", "Test R²", "Test MAE", "Features"]
        
        for model_key, metrics in results.items():
            rows.append([
                model_key.title(),
                metrics['model_type'],
                f"{metrics['train_r2']:.4f}",
                f"{metrics['train_mae']:.2f}",
                f"{metrics['test_r2']:.4f}",
                f"{metrics['test_mae']:.2f}",
                metrics['num_features'] # Assuming num_features is available
            ])
        
        print_table(rows, headers=headers, tablefmt="pretty")
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Print top features/coefficients for each model
        for model_key, metrics in results.items():
            if metrics['top_features']:
                print(f"\nTop 5 Features/Coefficients for {model_key.title()} Model:")
                top_5 = metrics['top_features'][:5]
                feature_rows = [[i+1, feature, f"{importance:.4f}"] for i, (feature, importance) in enumerate(top_5)]
                print_table(feature_rows, headers=["Rank", "Feature/Coefficient", "Importance/Value"], tablefmt="pretty")
    else:
        print("Model training failed or produced no results.")
        
def evaluate_models():
    """Evaluate and compare models (including fallback)"""
    print("\n=== Evaluating Models (Including Fallback) ===")
    
    # Initialize model evaluator
    evaluator = ModelEvaluator()
    
    # Compare models (this runs evaluation and generates plots/metrics)
    comparison_df = evaluator.compare_models()
    
    if comparison_df is not None and not comparison_df.empty:
        print("\n--- Model Comparison Metrics ---")
        print_table(comparison_df, headers="keys", tablefmt="pretty", showindex=False)
        print(f"\nEvaluation plots (errors, actual vs predicted, comparisons) saved in '{evaluator.results_dir}'")
    else:
        print("Model comparison failed or produced no results.")
        
    # Run cross-model evaluation
    cross_eval_df = evaluator.run_cross_model_evaluation()
    
    if cross_eval_df is not None and not cross_eval_df.empty:
        print("\n--- Cross-Model Evaluation (R² Scores) ---")
        try:
            pivot_r2 = cross_eval_df.pivot(
                index='model_trained_on', 
                columns='evaluated_on_data', 
                values='r2_score'
            )
            print_table(pivot_r2, headers="keys", tablefmt="pretty")
        except KeyError:
            print("Could not pivot R2 data. Raw cross-eval data:")
            print_table(cross_eval_df, headers="keys", tablefmt="pretty", showindex=False)
            
        print("\n--- Cross-Model Evaluation (MAE) ---")
        try:
             pivot_mae = cross_eval_df.pivot(
                 index='model_trained_on', 
                 columns='evaluated_on_data', 
                 values='mae'
             )
             print_table(pivot_mae, headers="keys", tablefmt="pretty")
        except KeyError:
             print("Could not pivot MAE data. Raw cross-eval data printed above.")

        print(f"\nCross-evaluation heatmaps saved in '{evaluator.results_dir}'")
    else:
        print("Cross-model evaluation failed or produced no results.")
        
def test_model_integration():
    """Test model integration (prediction with fallback) with Dream11 app"""
    print("\n=== Testing Model Integration (with Fallback) ===")
    
    # Initialize integrator (will load or train models)
    try:
         integrator = Dream11ModelIntegrator()
         print("Model Integrator initialized.")
    except Exception as e:
         print(f"Error initializing Model Integrator: {e}")
         print("Cannot proceed with integration test.")
         return

    # Create sample player data
    sample_players = pd.DataFrame({
        'Player Name': ['Virat Kohli', 'Rohit Sharma', 'Jasprit Bumrah', 'MS Dhoni', 'Hardik Pandya',
                      'Ravindra Jadeja', 'KL Rahul', 'Mohammed Shami', 'Rishabh Pant', 'Shreyas Iyer',
                      'Yuzvendra Chahal', 'Trent Boult', 'Glenn Maxwell', 'David Warner', 'Jos Buttler'],
        'Team': ['RCB', 'MI', 'MI', 'CSK', 'MI', 'CSK', 'PBKS', 'GT', 'DC', 'KKR',
                'RR', 'RR', 'RCB', 'DC', 'RR'],
        'Player Type': ['BAT', 'BAT', 'BOWL', 'WK', 'AR', 'AR', 'BAT', 'BOWL', 'WK', 'BAT',
                      'BOWL', 'BOWL', 'AR', 'BAT', 'WK'],
        'Credits': [10.0, 9.5, 9.0, 8.5, 9.5, 9.0, 8.5, 8.0, 8.5, 8.0,
                   8.0, 8.5, 9.0, 9.5, 9.0]
    })
    
    # --- Test Case 1: Balanced Pitch (Should use balanced model) ---
    print("\n--- Test Case 1: Balanced Pitch ---")
    match_info_balanced = {
        'venue': 'MA Chidambaram Stadium', 'home_team': 'CSK', 
        'away_team': 'MI', 'pitch_type': 'balanced'
    }
    predicted_df_bal = integrator.predict_player_points(sample_players.copy(), match_info_balanced)
    if predicted_df_bal is not None and 'predicted_points' in predicted_df_bal.columns and not predicted_df_bal['predicted_points'].isnull().all():
        print("Top 10 Predicted Points (Balanced):")
        print_table(predicted_df_bal.nlargest(10, 'predicted_points')[['Player Name', 'Player Type', 'Team', 'predicted_points']], headers="keys", tablefmt="pretty", showindex=False, floatfmt=".2f")
    else:
        print("Prediction failed for Balanced pitch.")

    # --- Test Case 2: Batting Pitch (Should use batting model) ---
    print("\n--- Test Case 2: Batting-Friendly Pitch ---")
    match_info_batting = {
        'venue': 'Wankhede Stadium', 'home_team': 'MI', 
        'away_team': 'RCB', 'pitch_type': 'batting_friendly'
    }
    predicted_df_bat = integrator.predict_player_points(sample_players.copy(), match_info_batting)
    if predicted_df_bat is not None and 'predicted_points' in predicted_df_bat.columns and not predicted_df_bat['predicted_points'].isnull().all():
        print("Top 10 Predicted Points (Batting-Friendly):")
        print_table(predicted_df_bat.nlargest(10, 'predicted_points')[['Player Name', 'Player Type', 'Team', 'predicted_points']], headers="keys", tablefmt="pretty", showindex=False, floatfmt=".2f")
    else:
        print("Prediction failed for Batting-Friendly pitch.")

    # --- Test Case 3: Unknown Pitch Type (Should use Fallback) ---
    # Simulate scenario where prediction might fail or pitch type is unknown
    print("\n--- Test Case 3: Unknown Pitch Type (Triggering Fallback) ---")
    match_info_unknown = {
        'venue': 'Unknown Venue', 'home_team': 'SRH', 
        'away_team': 'LSG', 'pitch_type': 'unknown_pitch' # Force fallback via invalid type
    }
    # The integrator should handle the invalid pitch type and use fallback
    predicted_df_unknown = integrator.predict_player_points(sample_players.copy(), match_info_unknown)
    if predicted_df_unknown is not None and 'predicted_points' in predicted_df_unknown.columns and not predicted_df_unknown['predicted_points'].isnull().all():
        print("Top 10 Predicted Points (Fallback Used for Unknown Pitch):")
        print_table(predicted_df_unknown.nlargest(10, 'predicted_points')[['Player Name', 'Player Type', 'Team', 'predicted_points']], headers="keys", tablefmt="pretty", showindex=False, floatfmt=".2f")
    else:
        print("Prediction failed even with fallback for Unknown pitch.")

    # --- Test Case 4: Test C/VC and Team Balance (Example with Balanced) ---
    print("\n--- Test Case 4: Team Balance & C/VC (Balanced Pitch Example) ---")
    if predicted_df_bal is not None and 'predicted_points' in predicted_df_bal.columns and not predicted_df_bal['predicted_points'].isnull().all():
        # Get recommended team balance
        balance = integrator.identify_team_balance_requirements(match_info_balanced)
        if balance:
            balance_data = [[role, values['min'], values['max'], values['optimal']] for role, values in balance.items()]
            print("\nRecommended Team Balance (Balanced Pitch):")
            print_table(balance_data, headers=["Role", "Min", "Max", "Optimal"], tablefmt="pretty")
        
        # Test captain/vice-captain suggestion using the predicted points
        # Assuming we select the top 11 players based on prediction for C/VC choice
        suggested_team_df = predicted_df_bal.nlargest(11, 'predicted_points') 
        captain, vice_captain = integrator.suggest_captain_vice_captain(suggested_team_df, match_info_balanced)
        
        cvc_data = []
        if captain is not None:
            cvc_data.append(["Captain", captain.get('Player Name', 'N/A'), captain.get('Player Type', 'N/A'), f"{captain.get('predicted_points', 0):.2f}"])
        if vice_captain is not None:
             cvc_data.append(["Vice-Captain", vice_captain.get('Player Name', 'N/A'), vice_captain.get('Player Type', 'N/A'), f"{vice_captain.get('predicted_points', 0):.2f}"])
        
        if cvc_data:
             print("\nRecommended Captain and Vice-Captain (Balanced Pitch):")
             print_table(cvc_data, headers=["Role", "Player", "Player Type", "Predicted Points"], tablefmt="pretty")
        else:
             print("\nCould not recommend Captain/Vice-Captain.")
    else:
         print("Skipping Team Balance & C/VC test due to previous prediction failure.")

def main():
    parser = argparse.ArgumentParser(description='Test and evaluate Dream11 prediction models with fallback')
    parser.add_argument('--train', action='store_true', help='Train new models (pitch-specific and fallback)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models (including fallback and cross-eval)')
    parser.add_argument('--test-integration', action='store_true', help='Test model integration prediction logic (with fallback)')
    parser.add_argument('--all', action='store_true', help='Run all steps: train, evaluate, and test integration')
    
    args = parser.parse_args()
    
    # Default to running all if no specific flag is given
    if not any([args.train, args.evaluate, args.test_integration, args.all]):
        print("No specific step selected, running all steps (--all)")
        args.all = True
        
    # Ensure dependent files exist before proceeding
    required_files = ["pitch_specific_models.py", "model_evaluator.py", "model_integration.py", "feature_engineering.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
         print(f"Error: Missing required script files: {', '.join(missing_files)}")
         print("Cannot proceed.")
         sys.exit(1)

    if args.train or args.all:
        train_and_evaluate_models()
        
    if args.evaluate or args.all:
        evaluate_models()
        
    if args.test_integration or args.all:
        test_model_integration()
        
    print("\nTesting script finished.")

if __name__ == "__main__":
    main() 