import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pitch_specific_models import PitchSpecificModelTrainer # Assumes this file exists and is updated
from feature_engineering import FeatureEngineer # Assumes this file exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluator')

class ModelEvaluator:
    """
    Evaluates and compares different models (including fallback) 
    for fantasy cricket point prediction
    """
    
    def __init__(self, data_dir="dataset", models_dir="models", results_dir="results"):
        """
        Initialize the model evaluator
        
        Args:
            data_dir (str): Directory containing data files
            models_dir (str): Directory containing model files
            results_dir (str): Directory to save evaluation results
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize model trainer (needed to prepare data and load models)
        self.model_trainer = PitchSpecificModelTrainer(data_dir=data_dir, models_dir=models_dir)
        
        # Feature engineer might be needed if re-running feature prep
        self.feature_engineer = FeatureEngineer(data_dir=data_dir) 
        
    def prepare_evaluation_data(self):
        """
        Prepare data for model evaluation (uses trainer's methods)
        
        Returns:
            dict: Dictionary of pitch-specific DataFrames, prepared features/targets
        """
        # Load training data (which gets split internally)
        data_by_pitch_type = self.model_trainer.load_training_data()
        if data_by_pitch_type is None:
             logger.error("Failed to load data for evaluation.")
             return None, None

        # Prepare features and targets (including for fallback model)
        features_targets = self.model_trainer.prepare_features_targets(data_by_pitch_type)
        if features_targets is None:
             logger.error("Failed to prepare features/targets for evaluation.")
             return data_by_pitch_type, None

        return data_by_pitch_type, features_targets
        
    def evaluate_models(self, features_targets=None, models=None):
        """
        Evaluate models (including fallback) on test data
        
        Args:
            features_targets (dict): Prepared features/targets for evaluation
            models (dict): Dictionary of models by pitch type + fallback
            
        Returns:
            dict: Evaluation results (R2, MAE, RMSE) for each model
        """
        # Prepare evaluation data if not provided
        if features_targets is None:
            _, features_targets = self.prepare_evaluation_data() # Ignore raw data return
            
        if features_targets is None:
            logger.error("No evaluation data (features/targets) available")
            return None
            
        # Load models if not provided
        if models is None:
            logger.info("Models not provided, attempting to load from disk...")
            models_loaded = self.model_trainer.load_models()
            if not models_loaded:
                 logger.warning("Failed to load models from disk. Evaluation cannot proceed without models.")
                 # Optionally, could trigger training here, but maybe evaluation shouldn't train.
                 return None
            models = self.model_trainer.models # Use the loaded models
                
        results = {}
        
        # Evaluate models for each key (pitch type + fallback)
        for model_key, ft_dict in features_targets.items():
            logger.info(f"Evaluating model for '{model_key}' type")
            
            model = models.get(model_key)
            
            if model is None:
                logger.warning(f"No model available for '{model_key}' type. Skipping evaluation.")
                continue
                
            # Check if test data exists for this key
            if 'X_test' not in ft_dict or 'y_test' not in ft_dict:
                 logger.warning(f"Test data missing for '{model_key}'. Skipping evaluation.")
                 continue

            X_test = ft_dict['X_test']
            y_test = ft_dict['y_test']

            # Ensure X_test has the same columns as the model expects (if possible)
            # This is crucial if features were generated differently or columns dropped
            model_features = None
            if hasattr(model, 'feature_names_in_'):
                 model_features = model.feature_names_in_
            
            if model_features is not None:
                 current_cols = X_test.columns
                 # Add missing columns to X_test with 0
                 for feature in model_features:
                      if feature not in current_cols:
                           X_test[feature] = 0
                 # Reorder and select columns to match model
                 try:
                     X_test = X_test[model_features]
                 except KeyError as e:
                      logger.error(f"Column mismatch error for model '{model_key}': {e}. Skipping evaluation.")
                      continue
            else:
                 logger.warning(f"Cannot verify feature names for '{model_key}' model. Assuming consistency.")

            # Make predictions
            try:
                 y_pred = model.predict(X_test)
            except Exception as pred_err:
                 logger.error(f"Error predicting with '{model_key}' model: {pred_err}")
                 continue # Skip this model if prediction fails

            # Calculate metrics
            try:
                 r2 = r2_score(y_test, y_pred)
                 mae = mean_absolute_error(y_test, y_pred)
                 rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            except Exception as metric_err:
                 logger.error(f"Error calculating metrics for '{model_key}': {metric_err}")
                 r2, mae, rmse = np.nan, np.nan, np.nan # Assign NaN if metrics fail
            
            # Store results
            results[model_key] = {
                'model_type': model.__class__.__name__,
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'y_test': y_test, # Storing actuals/predictions for plotting
                'y_pred': y_pred
            }
            
            logger.info(f"{model_key} evaluation: R² = {r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
            
        # Save evaluation results to file
        self._save_evaluation_results(results)
        
        return results
        
    def _save_evaluation_results(self, results):
        """
        Save evaluation metrics to a CSV file
        
        Args:
            results (dict): Evaluation results containing metrics
        """
        try:
            metrics_data = []
            for model_key, result in results.items():
                # Ensure metrics exist before trying to append
                if 'r2_score' in result:
                    metrics_data.append({
                        'model_key': model_key,
                        'model_type': result['model_type'],
                        'r2_score': result['r2_score'],
                        'mae': result['mae'],
                        'rmse': result['rmse']
                    })
                
            if not metrics_data:
                 logger.warning("No valid metrics found to save.")
                 return

            metrics_df = pd.DataFrame(metrics_data)
            metrics_file = os.path.join(self.results_dir, 'model_metrics.csv')
            metrics_df.to_csv(metrics_file, index=False)
            logger.info(f"Saved evaluation metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def plot_prediction_errors(self, evaluation_results):
        """
        Plot prediction errors (residuals) distribution for each model
        
        Args:
            evaluation_results (dict): Model evaluation results from evaluate_models
        """
        if not evaluation_results:
             logger.warning("No evaluation results provided for plotting errors.")
             return
        try:
            num_plots = len(evaluation_results)
            ncols = 2
            nrows = (num_plots + ncols - 1) // ncols # Calculate needed rows
            plt.figure(figsize=(7 * ncols, 5 * nrows))
            plot_index = 1
            
            for model_key, result in evaluation_results.items():
                 if 'y_test' not in result or 'y_pred' not in result:
                      logger.warning(f"Missing y_test or y_pred for '{model_key}'. Skipping error plot.")
                      continue
                 
                 plt.subplot(nrows, ncols, plot_index)
                 y_test = result['y_test']
                 y_pred = result['y_pred']
                 errors = y_test - y_pred
                 
                 sns.histplot(errors, kde=True)
                 plt.title(f"{model_key.title()} Model Prediction Errors")
                 plt.xlabel("Error (Actual - Predicted)")
                 plt.ylabel("Frequency")
                 
                 mean_error = np.mean(errors)
                 std_error = np.std(errors)
                 plt.axvline(mean_error, color='red', linestyle='--')
                 plt.text(0.05, 0.95, f"Mean: {mean_error:.2f}\nStd: {std_error:.2f}",
                          transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
                 plot_index += 1

            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, 'prediction_errors.png')
            plt.savefig(plot_file)
            logger.info(f"Saved prediction errors plot to {plot_file}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting prediction errors: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def plot_actual_vs_predicted(self, evaluation_results):
        """
        Plot actual vs predicted values for each model
        
        Args:
            evaluation_results (dict): Model evaluation results from evaluate_models
        """
        if not evaluation_results:
             logger.warning("No evaluation results provided for plotting actual vs predicted.")
             return
        try:
            num_plots = len(evaluation_results)
            ncols = 2
            nrows = (num_plots + ncols - 1) // ncols
            plt.figure(figsize=(7 * ncols, 6 * nrows))
            plot_index = 1

            for model_key, result in evaluation_results.items():
                 if 'y_test' not in result or 'y_pred' not in result:
                      logger.warning(f"Missing y_test or y_pred for '{model_key}'. Skipping actual vs predicted plot.")
                      continue

                 plt.subplot(nrows, ncols, plot_index)
                 y_test = result['y_test']
                 y_pred = result['y_pred']
                 
                 plt.scatter(y_test, y_pred, alpha=0.5)
                 max_val = max(np.max(y_test), np.max(y_pred)) * 1.1
                 min_val = min(np.min(y_test), np.min(y_pred)) * 0.9
                 plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                 
                 plt.title(f"{model_key.title()} Model: Actual vs Predicted")
                 plt.xlabel("Actual Points")
                 plt.ylabel("Predicted Points")
                 plt.xlim(min_val, max_val)
                 plt.ylim(min_val, max_val)
                 plt.grid(True, linestyle='--', alpha=0.6)
                 
                 # Add metrics annotations
                 r2 = result.get('r2_score', np.nan)
                 mae = result.get('mae', np.nan)
                 rmse = result.get('rmse', np.nan)
                 plt.text(0.05, 0.95, f"R²: {r2:.4f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}",
                          transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
                 plot_index += 1
                 
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, 'actual_vs_predicted.png')
            plt.savefig(plot_file)
            logger.info(f"Saved actual vs predicted plot to {plot_file}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting actual vs predicted: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def compare_models(self):
        """
        Run evaluation and generate comparison plots/results
        
        Returns:
            pd.DataFrame: Comparison results (metrics for each model)
        """
        try:
            logger.info("Starting model comparison...")
            # Evaluate models (this also prepares data and loads models if needed)
            evaluation_results = self.evaluate_models()
            
            if evaluation_results is None:
                logger.error("Model evaluation failed. Cannot generate comparison.")
                return None
                
            # Plot evaluation diagnostics
            self.plot_prediction_errors(evaluation_results)
            self.plot_actual_vs_predicted(evaluation_results)
            
            # Create comparison DataFrame from metrics
            comparison_data = []
            for model_key, result in evaluation_results.items():
                 if 'r2_score' in result:
                     comparison_data.append({
                         'model_key': model_key,
                         'model_type': result['model_type'],
                         'r2_score': result['r2_score'],
                         'mae': result['mae'],
                         'rmse': result['rmse']
                     })
            
            if not comparison_data:
                 logger.warning("No valid results to compare.")
                 return None

            comparison_df = pd.DataFrame(comparison_data)
            
            # Plot comparison metrics (e.g., bar charts)
            self._plot_comparison_metrics(comparison_df)
            
            logger.info("Model comparison finished.")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _plot_comparison_metrics(self, comparison_df):
        """
        Plot comparison metrics (R2, MAE) for all evaluated models
        
        Args:
            comparison_df (pd.DataFrame): Model comparison DataFrame from compare_models
        """
        if comparison_df.empty:
             logger.warning("Comparison DataFrame is empty, skipping metric plots.")
             return
        try:
            # Plot R² Score
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model_key', y='r2_score', data=comparison_df, palette="viridis")
            plt.title("Model Comparison: R² Score")
            plt.xlabel("Model Type")
            plt.ylabel("R² Score")
            plt.ylim(bottom=max(0, comparison_df['r2_score'].min() - 0.1), top=min(1, comparison_df['r2_score'].max() + 0.1)) # Adjust y-lim
            for i, v in enumerate(comparison_df['r2_score']):
                plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=9)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, 'r2_comparison.png')
            plt.savefig(plot_file)
            logger.info(f"Saved R² comparison plot to {plot_file}")
            plt.close()
            
            # Plot MAE
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model_key', y='mae', data=comparison_df, palette="magma")
            plt.title("Model Comparison: Mean Absolute Error (MAE)")
            plt.xlabel("Model Type")
            plt.ylabel("MAE")
            plt.ylim(bottom=0, top=comparison_df['mae'].max() * 1.1) # Adjust y-lim
            for i, v in enumerate(comparison_df['mae']):
                plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=9)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, 'mae_comparison.png')
            plt.savefig(plot_file)
            logger.info(f"Saved MAE comparison plot to {plot_file}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting comparison metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def run_cross_model_evaluation(self):
        """
        Evaluate each trained model on data prepared for *other* pitch types
        to assess generalization or specialization.
        
        Returns:
            pd.DataFrame: Cross-evaluation results (metrics for each model on each dataset)
        """
        try:
            logger.info("Starting cross-model evaluation...")
            # Prepare evaluation data (features/targets for all pitch types + fallback)
            _, features_targets = self.prepare_evaluation_data()
            if features_targets is None:
                logger.error("Cannot perform cross-evaluation without prepared data.")
                return None
                
            # Load models (ensure all, including fallback, are loaded)
            models_loaded = self.model_trainer.load_models()
            if not models_loaded:
                 logger.warning("Models failed to load. Cross-evaluation requires pre-trained models.")
                 return None
            models = self.model_trainer.models

            # Cross-evaluation results storage
            cross_eval_results = []
            
            # Iterate through each MODEL (trained on model_key data)
            for model_key, model in models.items():
                if model is None:
                    logger.warning(f"Skipping cross-evaluation for missing model: '{model_key}'")
                    continue
                    
                # Evaluate this model on each DATASET (prepared for data_key)
                for data_key, ft_dict in features_targets.items():
                    # Don't evaluate fallback model on pitch-specific feature sets if they differ significantly
                    if model_key == 'fallback' and data_key != 'fallback':
                         # Fallback uses simpler features, can't directly use enhanced features
                         logger.debug(f"Skipping evaluation of fallback model on {data_key} data (feature mismatch).")
                         continue 
                    # Don't evaluate pitch-specific models on fallback feature set
                    if model_key != 'fallback' and data_key == 'fallback':
                         logger.debug(f"Skipping evaluation of {model_key} model on fallback data (feature mismatch).")
                         continue

                    if 'X_test' not in ft_dict or 'y_test' not in ft_dict:
                         logger.warning(f"Test data missing for '{data_key}'. Skipping this cross-evaluation pair.")
                         continue

                    X_test = ft_dict['X_test'].copy() # Use copy to avoid modifying original dict data
                    y_test = ft_dict['y_test']
                    
                    # Ensure X_test columns match the model's expected features
                    model_features = None
                    if hasattr(model, 'feature_names_in_'):
                         model_features = model.feature_names_in_
                    
                    if model_features is not None:
                         # Add missing columns with 0
                         for feature in model_features:
                              if feature not in X_test.columns:
                                   X_test[feature] = 0
                         # Select and reorder
                         try:
                             X_test = X_test[model_features]
                         except KeyError as e:
                              logger.error(f"Column mismatch evaluating model '{model_key}' on data '{data_key}': {e}. Skipping.")
                              continue
                    else:
                         # Cannot reliably align columns for models like LinearRegression without stored names
                         # Assume consistency if model_key == data_key, skip otherwise for safety?
                         if model_key != data_key:
                             logger.warning(f"Cannot verify features for model '{model_key}' on data '{data_key}'. Skipping cross eval pair.")
                             continue

                    # Make predictions
                    try:
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        cross_eval_results.append({
                            'model_trained_on': model_key,
                            'evaluated_on_data': data_key,
                            'model_type': model.__class__.__name__,
                            'r2_score': r2,
                            'mae': mae,
                            'rmse': rmse
                        })
                        logger.info(f"Model '{model_key}' on '{data_key}' data: R²={r2:.4f}, MAE={mae:.2f}")
                    except Exception as e:
                        logger.error(f"Error during cross-evaluation (Model: {model_key}, Data: {data_key}): {e}")
                        
            if not cross_eval_results:
                 logger.warning("No cross-evaluation results generated.")
                 return None

            # Create DataFrame and save
            cross_eval_df = pd.DataFrame(cross_eval_results)
            results_file = os.path.join(self.results_dir, 'cross_model_evaluation.csv')
            cross_eval_df.to_csv(results_file, index=False)
            logger.info(f"Saved cross-evaluation results to {results_file}")
            
            # Plot cross-evaluation heatmaps
            self._plot_cross_evaluation(cross_eval_df)
            
            logger.info("Cross-model evaluation finished.")
            return cross_eval_df
            
        except Exception as e:
            logger.error(f"Error in run_cross_model_evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _plot_cross_evaluation(self, cross_eval_df):
        """
        Plot cross-evaluation results as heatmaps for R2 and MAE
        
        Args:
            cross_eval_df (pd.DataFrame): Cross-evaluation results DataFrame
        """
        if cross_eval_df.empty:
             logger.warning("Cross-evaluation DataFrame is empty, skipping heatmaps.")
             return
        try:
            # --- R² Score Heatmap ---
            plt.figure(figsize=(10, 8))
            try:
                 heatmap_data_r2 = cross_eval_df.pivot(
                     index='model_trained_on', 
                     columns='evaluated_on_data',
                     values='r2_score'
                 )
                 sns.heatmap(heatmap_data_r2, annot=True, fmt='.3f', cmap='viridis', 
                             linewidths=.5, linecolor='black', vmin=heatmap_data_r2.min().min() - 0.1, vmax=1.0)
                 plt.title("Cross-Evaluation: R² Score (Model vs. Data Type)")
                 plt.ylabel("Model Trained On")
                 plt.xlabel("Evaluated On Data Type")
                 plt.tight_layout()
                 plot_file_r2 = os.path.join(self.results_dir, 'cross_eval_r2_heatmap.png')
                 plt.savefig(plot_file_r2)
                 logger.info(f"Saved cross-evaluation R² heatmap to {plot_file_r2}")
            except KeyError as e:
                 logger.error(f"Error pivoting/plotting R2 heatmap: Missing key {e}")
            plt.close()
            
            # --- MAE Heatmap ---
            plt.figure(figsize=(10, 8))
            try:
                 heatmap_data_mae = cross_eval_df.pivot(
                     index='model_trained_on',
                     columns='evaluated_on_data',
                     values='mae'
                 )
                 sns.heatmap(heatmap_data_mae, annot=True, fmt='.2f', cmap='magma_r', # Lower MAE is better
                              linewidths=.5, linecolor='black') 
                 plt.title("Cross-Evaluation: MAE (Model vs. Data Type)")
                 plt.ylabel("Model Trained On")
                 plt.xlabel("Evaluated On Data Type")
                 plt.tight_layout()
                 plot_file_mae = os.path.join(self.results_dir, 'cross_eval_mae_heatmap.png')
                 plt.savefig(plot_file_mae)
                 logger.info(f"Saved cross-evaluation MAE heatmap to {plot_file_mae}")
            except KeyError as e:
                  logger.error(f"Error pivoting/plotting MAE heatmap: Missing key {e}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting cross-evaluation heatmaps: {str(e)}")
            import traceback
            traceback.print_exc()
            
# Stand-alone execution for testing
if __name__ == "__main__":
    logger.info("Running Model Evaluator standalone test...")
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models (will train if necessary, then evaluate)
    comparison = evaluator.compare_models()
    if comparison is not None:
        print("\n--- Model Comparison Metrics ---")
        try:
             # Use tabulate for better formatting if available
             from tabulate import tabulate
             print(tabulate(comparison, headers='keys', tablefmt='grid', showindex=False))
        except ImportError:
             print(comparison.to_string())
        
    # Run cross-model evaluation
    cross_eval = evaluator.run_cross_model_evaluation()
    if cross_eval is not None:
        print("\n--- Cross-Model Evaluation Metrics (R2 Score) ---")
        try:
             pivot_r2 = cross_eval.pivot(index='model_trained_on', columns='evaluated_on_data', values='r2_score')
             from tabulate import tabulate
             print(tabulate(pivot_r2, headers='keys', tablefmt='grid'))
        except ImportError:
             print(pivot_r2.to_string())
        except KeyError:
             print("Could not pivot R2 score data for display.")
             print(cross_eval)
             
    logger.info("Model Evaluator standalone test finished.") 