# Dream11 Fantasy Cricket Prediction System

## Overview
The Dream11 Fantasy Cricket Prediction System is an advanced analytics tool designed to optimize team selection for fantasy cricket competitions. The system leverages machine learning models, statistical analysis, and optimization techniques to recommend the most competitive team composition based on player data and match conditions.

## Features
- **Hybrid Model Ensemble**: Combines multiple prediction models including time-series analysis for improved accuracy
- **Monte Carlo Simulation**: Provides robust team recommendations by simulating various outcome scenarios
- **Anti-Fragile Strategy**: Ensures team composition can perform well across different match conditions
- **Pitch Type Optimization**: Adjusts team composition based on pitch characteristics
- **Captain/Vice-Captain Selection**: Intelligently selects optimal captain and vice-captain based on performance metrics
- **Partnership Analysis**: Identifies strong batting and bowling partnerships to optimize team composition

## Quick Start

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dream11-prediction-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place squad data in the `dataset` directory
   - Ensure CSV files include player names, teams, roles, and credits

### Running the System
1. Run the Day 2 demo to see the enhanced features in action:
   ```bash
   python day2_demo.py
   ```

2. For specific components:
   ```bash
   # Run hybrid model ensemble
   python hybrid_model_ensemble.py

   # Run Monte Carlo simulation for robust team selection
   python monte_carlo_simulation.py

   # Run anti-fragile strategy analysis
   python anti_fragile_strategy.py
   
   # Run partnership analyzer example
   python examples/partnership_analyzer_example.py
   ```

3. For the main application:
   ```bash
   python app.py
   ```

## System Components

### Data Processing
- **Feature Engineering**: Transforms raw player data into meaningful features
- **Data Standardization**: Ensures consistent data formats across different sources

### Prediction Models
- **Linear Regression**: Baseline performance prediction
- **Gradient Boosting**: Advanced performance prediction with feature importance
- **XGBoost**: Optimized gradient boosting implementation
- **Time Series Analysis**: Captures player form and trend data

### Team Optimization
- **Linear Programming**: Ensures team composition meets all constraints
- **Monte Carlo Simulation**: Handles uncertainty in player performance
- **Anti-Fragile Strategy**: Balances team composition for different match scenarios

### Enhanced Features (Day 2 Implementation)
- **Hybrid Model Ensemble**: Combines multiple models for more accurate predictions
- **Anti-Fragile Team Selection**: Creates teams that perform well across various scenarios
- **Monte Carlo Simulation**: Tests thousands of possible outcomes to find optimal teams
- **Advanced Captain Selection**: Uses sophisticated metrics to select the best captain and vice-captain

### Enhanced Features (Day 3 Implementation)
- **Partnership Analyzer**: Analyzes batting and bowling partnerships to optimize team selection
- **Form Analyzer**: Analyzes player form and consistency to improve predictions
- **Matchup Analyzer**: Analyzes player performance against specific opponents
- **Stadium Strategies**: Optimizes team selection based on venue characteristics
- **Pitch Type Optimizer**: Adjusts player values based on pitch conditions

## Enhanced Features Modules
The system includes several specialized modules for advanced team optimization:

### Partnership Analyzer
The `PartnershipAnalyzer` identifies strong batting and bowling partnerships to optimize team selection:
- Analyzes historical partnership data between players
- Calculates partnership strength based on statistical metrics
- Adjusts player predictions based on established partnerships
- Recommends strong partnerships to include in team selection

### Form Analyzer
The `FormAnalyzer` analyzes player form and consistency:
- Tracks player performance over time
- Calculates consistency scores and identifies form trends
- Adjusts predictions based on recent form
- Identifies in-form players for optimal team selection

### Matchup Analyzer
The `MatchupAnalyzer` analyzes player performance against specific opponents:
- Identifies favorable player vs. team matchups
- Analyzes batsman vs. bowler historical performance
- Adjusts player predictions based on opposition

### Stadium Strategies
The `StadiumStrategies` optimizes team selection based on venue characteristics:
- Maintains profiles for different stadiums
- Adjusts role importance based on venue conditions
- Recommends team composition based on stadium type

### Pitch Type Optimizer
The `PitchTypeOptimizer` adjusts player values based on pitch conditions:
- Identifies pitch type (batting/bowling/balanced/spin/pace-friendly)
- Adjusts player values based on their skills and the pitch type
- Optimizes team composition for specific pitch conditions

## Model Enhancement Roadmap

The following is a 5-day implementation plan to enhance model accuracy (from approximately 33% R² to 65-70%) and robustness (from 80% to 95%) using advanced but practical techniques:

### Current Model Analysis
- **Accuracy**: R² score ~0.33 (33%)
- **Model architecture**: Multiple ML models with ensemble approach
- **Robustness**: ~80% with Monte Carlo simulation

### Day 1: Temporal Fusion Transformer Implementation

Implementing advanced time-series modeling with TFT to better capture player form:

```python
# Installation: pip install pytorch-forecasting pytorch-lightning

# Create player time-series dataset
def create_time_series_dataset(player_data):
    player_data['match_date'] = pd.to_datetime(player_data['match_date'])
    player_data = player_data.sort_values(['player_id', 'match_date'])
    
    # Create time index within each player series
    player_data['time_idx'] = player_data.groupby('player_id').cumcount()
    
    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        player_data,
        time_idx="time_idx",
        target="fantasy_points",
        group_ids=["player_id"],
        max_encoder_length=5,  # Last 5 matches
        max_prediction_length=1,  # Next match
        time_varying_known_reals=["venue_code", "opposition_strength", "is_home"],
        time_varying_unknown_reals=["fantasy_points", "runs", "wickets", "strike_rate", "economy"]
    )
    return training

# Train TFT model
def train_tft_model(training_data):
    trainer = pl.Trainer(max_epochs=30, gpus=0)
    tft = TemporalFusionTransformer.from_dataset(
        training_data,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=0.001
    )
    trainer.fit(tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    return tft
```

**Expected outcome**: R² improvement from 0.33 to 0.40-0.42 (7-9% lift)

### Day 2: Enhanced Partnership Analysis with Graph Networks

Implementing graph-based player partnership analysis:

```python
# Installation: pip install stellargraph networkx

# Create partnership graph
def create_partnership_graph(partnership_data):
    # Create graph
    G = nx.Graph()
    
    # Add players as nodes
    player_features = {}
    all_players = set()
    for _, row in partnership_data.iterrows():
        all_players.add(row['player1'])
        all_players.add(row['player2'])
    
    for player in all_players:
        # Node attributes from player stats
        G.add_node(player)
    
    # Add partnerships as edges with weights
    for _, row in partnership_data.iterrows():
        player1, player2 = row['player1'], row['player2']
        strength = row.get('strength', 'neutral')
        
        # Convert strength to numeric value
        strength_value = {
            'strong': 1.0,
            'good': 0.75,
            'neutral': 0.5,
            'poor': 0.25
        }.get(strength, 0.5)
        
        G.add_edge(player1, player2, weight=strength_value)
    
    return G

# Implement GCN model
def create_gcn_model(graph):
    # Convert to StellarGraph
    sg_graph = sg.StellarGraph.from_networkx(graph)
    
    # Create GCN model
    generator = sg.mapper.FullBatchNodeGenerator(sg_graph)
    gcn = GCN(
        layer_sizes=[32, 16], 
        activations=['relu', 'relu'],
        generator=generator,
        dropout=0.2
    )
    
    x_in, x_out = gcn.in_out_tensors()
    predictions = Dense(1)(x_out)
    
    model = Model(inputs=x_in, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='mean_squared_error',
    )
    
    return model, generator
```

**Expected outcome**: R² improvement to 0.45-0.48 (additional 5-6% lift)

### Day 3: Anti-Fragile Strategy and Robust Optimization

Enhancing team robustness with improved anti-fragile approaches:

```python
# Enhance anti-fragile strategy with confidence intervals
def enhance_anti_fragile_strategy(player_data):
    # Calculate consistency with Bayesian approach
    consistency_scores = {}
    for player in player_data['player_name'].unique():
        player_history = get_player_history(player)
        if len(player_history) >= 3:
            # Calculate mean and std
            mean = player_history['fantasy_points'].mean()
            std = player_history['fantasy_points'].std()
            
            # Calculate coefficient of variation (lower is more consistent)
            cv = std / mean if mean > 0 else float('inf')
            
            # Calculate lower bound (more conservative estimate)
            lower_bound = mean - 1.96 * std / np.sqrt(len(player_history))
            
            consistency_scores[player] = {
                'mean': mean,
                'std': std,
                'cv': cv,
                'lower_bound': max(0, lower_bound)
            }
    
    return consistency_scores

# Implement robust optimization
def quantum_inspired_optimization(problem):
    # Create QUBO model (simplified version of quantum approach)
    # This is a classical implementation inspired by quantum methods
    
    # Use simulated annealing (inspired by quantum annealing)
    from scipy.optimize import dual_annealing
    
    # Define objective function with penalties
    def objective(x):
        # Convert binary vector to player selection
        selected = [pid for i, pid in enumerate(problem['players']) if x[i] > 0.5]
        
        # Calculate team value
        value = sum(problem['players'][pid]['points'] for pid in selected)
        
        # Calculate penalties
        penalties = calculate_constraint_penalties(selected, problem)
        
        return -(value - penalties)  # Negative because we're minimizing
    
    # Run optimization
    bounds = [(0, 1) for _ in range(len(problem['players']))]
    result = dual_annealing(objective, bounds, maxiter=1000)
    
    # Convert result to player selection
    selected_players = [pid for i, pid in enumerate(problem['players']) 
                       if result.x[i] > 0.5]
    
    return selected_players
```

**Expected outcome**: R² improvement to 0.52-0.55 (additional 7% lift), robustness increase to 85-88%

### Day 4: Multi-modal Player Embeddings and Performance Optimization

Adding richer player representations using NLP techniques:

```python
# Create player embeddings using text descriptions
def create_player_embeddings(player_data):
    # Load pre-trained model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Create player descriptions
    descriptions = {}
    for _, player in player_data.iterrows():
        # Create contextual description
        desc = f"{player['player_name']} is a {player['role']} who performs well "
        
        # Add role-specific details
        if player['role'] == 'BAT':
            desc += f"with a strike rate of {player.get('strike_rate', 'unknown')}. "
        elif player['role'] == 'BOWL':
            desc += f"with an economy of {player.get('economy', 'unknown')}. "
        
        # Add pitch preferences if available
        if 'pitch_preference' in player:
            desc += f"Prefers {player['pitch_preference']} pitches. "
        
        descriptions[player['player_name']] = desc
    
    # Create embeddings
    embeddings = {}
    for player, desc in descriptions.items():
        embeddings[player] = model.encode(desc)
    
    return embeddings
```

**Expected outcome**: R² improvement to 0.58-0.62 (additional 6-7% lift)

### Day 5: Integration, Testing, and Final Optimizations

Final integration of all components:

```python
# Integrate all components
def integrate_all_components():
    # Create final pipeline that combines:
    # - TFT for time-series prediction
    # - GCN for partnership modeling
    # - Enhanced anti-fragile strategy
    # - Player embeddings
    # - Robust optimization
    pass

# Comprehensive testing
def test_against_historical_matches():
    # Test against 50+ historical matches
    pass

# Final model tuning
def final_tuning():
    # Final adjustments based on test results
    pass
```

**Expected outcome**: Final R² of 0.62-0.65 (almost double the original), robustness of 92-95%

### Expected Performance Improvement Summary

| Day | Component | R² Before | R² After | Robustness |
|-----|-----------|-----------|----------|------------|
| 1   | TFT       | 0.33      | 0.40-0.42| 80% → 82%  |
| 2   | GCN       | 0.42      | 0.48-0.50| 82% → 85%  |
| 3   | Robust Opt| 0.50      | 0.55-0.58| 85% → 88%  |
| 4   | Embeddings| 0.58      | 0.62-0.65| 88% → 92%  |
| 5   | Integration| 0.65     | 0.67-0.70| 92% → 95%  |

### Implementation Notes

1. **Hardware Requirements**:
   - Standard CPU for most operations
   - 8GB+ RAM for graph networks and TFT
   - No GPU required (but would speed up TFT training)

2. **Implementation Constraints**:
   - Keep existing pitch-specific models while adding enhanced features
   - Maintain linear regression fallbacks for robustness
   - Ensure historical player match data with timestamps is available

3. **Integration with Existing System**:
   - All new components will build on the existing architecture
   - Minimal changes to the core optimization engine
   - Focus on adding enhanced predictive capability

## File Structure
- `app.py`: Main application file
- `team_optimizer.py`: Team selection and optimization logic
- `feature_engineering.py`: Feature creation and preprocessing
- `model_integration.py`: Integration of prediction models
- `hybrid_model_ensemble.py`: Implementation of model ensemble
- `monte_carlo_simulation.py`: Monte Carlo simulation for team selection
- `anti_fragile_strategy.py`: Anti-fragile strategy implementation
- `calculate_fantasy_points.py`: Fantasy points calculation logic
- `enhanced_features/`: Directory containing enhanced feature modules
  - `__init__.py`: Package initialization file
  - `partnership_analyzer.py`: Partnership analysis implementation
  - `form_analyzer.py`: Form analysis implementation
  - `matchup_analyzer.py`: Matchup analysis implementation
  - `stadium_strategies.py`: Stadium-based strategies
  - `pitch_type_optimizer.py`: Pitch type optimization
  - `enhanced_features.py`: Integration of all enhanced features
- `examples/`: Directory containing example scripts
  - `partnership_analyzer_example.py`: Example usage of the partnership analyzer
- `requirements.txt`: Required Python packages
- `day2_demo.py`: Demonstration of enhanced features

## Logging
The system provides detailed logging at various stages:
- Data loading and validation
- Feature engineering
- Model predictions
- Team optimization
- Simulation results
- Partnership analysis

Logs can be found in the following files:
- `app.log`: Main application logs
- `day2_demo.log`: Demo run logs

## Troubleshooting
- Ensure all required data files are present in the `dataset` directory
- Check log files for detailed error information
- Verify that the `models` directory exists for saving/loading trained models

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.