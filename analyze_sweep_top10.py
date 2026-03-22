import wandb
import pandas as pd

# Initialize wandb API
api = wandb.Api()

# Get the sweep
sweep = api.sweep("fatty_data/CF-DQN-cleanrl/r6qlovfv")

print("Fetching sweep runs...")
results = []

for run in sweep.runs:
    # Get the metric value
    q_values_mean = run.summary.get('losses/q_values_all_mean')
    
    if q_values_mean is not None:
        results.append({
            'run_id': run.id,
            'run_name': run.name,
            'seed': run.config.get('seed'),
            'n_frequencies': run.config.get('n_frequencies'),
            'freq_max': run.config.get('freq_max'),
            'penalty_weight': run.config.get('penalty_weight'),
            'q_values_mean': q_values_mean,
            'state': run.state
        })

# Create DataFrame
df = pd.DataFrame(results)

# Sort by q_values_mean (higher is better for Q-values typically)
df_sorted = df.sort_values('q_values_mean', ascending=False)

print("\n" + "="*80)
print("TOP 10 RUNS BY Q-VALUES MEAN")
print("="*80)
print(df_sorted.head(10).to_string(index=False))

print("\n" + "="*80)
print("HYPERPARAMETER TRENDS IN TOP 10")
print("="*80)

top10 = df_sorted.head(10)

print("\nFrequency of hyperparameters in top 10:")
print(f"\nn_frequencies distribution:")
print(top10['n_frequencies'].value_counts().sort_index())
print(f"\nfreq_max distribution:")
print(top10['freq_max'].value_counts().sort_index())
print(f"\npenalty_weight distribution:")
print(top10['penalty_weight'].value_counts().sort_index())
print(f"\nseed distribution:")
print(top10['seed'].value_counts().sort_index())

print("\n" + "="*80)
print("AVERAGE VALUES IN TOP 10 vs ALL RUNS")
print("="*80)

for col in ['n_frequencies', 'freq_max', 'penalty_weight']:
    top10_mean = top10[col].mean()
    all_mean = df[col].mean()
    print(f"\n{col}:")
    print(f"  Top 10 avg: {top10_mean:.2f}")
    print(f"  All runs avg: {all_mean:.2f}")
    print(f"  Difference: {top10_mean - all_mean:+.2f}")

# Group by hyperparameters and show aggregated results
print("\n" + "="*80)
print("BEST HYPERPARAMETER COMBINATIONS (averaged over seeds)")
print("="*80)

grouped = df.groupby(['n_frequencies', 'freq_max', 'penalty_weight']).agg({
    'q_values_mean': ['mean', 'std', 'count']
}).round(3)

grouped.columns = ['mean_q_values', 'std_q_values', 'n_runs']
grouped_sorted = grouped.sort_values('mean_q_values', ascending=False)

print(grouped_sorted.head(10).to_string())

print("\n" + "="*80)
print(f"Total runs analyzed: {len(df)}")
print(f"Runs with q_values_mean: {len(df)}")
print("="*80)