import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plotting_functions.plot_pitch import plot_half_pitch
from plotting_functions.plot_pitch import plot_pitch

# Load data
shots_df = pd.read_parquet('data/shots.parquet')

# Basic stats
print("Total shots:", len(shots_df))
print("\nShots by result:")
print(shots_df['result'].value_counts())
print("\nShots by situation:")
print(shots_df['situation'].value_counts())
print("\nAverage xG by situation:")
print(shots_df.groupby('situation')['xG'].mean().sort_values(ascending=False))
print("\nAverage xG by zone (X coordinate buckets):")
shots_df['x_zone'] = pd.cut(shots_df['X'], bins=3, labels=['Deep', 'Mid', 'Close'])
print(shots_df.groupby('x_zone')['xG'].mean())
print(shots_df.head())

# Plot shot locations on a pitch

#Colour by result
colors = {
    'Goal': 'green',
    'SavedShot': 'blue', 
    'MissedShots': 'red',
    'BlockedShot': 'orange'
}

for result, group in shots_df.groupby('result'):
    plt.scatter(
        group['X'] * 100 +0.5 ,  # Understat uses 0-1, mplsoccer expects 0-100
        group['Y'] * 70,
        color=colors.get(str(result), 'gray'),
        s=1,
        alpha=0.5,
        label=result
    )

plt.legend(loc='lower left')
plt.title('EPL 2024/25 Shot Map')
plot_pitch()

plt.savefig('data/shot_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("Shot map saved!")

