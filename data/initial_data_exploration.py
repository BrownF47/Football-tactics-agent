import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

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

# Plot shot locations on a pitch
pitch = VerticalPitch(pitch_type='statsbomb', half=True)
fig, ax= pitch.draw(figsize=(8, 6))


# Colour by result
colors = {
    'Goal': 'green',
    'SavedShot': 'blue', 
    'MissedShots': 'red',
    'BlockedShot': 'orange'
}

for result, group in shots_df.groupby('result'):
    pitch.scatter(
        group['X'] * 100,  # Understat uses 0-1, mplsoccer expects 0-100
        group['Y'] * 100,
        ax=ax,
        color=colors.get(result, 'grey'),
        s=20,
        alpha=0.5,
        label=result
    )

ax.legend(loc='lower left')
ax.set_title('EPL 2024/25 Shot Map')
plt.savefig('data/shot_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("Shot map saved!")