import requests
import understatapi
import json
import pandas as pd
import time


client = understatapi.UnderstatClient()

# Step 1: Get all match IDs for the season

league_data = client.league(league='EPL').get_match_data(season='2024')
match_ids = [match['id'] for match in league_data]
print(f"Found {len(match_ids)} matches")

# Step 2: Loop over all matches and collect shot data

all_shots = []

for i, match_id in enumerate(match_ids):
    try:
        shot_data = client.match(match=match_id).get_shot_data()
        
        # Combine home and away shots
        shots = shot_data['h'] + shot_data['a']
        all_shots.extend(shots)
        
        print(f"[{i+1}/{len(match_ids)}] Match {match_id}: {len(shots)} shots collected")
        
        time.sleep(1)  # Be polite — don't hammer the server
        
    except Exception as e:
        print(f"Failed on match {match_id}: {e}")
        continue

# Step 3: Convert to DataFrame and save
shots_df = pd.DataFrame(all_shots)

# Convert numeric columns from strings
shots_df['X'] = shots_df['X'].astype(float)
shots_df['Y'] = shots_df['Y'].astype(float)
shots_df['xG'] = shots_df['xG'].astype(float)
shots_df['minute'] = shots_df['minute'].astype(int)

print(f"\nTotal shots collected: {len(shots_df)}")
print(shots_df.head())
print(shots_df.columns.tolist())

# Save to parquet for fast reloading
shots_df.to_parquet('data/shots.parquet', index=False)
print("Saved to data/shots.parquet")