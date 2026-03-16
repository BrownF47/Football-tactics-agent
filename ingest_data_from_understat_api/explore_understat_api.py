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
