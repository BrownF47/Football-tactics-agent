import requests
import understatapi


client = understatapi.UnderstatClient()

league_data = client.league(league='EPL').get_match_data(season='2024')
    
print(league_data[0])

shot_data = client.match(match='26602').get_shot_data()

print(shot_data)