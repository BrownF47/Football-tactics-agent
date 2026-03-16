This project scrapes real football match event data, frames match situations as a Markov Decision Process (MDP), and trains a Reinforcement Learning agent to learn optimal in-game tactical decisions. The final deliverable is an interactive Streamlit dashboard visualising the agent's learned tactics on a pitch map.


Scraper:

Tried to just use beautiful soup but the data was not in the html. Selenium used to actually open browser and scrape. Turns out you can no longer scape Understat, trying the api.


Plotting functions: 

A couple of functions to plot full and half pitches.



Environment:

reset() → agent starts in random zone
    ↓
step(action) → agent moves or shoots
    ↓
    ├── if action != shoot → reward = 0, continue
    └── if action == shoot → reward = real xG from that zone, episode ends