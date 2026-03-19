import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from stable_baselines3 import PPO
import joblib
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.gym_environment import FootballEnv

# --- Page Config ---
st.set_page_config(
    page_title="TacticsRL",
    page_icon="⚽",
    layout="wide"
)

# --- Load Data and Models ---
@st.cache_resource
def load_resources():
    shots_df = pd.read_parquet('data/shots.parquet')
    model = PPO.load('agent/ppo_football')
    xg_model = joblib.load('find_xG/xg_model.pkl')
    env = FootballEnv(shots_df)
    return shots_df, model, xg_model, env

shots_df, model, xg_model, env = load_resources()

# --- Title ---
st.title("🏟️ TacticsRL")
st.markdown("*Learning football tactics from real EPL shot data using Reinforcement Learning*")
st.divider()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🤖 Agent Tactics", "📈 Results"])

# ============================================================
# TAB 1 — Data Explorer
# ============================================================
with tab1:
    st.subheader("EPL Shot Data Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shots", f"{len(shots_df):,}")
    col2.metric("Total Goals", f"{(shots_df['result'] == 'Goal').sum():,}")
    col3.metric("Avg xG per Shot", f"{shots_df['xG'].mean():.3f}")

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        situation_filter = st.multiselect(
            "Filter by situation",
            options=shots_df['situation'].unique().tolist(),
            default=shots_df['situation'].unique().tolist()
        )
    with col2:
        result_filter = st.multiselect(
            "Filter by result",
            options=shots_df['result'].unique().tolist(),
            default=shots_df['result'].unique().tolist()
        )

    filtered_df = shots_df[
        (shots_df['situation'].isin(situation_filter)) &
        (shots_df['result'].isin(result_filter))
    ]

    # Shot map
    fig, ax = plt.subplots(figsize=(5, 7))

    colors = {'Goal': 'green', 'SavedShot': 'blue',
              'MissedShots': 'red', 'BlockedShot': 'orange'}

    for result, group in filtered_df.groupby('result'):
        ax.scatter(
            group['X'] * 50,
            group['Y'] * 70,
            color=colors.get(result, 'grey'),
            s=5, alpha=0.5, label=result
        )

    ax.legend(loc='lower left', facecolor='white')
    ax.set_title(f'Shot Map ({len(filtered_df):,} shots)')
    st.pyplot(fig)
    plt.close()

    # xG by situation table
    st.subheader("Average xG by Situation")
    xg_table = shots_df.groupby('situation')['xG'].agg(['mean', 'count'])
    xg_table.columns = ['Avg xG', 'Shot Count']
    xg_table = xg_table.sort_values('Avg xG', ascending=False)
    xg_table['Avg xG'] = xg_table['Avg xG'].round(4)
    st.dataframe(xg_table, use_container_width=True)

# ============================================================
# TAB 2 — Agent Tactics
# ============================================================
with tab2:
    st.subheader("Learned Agent Tactics")
    st.markdown("What action does the agent choose at every position on the pitch?")

    action_labels = {
        0: "Move Wide",
        1: "Move Central",
        2: "Move Closer",
        3: "Hold",
        4: "Shoot"
    }
    action_colors = {
        0: "#3498db",   # Blue
        1: "#9b59b6",   # Purple
        2: "#e67e22",   # Orange
        3: "#95a5a6",   # Grey
        4: "#2ecc71"    # Green
    }

    # Generate grid of positions
    x_vals = np.linspace(0.5, 1.0, 25)  # Attacking half only
    y_vals = np.linspace(0.0, 1.0, 25)

    positions = []
    for x in x_vals:
        for y in y_vals:
            obs = np.array([x, y, 0.5, 0.5], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            positions.append({'x': x, 'y': y, 'action': int(action)})

    positions_df = pd.DataFrame(positions)

    # Plot tactics heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    pitch = VerticalPitch(pitch_type='statsbomb', half=True,
                          pitch_color='#1a1a2e', line_color='white')
    pitch.draw(ax=ax)

    for _, row in positions_df.iterrows():
        color = action_colors[row['action']]
        ax.scatter(
            row['y'] * 100,
            row['x'] * 100,
            c=color, s=60, alpha=0.7, zorder=3
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=action_colors[a], label=action_labels[a])
                       for a in action_labels]
    ax.legend(handles=legend_elements, loc='lower left', facecolor='white')
    ax.set_title("Agent's Preferred Action by Pitch Position")
    st.pyplot(fig)
    plt.close()

    # Show action distribution
    st.subheader("Overall Action Distribution")
    action_counts = positions_df['action'].value_counts().sort_index()
    action_counts.index = [action_labels[i] for i in action_counts.index]
    st.bar_chart(action_counts)

# ============================================================
# TAB 3 — Results
# ============================================================
with tab3:
    st.subheader("Agent Performance vs Baselines")

    col1, col2, col3 = st.columns(3)
    col1.metric("PPO Agent", "0.6416 xG", "+55.1% vs Random")
    col2.metric("Always Shoot", "0.4455 xG", baseline=None)
    col3.metric("Random Agent", "0.4137 xG", baseline=None)

    st.markdown("---")
    st.subheader("What the Agent Learned")
    st.markdown("""
    - The agent learned that **repositioning before shooting** leads to significantly higher xG
    - It preferentially moves to **central positions** close to goal before shooting
    - A **55% improvement** over random action selection demonstrates genuine tactical learning
    - Results are driven by a **logistic regression xG model** fitted on 10,000 real EPL shots
    """)

    st.subheader("Model Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**xG Model**")
        st.markdown("- Logistic Regression\n- Features: distance, angle\n- Trained on 10,000 EPL shots")
    with col2:
        st.markdown("**RL Agent**")
        st.markdown("- Algorithm: PPO\n- Training steps: 200,000\n- Repositioning cost: 0.01")