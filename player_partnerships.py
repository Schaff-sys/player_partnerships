import pandas as pd 
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import networkx as nx


# Environment Setup
# -------------------
env_path = Path(__file__).resolve().parent / ".env"

print("Loading environment variables from:", env_path)
load_dotenv(dotenv_path=env_path, override=True)

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_name = os.getenv("db_name")


if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError("Variables de entorno incompletas.")

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

## Loading and adjusting data
# Read the entire table into a DataFrame
## Loading and adjusting data
# Read the entire table into a DataFrame
df = pd.read_sql_query(text("SELECT * FROM mergeddatabasesall;"), con=engine)

competitions = df.set_index("match_competitionDisplayName")["match_competitionId"].to_dict()


teams = df.set_index("team_name")["team_id"].to_dict()


st.title('Water Polo Passing Networks')

equipo = st.selectbox(
    "Selecciona un equipo:",
    list(teams.keys())
)

team = teams[equipo]

competition = st.selectbox(
    "Selecciona una competición:",
    ["All"] +list(competitions.keys())
)

if competition != "All":
    competition_id = competitions[competition]
    df = df[df['match_competitionId'] == competition_id]


df = df[df['team_id']==team]

partnerships = df[['shot_taken_by', 'shot_assist_by']]

partnerships = partnerships.replace(["", " "], np.nan)

partnerships = partnerships.dropna(subset=['shot_taken_by'])

partnerships = partnerships.dropna(subset=['shot_assist_by'])

duplicates = partnerships[partnerships.duplicated(keep=False)]

pairs = duplicates.groupby(['shot_taken_by', 'shot_assist_by']).size().reset_index(name='weight')

# Add a slider to select minimum number of passes
min_passes = st.slider(
    "Minimum number of passes to include:",
    min_value=1,
    max_value=int(pairs['weight'].max()),  # max based on your data
    value=1  # default
)

# Filter based on slider value
pairs_filtered = pairs[pairs['weight'] >= min_passes]


G = nx.DiGraph()

for idx, row in pairs_filtered.iterrows():
    G.add_edge(row['shot_assist_by'], row['shot_taken_by'], weight=row['weight'])

pos = nx.spring_layout(G, k=1)

edges = G.edges(data=True)
weights = [d['weight'] for (_, _, d) in edges]

plt.figure(figsize=(10,8))

if len(weights) == 0:
    st.write("No passing partnerships found for this selection.")
else:
    # Normalize edge weights so thickness scales better
    min_w, max_w = min(weights), max(weights)
    if min_w == max_w:  # avoid division by zero if all weights equal
        scaled_weights = [3 for _ in weights]  # constant thickness
    else:
        scaled_weights = [1 + 9 * ((w - min_w) / (max_w - min_w)) for w in weights]  # scale 1–10

    plt.figure(figsize=(10,8))
    nx.draw(
        G, pos, with_labels=True, node_size=9000, font_size=10,
        width=scaled_weights, edge_color="gray", arrowsize=50
    )
    if competition != "All":
        plt.title(f"Goal-Assister Network for {team} in {competition}")
    else:
        plt.title(f"Goal-Assister Network for {equipo} in all competitions")
    st.pyplot(plt)
