import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.cluster import KMeans
from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Define a helper function to parse dates
def parse_dates(df):
    try:
        df['date'] = pd.to_datetime(df['year']*1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='T')
        df.set_index('date', inplace=True)
    except Exception as e:
        logging.error("Failed to parse dates: %s", e)
        raise

@tool
def load_time_series(file_path: str) -> pd.DataFrame:
    """Load time series data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        return df
    except Exception as e:
        logging.error("Error loading time series data from file %s: %s", file_path, e)
        raise

@tool
def calculate_monthly_average_temperature(file_path: str) -> pd.DataFrame:
    """Calculate monthly average temperature from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        monthly_avg = df.resample('M').agg({'tre200s0': 'mean'})
        return monthly_avg.reset_index()
    except Exception as e:
        logging.error("Error processing file %s: %s", file_path, e)
        raise

@tool
def kmeans_cluster_time_series(file_path: str, n_clusters: int) -> pd.DataFrame:
    """Apply KMeans clustering to time series data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df[['tre200s0']].values.reshape(-1, 1))
        df['cluster'] = clusters
        return df.reset_index()
    except Exception as e:
        logging.error("Error during KMeans clustering: %s", e)
        raise

@tool
def moving_average(file_path: str, window_size: int) -> pd.DataFrame:
    """Calculate the moving average of a time series from a CSV file."""
    try:
        df = pd.read_csv(file_path, parse_dates=True, index_col='date')
        df['moving_average'] = df['tre200s0'].rolling(window=window_size).mean()
        return df.reset_index()
    except Exception as e:
        logging.error("Error processing file %s: %s", file_path, e)
        raise

# Initialize the ChatGPT model
llm = ChatOpenAI(model="gpt-4-0125-preview")
tools = [calculate_monthly_average_temperature, kmeans_cluster_time_series, moving_average]

# Get the prompt and create an agent executor
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit application
st.title("Analyse de la Température et Clustering")

# File uploader
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

# User inputs
if uploaded_file:
    file_path = uploaded_file.name  # Streamlit handle the file name automatically
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    n_clusters = st.slider("Nombre de clusters pour KMeans", min_value=2, max_value=10, value=3)
    
    # Calculate monthly average temperature
    if st.button("Calculer la température moyenne mensuelle"):
        try:
            result = agent_executor.invoke({
                "input": f"calcule temperature moyenne de la ville Sion pour l'année 2020, file_path {file_path}"
            })
            st.write("Température moyenne mensuelle :")
            st.dataframe(result["output"])
        except Exception as e:
            st.error(f"Erreur : {e}")

    # KMeans clustering
    if st.button("Clusteriser les données"):
        try:
            result = agent_executor.invoke({
                "input": f"clusteriser les données, file_path {file_path}, n_clusters {n_clusters}"
            })
            st.write("Résultats du clustering :")
            st.dataframe(result["output"])
        except Exception as e:
            st.error(f"Erreur : {e}")

    # Moving average
    window_size = st.slider("Window size for moving average", min_value=1, max_value=30, value=5)

    if st.button("Calculate Moving Average"):
        try:
            result = agent_executor.invoke({
                "input": f"Calculate moving average with window size {window_size}, file_path {file_path}"
            })
            st.write("Moving Average Results:")
            st.dataframe(result["output"])
        except Exception as e:
            st.error(f"Error: {e}")

# Adding a section for user questions
st.header("Posez vos questions")

user_question = st.text_input("Entrez votre question ici :")

if st.button("Poser la question"):
    if user_question:
        try:
            response = agent_executor.invoke({"input": user_question})
            st.write("Réponse :")
            st.write(response["output"])
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.error("Veuillez entrer une question.")
