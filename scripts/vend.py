import streamlit as st
import pandas as pd
import logging
import matplotlib.pyplot as plt
from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from sklearn.cluster import KMeans
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure logging configuration is correctly applied
def configure_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

configure_logging()

# Define helper functions
def parse_dates(df):
    """
    Parse date components into a datetime object and set as index.
    Args:
        df (pd.DataFrame): DataFrame containing date components.

    Raises:
        Exception: If date parsing fails.
    """
    try:
        df['date'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='m')
        df.set_index('date', inplace=True)
    except Exception as e:
        logging.error("Failed to parse dates: %s", e)
        raise

@tool
def load_time_series(file_path: str) -> pd.DataFrame:
    """
    Load time series data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame with time series data.
    Raises:
        Exception: If loading the data fails.
    """
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        return df
    except Exception as e:
        logging.error("Error loading time series data from file %s: %s", file_path, e)
        raise

@tool
def calculate_monthly_average_temperature(file_path: str) -> pd.DataFrame:
    """
    Calculate monthly average temperature from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame with monthly average temperatures.
    Raises:
        Exception: If processing the data fails.
    """
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
    """
    Apply KMeans clustering to time series data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
        n_clusters (int): Number of clusters for KMeans.
    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    Raises:
        Exception: If clustering fails.
    """
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
def cluster_temperatures_tslearn(file_path: str, n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster the temperatures using time series clustering with tslearn.
    Args:
        file_path (str): Path to the CSV file containing temperature data.
        n_clusters (int): Number of clusters for the clustering algorithm.
    Returns:
        pd.DataFrame: DataFrame containing the cluster labels for each time series.
    Raises:
        Exception: If clustering fails.
    """
    try:
        data = pd.read_csv(file_path)
        parse_dates(data)
        data['month'] = data.index.month
        monthly_avg_temp = data.groupby(['name', 'month'])['tre200s0'].mean().reset_index()
        pivot_monthly_avg_temp = monthly_avg_temp.pivot(index='name', columns='month', values='tre200s0')
        pivot_monthly_avg_temp_filled = pivot_monthly_avg_temp.fillna(pivot_monthly_avg_temp.mean())
        formatted_dataset = to_time_series_dataset(pivot_monthly_avg_temp_filled.to_numpy())
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=33)
        labels = model.fit_predict(formatted_dataset)
        result_df = pd.DataFrame({'name': pivot_monthly_avg_temp_filled.index, 'cluster': labels})
        plt.figure(figsize=(10, 6))
        for i, center in enumerate(model.cluster_centers_):
            plt.plot(center.ravel(), label=f'Cluster {i}')
        plt.title('Centres des Clusters de Température Moyenne Mensuelle par Station')
        plt.xlabel('Mois')
        plt.ylabel('Température Moyenne (°C)')
        plt.xticks(ticks=range(12), labels=range(1, 13))
        plt.legend()
        plt.show()
        return result_df
    except Exception as e:
        logging.error("Error clustering temperatures with tslearn: %s", e)
        raise

# Initialize the ChatGPT model
llm = ChatOpenAI(model="gpt-4-0125-preview")
tools = [calculate_monthly_average_temperature, kmeans_cluster_time_series, cluster_temperatures_tslearn]
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit app layout
st.title("Meteorology Data Analysis and Clustering")
st.write("This app allows you to load meteorological data, perform analyses, and get insights using an AI agent.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Input fields for agent
question = st.text_area("Ask a question about the data:")

# Button to process the question
if st.button("Submit"):
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = "temp.csv"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the CSV file
        df = pd.read_csv(temp_file_path)
        parse_dates(df)
        
        # Display the dataframe
        st.write("Uploaded Data")
        st.write(df.head())
        
        # Convert dataframe to HTML to provide more structured data to the agent
        html_table = df.head().to_html()
        
        # Prepare context for the LLM
        context = f"The CSV file contains the following columns: {', '.join(df.columns)}. Here are the first few rows of the data:\n{html_table}\n"
        
        # Provide file data as a string for complex questions
        file_data = df.to_csv(index=False)
        
        # Define the input for the agent
        input_data = {
            "input": f"{context}\n{question}",
            "file_data": file_data
        }
        
        # Invoke the agent for the question
        try:
            response = agent_executor.invoke(input_data)
            st.write(response)
        except Exception as e:
            st.error(f"Error invoking the agent: {e}")
    else:
        st.error("Please upload a CSV file first.")
