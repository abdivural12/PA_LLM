import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
import logging
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%Y-%m-%d %H:%M:%S - %(levelname)s - %(message)s')

# Load environment variables (if any other than the CSV file path)
load_dotenv()

# Define a helper function to parse dates
def parse_dates(df):
    try:
        df['date'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='m')
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
def cluster_temperatures_tslearn(file_path: str, n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster the temperatures using time series clustering with tslearn.
    
    Args:
        file_path (str): Path to the CSV file containing temperature data.
        n_clusters (int): Number of clusters for the clustering algorithm.
        
    Returns:
        pd.DataFrame: DataFrame containing the cluster labels for each time series.
    """
    try:
        # Load the data
        data = pd.read_csv(file_path)
        
        # Ensure the 'date' column is created correctly
        parse_dates(data)
        
        # Filter for the year 2020
        data_2020 = data[data.index.year == 2020]
        
        # Calculate the average temperature per month
        data_2020['month'] = data_2020.index.month
        monthly_avg_temp_2020 = data_2020.groupby(['name', 'month'])['tre200s0'].mean().reset_index()
        
        # Pivot the DataFrame
        pivot_monthly_avg_temp = monthly_avg_temp_2020.pivot(index='name', columns='month', values='tre200s0')
        
        # Fill NaN values
        pivot_monthly_avg_temp_filled = pivot_monthly_avg_temp.fillna(pivot_monthly_avg_temp.mean())
        
        # Convert the pivoted DataFrame to a time series dataset
        formatted_dataset = to_time_series_dataset(pivot_monthly_avg_temp_filled.to_numpy())
        
        # Define and fit the clustering model
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=33)
        labels = model.fit_predict(formatted_dataset)
        
        # Create a DataFrame with the results
        result_df = pd.DataFrame({
            'name': pivot_monthly_avg_temp_filled.index,
            'cluster': labels
        })
        
        # Plot the cluster centers
        plt.figure(figsize=(10, 6))
        for i, center in enumerate(model.cluster_centers_):
            plt.plot(center.ravel(), label=f'Cluster {i}')
        plt.title('Centres des Clusters de Température Moyenne Mensuelle par Station (2020)')
        plt.xlabel('Mois')
        plt.ylabel('Température Moyenne (°C)')
        plt.xticks(ticks=range(12), labels=range(1, 13))  # De 1 (Janvier) à 12 (Décembre)
        plt.legend()
        plt.show()
        
        return result_df
    
    except Exception as e:
        logging.error("Error clustering temperatures with tslearn: %s", e)
        raise

# Initialize the ChatGPT model
llm = ChatOpenAI(model="gpt-4-0125-preview")
tools = [calculate_monthly_average_temperature, kmeans_cluster_time_series, cluster_temperatures_tslearn]  # Including the new tool

# Get the prompt and create an agent executor
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.pretty_print()

# Create an agent executor by passing in the agent and tools
# Construct the tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
