import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.cluster import KMeans
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize the ChatGPT model
llm = ChatOpenAI(model="gpt-4-0125-preview")

# Get the prompt and create an agent executor
prompt = hub.pull("hwchase17/openai-tools-agent")

# Load data
global_data = None

# Streamlit application
st.title("Analyse de la Température et Clustering")

# File uploader
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file:
    file_path = uploaded_file.name  # Streamlit handle the file name automatically
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load data into a global variable
    global_data = pd.read_csv(file_path)
    st.write("Données chargées avec succès.")

    n_clusters = st.slider("Nombre de clusters pour KMeans", min_value=2, max_value=10, value=3)

    # Calculate monthly average temperature
    if st.button("Calculer la température moyenne mensuelle"):
        try:
            # Calculation logic
            global_data['date'] = pd.to_datetime(global_data['year']*1000 + global_data['day_of_year'], format='%Y%j') + pd.to_timedelta(global_data['minute_of_day'], unit='T')
            global_data.set_index('date', inplace=True)
            monthly_avg = global_data.resample('M').agg({'tre200s0': 'mean'})
            st.write("Température moyenne mensuelle :")
            st.dataframe(monthly_avg.reset_index())
        except Exception as e:
            st.error(f"Erreur : {e}")

    # KMeans clustering
    if st.button("Clusteriser les données"):
        try:
            # Clustering logic
            kmeans = KMeans(n_clusters=n_clusters)
            global_data['cluster'] = kmeans.fit_predict(global_data[['tre200s0']].values.reshape(-1, 1))
            st.write("Résultats du clustering :")
            st.dataframe(global_data.reset_index())
        except Exception as e:
            st.error(f"Erreur : {e}")

    # Moving average
    window_size = st.slider("Window size for moving average", min_value=1, max_value=30, value=5)

    if st.button("Calculate Moving Average"):
        try:
            global_data['moving_average'] = global_data['tre200s0'].rolling(window=window_size).mean()
            st.write("Moving Average Results:")
            st.dataframe(global_data.reset_index())
        except Exception as e:
            st.error(f"Error: {e}")

# Adding a section for user questions
st.header("Posez vos questions")

user_question = st.text_input("Entrez votre question ici :")

if st.button("Poser la question"):
    if user_question and global_data is not None:
        try:
            # Convert the DataFrame to a string for context
            data_string = global_data.to_csv(index=False)
            context = f"The following is the data from the uploaded CSV file:\n\n{data_string}"

            # Structure the messages for the LLM
            messages = [
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": context},
                {"role": "user", "content": user_question}
            ]

            # Make the API call to the agent LLM
            response = llm(messages=messages)
            
            # Ensure the response structure is correct
            if 'choices' in response and len(response['choices']) > 0:
                st.write("Réponse :")
                st.write(response['choices'][0]['message']['content'])
            else:
                st.error("Aucune réponse valide reçue du modèle.")
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.error("Veuillez entrer une question et charger un fichier.")
