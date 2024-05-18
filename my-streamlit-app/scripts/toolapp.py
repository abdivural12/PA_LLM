import streamlit as st
import pandas as pd
from tools_and_agent import agent_executor
import json

st.title("Analyse de la Température et Clustering")

# Multiple file uploader
uploaded_files = st.file_uploader("Choisissez des fichiers CSV", type="csv", accept_multiple_files=True)

if uploaded_files:
    # Concaténer les fichiers téléchargés
    data_frames = []
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file)
        data_frames.append(data)
    
    # Combiner tous les morceaux en un seul DataFrame
    combined_data = pd.concat(data_frames)
    
    st.write("Données chargées avec succès. Nombre de lignes chargées :", len(combined_data))

    n_clusters = st.slider("Nombre de clusters pour KMeans", min_value=2, max_value=10, value=4)

    # Calculate monthly average temperature
    if st.button("Calculer la température moyenne mensuelle"):
        try:
            file_path = "temp_combined_data.csv"
            combined_data.to_csv(file_path, index=False)
            result = agent_executor.invoke({
                "input": f"calcule temperature moyenne de la ville Sion pour l'année 2020, file_path {file_path}"
            })
            st.write("Température moyenne mensuelle :")
            st.dataframe(result["output"])
        except Exception as e:
            st.error(f"Erreur : {e}")

    # KMeans clustering
    if st.button("Clusteriser les données avec KMeans"):
        try:
            file_path = "temp_combined_data.csv"
            combined_data.to_csv(file_path, index=False)
            result = agent_executor.invoke({
                "input": f"clusteriser les données, file_path {file_path}, n_clusters {n_clusters}"
            })
            st.write("Résultats du clustering :")
            st.dataframe(result["output"])
        except Exception as e:
            st.error(f"Erreur : {e}")

    # Time series clustering with tslearn
    if st.button("Clusteriser les températures avec tslearn"):
        try:
            file_path = "temp_combined_data.csv"
            combined_data.to_csv(file_path, index=False)
            result = agent_executor.invoke({
                "input": f"clusteriser les temperatures avec tslearn, file_path {file_path}, n_clusters {n_clusters}"
            })
            st.write("Résultats du clustering des températures :")
            st.dataframe(result["output"])
        except Exception as e:
            st.error(f"Erreur : {e}")

# Adding a section for user questions
st.header("Posez vos questions")

user_question = st.text_input("Entrez votre question ici :")

if st.button("Poser la question"):
    if user_question and uploaded_files:
        try:
            # Sample the data to reduce size for the context
            sample_size = min(1000, len(combined_data))
            data_sample = combined_data.sample(sample_size)

            # Convert the sampled data to JSON
            data_string = data_sample.to_json(orient="split")
            context = f"The following is a sample of the data from the uploaded CSV files in JSON format:\n\n{data_string}"

            # Structure the messages for the LLM
            messages = [
                {"role": "system", "content": "You are a data analysis assistant. You can analyze and answer questions about the data provided in JSON format."},
                {"role": "user", "content": context},
                {"role": "user", "content": user_question}
            ]

            # Make the API call to the agent LLM
            result = agent_executor.invoke({"input": json.dumps(messages)})
            st.write("Réponse :")
            st.write(result["output"])
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.error("Veuillez entrer une question et charger un fichier.")
