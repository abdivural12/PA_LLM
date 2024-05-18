from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.llms import OpenAI as LangChainOpenAI
from dotenv import load_dotenv
import os
import streamlit as st

def main():
    # Load environment variables
    load_dotenv()

    # Retrieve the OpenAI API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')

    # Set up the Streamlit page
    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    # Define the path of the CSV file
    file_path = "C:/Users/Abdi/Desktop/data/raw/meteo_idaweb.csv"

    # Check if the CSV file exists
    if os.path.exists(file_path):
        # Create a CSV agent using LangChain and OpenAI
        csv_agent = create_csv_agent(
            llm=LangChainOpenAI(api_key=api_key),  # Use the correct parameter name 'llm'
            path=file_path,  # Use the correct parameter name 'path'
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        # User input to ask a question
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question:
            with st.spinner("In progress..."):
                # Run the agent to get the response
                response = csv_agent.run(user_question)
                st.write(response)
    else:
        st.error("The specified CSV file does not exist. Please check the file path.")

if __name__ == "__main__":
    main()
