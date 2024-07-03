import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key and Cohere API key
groq_api_key = os.environ['GROQ_API_KEY']
cohere_api_key = os.environ["COHERE_API_KEY"]

# Dictionary of available LLM models
llm_models = {
    "Mixtral": "mixtral-8x7b-32768",
    "Gemma": "gemma-7b-it",
    "Llama3 (8B)": "llama3-8b-8192",
    "Llama3 (70B)": "llama3-70b-8192",
}

# Initialize session state if not already done
if "vector" not in st.session_state:
    st.session_state.embeddings = CohereEmbeddings()

# Page config
st.set_page_config(
    page_title="WebMeAnything",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #333333; /* Dark grey background */
            color: #ffffff; /* White text */
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
        }
        .stApp h1 {
            color: #4B8BBE; /* Blue header */
            text-align: center;
            margin-bottom: 30px;
        }
        .stApp input[type="text"], .stApp select {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 12px 16px; /* Increased padding for input fields */
            margin-bottom: 20px;
            width: 100%; /* Full width for input fields */
            max-width: 800px; /* Maximum width for input fields */
            box-sizing: border-box;
            color: #ffffff; /* White text for input */
        }
        .stApp .btn-primary {
            background-color: #4B8BBE;
            color: #ffffff;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .stApp .btn-primary:hover {
            background-color: #3579A8;
        }
        .stApp .expander {
            background-color: #444444; /* Dark grey expander background */
            border: 1px solid #666666;
            border-radius: 5px;
            margin-top: 20px;
            padding: 15px;
        }
        .stApp .expander .expanderHeader {
            color: #4B8BBE; /* Blue header for expander */
            font-weight: bold;
            cursor: pointer;
            font-size: 18px;
        }
        .stApp .expander .expanderContent {
            margin-top: 10px;
        }
        .powered-by {
            position: fixed;
            bottom: 10px;
            right: 10px;
            color: #aaaaaa;
            font-size: 12px;
        }
        .stApp .sidebar .sidebar-content {
            background-color: #2f2f2f; /* Darker sidebar background */
            color: #ffffff; /* White text */
        }
        .stApp .sidebar .sidebar-content .stMulti {
            margin-top: 20px;
        }
        .stApp .sidebar .sidebar-content .stSelectbox {
            color: #ffffff; /* White text */
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("💡 WebMeAnything")
st.markdown("### Your Intelligent Web-Page Document Analysis Tool")

# Input for URL
url = st.text_input("Enter the website URL to analyze", help="Provide the URL of the website you want to analyze", key="url_input", type="default")

# Select model
llm_model_choice = st.selectbox("Choose Large Language Model", list(llm_models.keys()))

if url and llm_model_choice:
    with st.spinner("Loading and processing the website content..."):
        # Load and process the website content dynamically
        st.session_state.loader = WebBaseLoader(url)
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        # Initialize the language model
        llm_model_name = llm_models[llm_model_choice]
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model_name)
        
        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}
        """
        )
        
        # Create document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Input prompt
        prompt = st.text_input("Input your prompt here", help="Type your question and press Enter")

        # Display response with response time
        if prompt:
            with st.spinner("Generating response..."):
                start_time = time.time()
                response = retrieval_chain.invoke({"input": prompt})
                response_time = time.time() - start_time
                st.success(f"Response time: {response_time:.2f} seconds")
                st.write(response['answer'])

                # With a Streamlit expander
                with st.expander("Document Similarity Search"):
                    # Find the relevant chunks
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")

# Powered by attribution
st.markdown('---')
st.markdown('Powered by <a href="https://www.streamlit.io/" target="_blank">Streamlit</a>, ChatGroq, Cohere, and FAISS', unsafe_allow_html=True)
