import streamlit as st
import langchain_community
import langchain_openai
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings

###
# This is the main file for the chatbot.
# It allows the user to upload a text file and ask questions about the data in the file.
# Authors: @alex_seitz_dev
# NOTE: To run this file, you must have the following installed:
# streamlit, langchain, langchain_openai, langchain_community, langchain_community.vectorstores
# You can install these packages using pip: 
# pip install streamlit langchain langchain_openai langchain_community langchain_community.vectorstores
# You also need to have an OpenAI API key. You can get one at https://platform.openai.com/api-keys
# IMPORTANT:When running the file, you will need to run it through the terminal using streamlit 
# the run the command: 
# python -m streamlit run chat.py
###

st.title("Personalized LLM-App")

# User data uploader in the sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your text file", type="txt")
    openai_api_key = "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"

# Add near the top of your file
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

def process_text_file(uploaded_file):
    """ The process_text_file function is 
    used to process the uploaded text file 
    and create a vector database for the 
    chatbot. This allows the chatbot to be
    able to understand the user's data.
    
    Args:
        uploaded_file (file): The uploaded text file.
        
    Returns:
        vectorstore (FAISS): The vector database for the chatbot.
    """
    # Read the content of the uploaded file
    text = uploaded_file.read().decode("utf-8")
    
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and store in vector database
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore

def generate_response(vectorstore, query):
    """ The generate_response function is 
    used to generate a response to the user's query.
    
    Args:
        vectorstore (FAISS): The vector database for the chatbot.
        query (str): The user's query.
        
    Returns:
        response (str): The response to the user's query.
    """
    if st.session_state.conversation_chain is None:
        # Create memory and conversation chain only once
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0.7, openai_api_key=openai_api_key),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    
    response = st.session_state.conversation_chain({"question": query})
    return response['answer']

# Optional: Add a way to display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main interface
if uploaded_file is not None:
    vectorstore = process_text_file(uploaded_file)
    st.success("File processed successfully!")
    
    with st.form("my_form"):
        text = st.text_area('Enter your question:', '')
        submitted = st.form_submit_button("Submit")
        
        if not openai_api_key.startswith('sk-'):
            st.warning("Please enter your OpenAI API key!")
            
        if submitted:
            response = generate_response(vectorstore, text)
            # Store messages
            st.session_state.messages.append({"role": "user", "content": text})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.write("You:", message["content"])
                else:
                    st.write("Assistant:", message["content"])
else:
    st.info("Please upload a text file to begin.")

if st.button("Clear Chat"):
    st.session_state.conversation_chain = None
    st.session_state.messages = []


