import streamlit as st
import pandas as pd
import torch
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import time
import logging

logging.basicConfig(level=logging.INFO)

def load_vector_db():
# Load the pdf files from the path
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    return vector_store

# Create LLM
llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    device="cuda" if torch.cuda.is_available() else "cpu",
    config={'max_new_tokens': 128, 'temperature': 0.01}
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vector_store = load_vector_db()

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! You can ask me anything about mental health. 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Conversation function
def conversation_chat(query):
    try:
        start_time = time.time()
        result = chain({"question": query, "chat_history": st.session_state['history']})
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if result.get("answer") is None:
            logging.warning("No valid answer found from the model.")
            return "Sorry, I couldn't understand your question.", elapsed_time
        
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"], elapsed_time
    
    except Exception as e:
        logging.error(f"An error occurred during conversation: {str(e)}")
        return "Sorry, there was an error in processing your request.", 0.0

# Display chat history
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Send')

            if submit_button and not user_input:
                st.warning('Please enter a question.')

        if submit_button and user_input:
            output, elapsed_time = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['elapsed_time'] = elapsed_time  # Store elapsed time in session state

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

                if 'elapsed_time' in st.session_state:
                    st.write(f"Time taken for this response: {st.session_state['elapsed_time']:.2f} seconds")

# Initialize session state
initialize_session_state()

# Load the glossary dataset
def glossary_dataset():
    glossary_data = "C:\\Users\\User\\python project\\MHchat\\Book1.csv"
    df = pd.read_csv(glossary_data)
    return df

# Create sidebar for navigation
st.sidebar.title("Navigation")
navigation = st.sidebar.radio("Go to", ["Chatbot", "Glossaries"])

if navigation == "Chatbot":
    display_chat_history()
elif navigation == "Glossaries":
    st.sidebar.title("Glossaries")
    st.sidebar.write("Here you can find information about various mental health terms and their definitions.")
    df = glossary_dataset()
    st.write("Glossary Dataset:")
    st.dataframe(df)

st.sidebar.link_button("Provide Us Feedback","https://forms.gle/wQh237zXgUonGtnU7")
