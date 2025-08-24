
import os
from dotenv import find_dotenv, load_dotenv
import openai
from openai import RateLimitError
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message # pip install streamlit_chat

st.sidebar.title('Langchain Retrieval')
# Load env variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

#==== Streamlit front-end ====
st.title("Docs QA Bot")
st.header("Upload your docs and ask questions...")


persist_dir = './data'

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None


# File uploader
uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=False)

if uploaded_files and 'vector_db' not in st.session_state:
    documents = []
    for file in uploaded_files:
        file_path = os.path.join("./temp", uploaded_files.name)
        os.makedirs("./temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_files.read())


        # Choose loader based on extension
        if uploaded_files.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_files.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)


        documents.extend(loader.load())


        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)

        # Rebuild collection
        vectordb = Chroma.from_documents(
                                documents=docs,
                                embedding=embedding,
                                persist_directory=persist_dir
                    )
        vectordb.persist()
        st.session_state['vector_db'] = vectordb

        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    vectordb.as_retriever(search_kwargs={'k': 6}),
                    return_source_documents=True,
                    verbose=False
                    )
        st.session_state['qa_chain'] = qa_chain
        st.success("Documents uploaded and processed!")

# Get query from user
user_input = st.chat_input("Ask a question about your uploaded documents...")


if user_input and st.session_state['qa_chain']:
    st.session_state.past.append(user_input)
    try:
        result = st.session_state['qa_chain']({'question': user_input, 'chat_history': []})
        st.session_state.generated.append(result['answer'])
    except RateLimitError:
        st.session_state.generated.append('Request too large for backend')


# Display chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i)+ '_user')
        message(st.session_state['generated'][i], key=str(i))
