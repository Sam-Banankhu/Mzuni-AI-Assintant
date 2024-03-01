import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    
    for pdf in pdf_docs:
        pdf_reader =  PdfReader(pdf)
        for page in pdf_reader.pages:
            text += remove_special_characters(page.extract_text())
            # text += page.extract_text()
    return text

def remove_special_characters(text):
    # Define regex pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'  

    clean_text = re.sub(pattern, '', text)
    # Replace consecutive white spaces with a single space
    # clean_text = re.sub(r'\s+', ' ', clean_text)
    # Remove leading and trailing white spaces
    clean_text = clean_text.strip()
    
    return clean_text

def get_text_chunks(text):
    textsplitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = textsplitter.split_text(text)
    
    return chunks

def create_vectors_store(text_chunks):
    embeddings = OpenAIEmbeddings()  # Assuming OpenAIEmbeddings is a class or function that generates embeddings
    # index = FAISS()  # Assuming FAISS is a class or function that creates a FAISS index
    
    vector_store = FAISS.from_texts(text_chunks, embeddings)
        
    return vector_store

def get_conversation_chains(vectors_db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
     # Creating the retriever configuration dictionary
    retriever_config = {
        "name": "FAISS",
        "vector_store": vectors_db
    }
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, memory=memory, retriever= vectors_db.as_retriever()
    )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="Mzuni AI Assistant", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = None
        # pass
        
    st.header("Mzuni AI Assistant :books:")
    user_question = st.text_input("Ask me what you want to know about Mzuzu University")
    if user_question:
        handle_userInput(user_question)
    
    st.write(user_template.replace("{{MSG}}", "Hello Chat:"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello User:"), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Please upload your documents here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    # get pdf files
                    raw_text = get_pdf_text(pdf_docs)
                    # st.write(raw_text)
                    # get chunks of pdf files
                    text_chunks = get_text_chunks(raw_text)
                    # st.write(text_chunks)text_chunks, vectorstext_chunks, vectorsxt_chunks, vectors
                    # create vectors store
                    vector_db = create_vectors_store(text_chunks=text_chunks)
                    
                    # query = "What are the gaps in Malawi National Sanitation Policy 2008?"
                    # docs = vector_db.similarity_search(query)
                    # st.write(docs[0].page_content)
                    st.session_state.conversation = get_conversation_chains(vector_db)
    
                except Exception as e:
                    st.error(e)
                    
    # st.session_state.conversation
        
    # with st.sidebar:
    #     st.sidebar.title("About")
    #     st.sidebar.markdown("This is an AI assistant for Mzuzu University")

if __name__ == "__main__":
    main()