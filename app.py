import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from file_processing.file_preprocessing import *


def get_pdf_text(pdf_docs):
    """_summary_

    Args:
        pdf_docs (_type_): _pdf_file_

    Returns:
        _type_: _str_
    """
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

def main():
    load_dotenv()
    st.set_page_config(page_title="Mzuni AI Assistant", page_icon=":books:")
    st.header("Mzuni AI Assistant :books:")
    st.text_input("Ask me what you want to know about Mzuzu University")
    
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
                    vectors = create_vectors_store(text_chunks=text_chunks)
                except Exception as e:
                    st.error(e)
        
        
    # with st.sidebar:
    #     st.sidebar.title("About")
    #     st.sidebar.markdown("This is an AI assistant for Mzuzu University")

if __name__ == "__main__":
    main()