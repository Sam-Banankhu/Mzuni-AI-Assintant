import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
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
            text += page.extract_text()
    return text


def main():
    load_dotenv()
    st.set_page_config(page_title="Mzuni AI Assistant", page_icon=":books:")
    st.header("Mzuni AI Assistant :books:")
    st.text_input("Ask me what you want to know about Mzuzu University")
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Please upload your documents here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf files
                raw_text = get_pdf_text(pdf_docs)
                print(raw_text)
                # get chunks of pdf files
                
                # create vectors store
            
        
        
    # with st.sidebar:
    #     st.sidebar.title("About")
    #     st.sidebar.markdown("This is an AI assistant for Mzuzu University")

if __name__ == "__main__":
    main()