import streamlit as st

def main():
    st.set_page_config(page_title="Mzuni AI Assistant", page_icon=":books:")
    st.header("Mzuni AI Assistant :books:")
    st.text_input("Ask me what you want to know about Mzuzu University")
    
    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Please upload your documents here")
        st.button("Process")
        
        
    # with st.sidebar:
    #     st.sidebar.title("About")
    #     st.sidebar.markdown("This is an AI assistant for Mzuzu University")

if __name__ == "__main__":
    main()