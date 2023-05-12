import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os
import requests

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
openai_api_key = os.environ.get("OPENAI_API_KEY")

# UI
st.title("PDF Search with GPT-4")
st.subheader("Search within PDF documents using GPT-4")
pdf_file_option = st.radio("Select PDF source", ("Enter a URL","Upload a file", "Enter raw text"))

pdf_url = None
pdf_file = None
raw_text = None

if pdf_file_option == "Enter a URL":
    pdf_url = st.text_input("Enter PDF URL")
elif pdf_file_option == "Upload a file":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
elif pdf_file_option == "Enter raw text":
    raw_text = st.text_area("Enter raw text")

if pdf_url or pdf_file or raw_text:
    st.write("Processing PDF...")

    if pdf_file:
        pdf_file_path = pdf_file.name
        with open(pdf_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
    elif pdf_url:
        pdf_file_path = pdf_url
        response = requests.get(pdf_file_path)
        with open("temp_pdf.pdf", "wb") as f:
            f.write(response.content)
        pdf_file_path = "temp_pdf.pdf"
    elif raw_text:
        pdf_file_path = None

    if pdf_file_path:
        # Process PDF
        reader = PdfReader(pdf_file_path)
        raw_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    # Load QA chain
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(ChatOpenAI(), chain_type="stuff")

    # Initialize session state and chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    # User input
    query = st.text_input("Enter your query")

    if st.button("Search"):
        if query:
            st.session_state.chat_history.insert(0, ("result", ""))
            st.session_state.chat_history.insert(0, ("query", query))
            docs = docsearch.similarity_search(query)
            result = chain.run(input_documents=docs, question=query)
            st.session_state.chat_history[1] = ("result", result)

            # Clear input box by resetting the state
            st.experimental_rerun()

    # Display chat history
    st.markdown("---")
    st.header("Chat History")
    for entry_type, entry_content in st.session_state.chat_history:
        if entry_type == "query":
           
            st.markdown(
                f'<div style="background-color: #444654; padding: 10px; color:#ffffff; border-radius: 5px; margin-bottom: 5px;"><strong>You:</strong> {entry_content}</div>',
                unsafe_allow_html=True,
            )
        elif entry_type == "result" and entry_content:
            st.markdown(
                f'<div style="background-color: #333541; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 5px;"><strong>GPT-4:</strong> {entry_content}</div>',
                unsafe_allow_html=True,
            )

    if st.button("Clear"):
        st.session_state.chat_history = []
else:
    st.write("Upload a PDF, enter a URL, or input raw text to begin.")
