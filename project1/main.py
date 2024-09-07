# command to execute:- streamlit run main.py
import os
import streamlit as st
import pickle
import time
import langchain

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

st.title("Gen AI first project")
st.sidebar.title("URL")

loader_text = st.empty()
urls = []
data = None
docs = None

for i in range(3):
    url = st.sidebar.text_input(f"Url {i+1}")
    urls.append(url)

submit = st.sidebar.button("Process URL")

if submit:
    loader_text.text("URL Loading...")
    data = UnstructuredURLLoader(
        urls=urls
    ).load()
    # st.write(len(data))

if data:
    loader_text.text("Text splitter is running...")
    docs = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', " "],
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(data)
    # st.write(len(docs))

if docs:
    loader_text.text("Embedding is going on...")
    embeddings = OpenAIEmbeddings()
    vector_index_openai = FAISS.from_documents(docs, embeddings)
    time.sleep(2)

    vector_index_openai.save_local("./pkl/faiss_store")

query = loader_text.text_input("Your Question-")

if query:
    vectorIndex = FAISS.load_local(
        "./pkl/faiss_store",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0.9, max_tokens=500),
        retriever=vectorIndex.as_retriever())
    # st.write(chain)

    # query = "what is price of google pixel 9 in rupees?"

    # langchain.debug=True
    result = chain({'question': query}, return_only_outputs=True)
    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources", "")
    if sources:
        st.header("Source")
        list_sources = sources.split('\n')
        for source in list_sources:
            st.write(source)

