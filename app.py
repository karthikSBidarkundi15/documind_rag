import os
import streamlit as st

import config
from ingest import load_documents, split_documents, build_vectorstore
from rag_chain import load_vectorstore, get_answer
from langchain_groq import ChatGroq

st.set_page_config(page_title="DocuMind — Chat with your PDFs", page_icon="📄", layout="centered")

st.title("📄 DocuMind")
st.caption("Upload documents, then ask questions grounded in their actual content.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

with st.sidebar:
    st.header("1. Build your knowledge base")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with st.spinner("Reading, chunking, and embedding your documents..."):
            for file in uploaded_files:
                path = os.path.join(config.DATA_DIR, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

            docs = load_documents()
            chunks = split_documents(docs)
            st.session_state.vectordb = build_vectorstore(chunks)

        st.success(f"Processed {len(uploaded_files)} file(s) into {len(chunks)} chunks.")

    st.divider()
    st.header("2. Or load an existing knowledge base")
    if st.button("Load previously processed documents"):
        st.session_state.vectordb = load_vectorstore()
        st.success("Loaded existing vector store.")

if st.session_state.vectordb is None:
    st.info("Upload and process documents in the sidebar to get started.")
else:
    llm = ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_answer(question, st.session_state.vectordb, llm=llm)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.write(f"- {s}")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})