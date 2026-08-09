from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

import config

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the
context provided below, which was retrieved from the user's own documents.

Rules:
- If the answer is not contained in the context, say "I couldn't find that in the documents"
  instead of guessing.
- Be concise and direct.
- When useful, refer to specific facts from the context rather than paraphrasing loosely.

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vectordb = Chroma(
        persist_directory=config.VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name="documind_collection",
    )
    return vectordb


def retrieve_context(vectordb, question, k=config.TOP_K):
    results = vectordb.similarity_search_with_score(question, k=k)
    return results


def format_context(results):
    context_parts = []
    sources = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(doc.page_content)
        sources.append(f"{source} (page {page})")
    return "\n\n---\n\n".join(context_parts), sources


def get_answer(question, vectordb, llm=None):
    if llm is None:
        llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
        )

    results = retrieve_context(vectordb, question)
    context, sources = format_context(results)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    messages = prompt.format_messages(context=context, question=question)

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sorted(set(sources)),
    }


if __name__ == "__main__":
    vectordb = load_vectorstore()
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        result = get_answer(q, vectordb)
        print("\nAnswer:", result["answer"])
        print("Sources:", result["sources"])