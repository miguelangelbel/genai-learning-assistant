import os
from dotenv import load_dotenv
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


LLM_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


texts = [
    "Amazon S3 provides object storage through buckets.",
    "Amazon EC2 provides scalable virtual computing instances.",
    "AWS Lambda runs code without managing servers.",
    "Amazon RDS provides managed relational databases.",
    "Amazon CloudFront is a content delivery network."
]

documents = [
    Document(page_content=text)
    for text in texts
]


contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Given the conversation history and the latest user question,
        rewrite the question so that it can be understood independently.

        Do not answer the question.
        Only return the rewritten standalone question.
        """
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        "{question}"
    )
])


answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the user's question using only the provided context.

        If the answer cannot be found in the context, say:
        "The information is not available in the provided context."

        Context:
        {context}
        """
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        "{question}"
    )
])


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


@lru_cache(maxsize=1)
def get_rag_components():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = InMemoryVectorStore(embeddings)
    _ = vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=api_key
    )

    contextualize_chain = (
        contextualize_prompt
        | llm
        | StrOutputParser()
    )

    answer_chain = (
        answer_prompt
        | llm
        | StrOutputParser()
    )

    return retriever, contextualize_chain, answer_chain


def conversational_rag(question, history):
    retriever, contextualize_chain, answer_chain = (
        get_rag_components()
    )

    standalone_question = contextualize_chain.invoke({
        "history": history,
        "question": question
    })

    retrieved_docs = retriever.invoke(
        standalone_question
    )

    context = format_docs(retrieved_docs)

    answer = answer_chain.invoke({
        "context": context,
        "history": history,
        "question": question
    })

    history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])

    return answer