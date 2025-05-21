import logging
import os

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from setup_vector_store import get_vactor_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def setup_rag_chain(vector_store: FAISS) -> RetrievalQA:
    logger.info("Setting up RAG-chain")
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY does not exist")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=SecretStr(GEMINI_API_KEY),
            temperature=1.0,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )
        logger.info("RAG-chain setup complete")
        return qa_chain
    except Exception as e:
        logger.error(f"Failed RAG-chain setting up: {e}")
        raise


def get_qa_chain():
    try:
        vector_store = get_vactor_store()
        qa_chain = setup_rag_chain(vector_store)
        return qa_chain
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    qa_chain = get_qa_chain()
