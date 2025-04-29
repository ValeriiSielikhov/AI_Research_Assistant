import logging

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pyspark.sql import SparkSession

from generate_embedding import EMBEDDINGS_MODEL, FAISS_INDEX_PATH, MAPPING_PATH
from utils import initialize_spark

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_faiss_index(index_path: str) -> faiss.Index:
    logger.info(f"Loading FAISS-index from {index_path}")
    try:
        index = faiss.read_index(index_path)
        logger.info(f"FAISS-index loaded, count of vectors: {index.ntotal}")
        return index
    except Exception as e:
        logger.error(f"Failed loading FAISS-index: {e}")
        raise


def load_documents_and_mapping(
    spark: SparkSession, mapping_path: str, data_path: str
) -> dict:
    logger.info(f"Loading mapping from {mapping_path}")
    try:
        df_mapping = spark.read.parquet(mapping_path)
        mapping = {row["faiss_index"]: row["entry_id"] for row in df_mapping.collect()}

        df = spark.read.parquet(data_path)
        rows = df.collect()
        documents = {
            row["entry_id"]: Document(
                page_content=row["abstract_cleaned"],
                metadata={
                    "entry_id": row["entry_id"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "published": row["published"],
                },
            )
            for row in rows
        }
        logger.info(f"Loaded mapping for {len(mapping)} docs")
        return mapping, documents
    except Exception as e:
        logger.error(f"Failed loading map: {e}")
        raise


def setup_vector_store(index: faiss.Index, mapping: dict, documents: dict) -> FAISS:
    logger.info("Setting up vector store")
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        docstore = InMemoryDocstore(documents)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=mapping,
        )
        logger.info("Vector store setup complete")
        return vector_store
    except Exception as e:
        logger.error(f"Failed setting up vector store: {e}")
        raise


def main():
    try:
        spark = initialize_spark(logger, "Vector Store Setup")
        index = load_faiss_index(FAISS_INDEX_PATH)
        mapping, documents = load_documents_and_mapping(
            spark, MAPPING_PATH, "data/arxiv_data"
        )
        vector_store = setup_vector_store(index, mapping, documents)
        return vector_store
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    vector_store = main()
