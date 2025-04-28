import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import ArxivLoader
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


QUERY = "large language model"
MAX_DOCS = 2
OUTPUT_PATH = "data/arxiv_data"


def initialize_spark() -> SparkSession:
    try:
        spark = SparkSession.builder.appName("ArxivDataLoader").getOrCreate()
        logger.info("Spark session initialized successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialized Spark session: {e}")
        raise


def save_to_csv(
    spark: SparkSession, data: List[Dict[str, str]], output_path: str
) -> None:
    logger.info(f"Saving data to {output_path}")
    try:
        df = spark.createDataFrame(data)
        df_cleaned = df.withColumn(
            "abstract_cleaned", regexp_replace(col("abstract"), "[^a-zA-Z0-9\s]", "")
        )
        df_cleaned.persist()
        df_cleaned.write.parquet(output_path, mode="overwrite")
        count = df_cleaned.count()
        logger.info(f"Saved {count} articles to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise
    finally:
        df_cleaned.unpersist()


def preprosess_documents(docs) -> list:
    logger.info("Preprocessing documents")
    data = []
    for doc in docs:
        metadata = doc.metadata
        abstract = (
            doc.page_content
            if not metadata.get("Abstract")
            else metadata.get("Abstract", "")
        )
        data.append(
            {
                "title": metadata.get("Title", ""),
                "abstract": abstract,
                "authors": metadata.get("Authors", ""),
                "summary": metadata.get("Summary", ""),
                "published": metadata.get("Published", ""),
                "entry_id": metadata.get("entry_id", ""),
            }
        )
    logger.info(f"Preprocessed {len(data)} documents")
    return data


def load_arxiv_documents(query: str, max_docs: int) -> list[Any]:
    logger.info(f"Loading up to {max_docs} documents from Arxiv with query: {query}")
    try:
        loader = ArxivLoader(
            query=query, load_max_docs=max_docs, top_k_results=max_docs
        )
        docs = loader.load()
        logger.info(f"Loadede {len(docs)} documents")
        return docs
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise


def main():
    try:
        spark = initialize_spark()
        docs = load_arxiv_documents(query=QUERY, max_docs=MAX_DOCS)
        data = preprosess_documents(docs)
        save_to_csv(spark, data, OUTPUT_PATH)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        spark.stop()
        logger.info("spark session stopped")


if __name__ == "__main__":
    main()
