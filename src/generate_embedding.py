import logging
from typing import List, Tuple

import faiss
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sentence_transformers import SentenceTransformer

from utils import initialize_spark

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


INPUT_PATH = "data/arxiv_data"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/faiss_index"
MAPPING_PATH = "data/embeddings_mapping"


def load_data(spark: SparkSession, input_path: str) -> List[Tuple[str, str]]:
    logger.info(f"Loading data from {input_path}")
    try:
        df = spark.read.parquet(input_path)
        data = (
            df.select("entry_id", "abstract_cleaned")
            .filter(col("abstract_cleaned").isNotNull())
            .collect()
        )
        result = [(row["entry_id"], row["abstract_cleaned"]) for row in data]
        logger.info(f"Loaded {len(result)} abstracts")
        return result
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    logger.info(f"Embeding generation for {len(texts)} docs")
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        logger.info(f"Embeding size: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed embeding generation: {e}")
        raise


def save_to_faiss(
    embeddings: np.ndarray,
    entry_ids: List[str],
    index_path: str,
    mapping_path: str,
    spark: SparkSession,
) -> None:
    logger.info(f"Saving FAISS-indexs to {index_path}")
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # number of the docs smaller than 1000
        index.add(embeddings)
        faiss.write_index(index, index_path)
        logger.info("FAISS-indexs saved successfully")

        mapping_data = [
            {"entry_id": entry_id, "faiss_index": i}
            for i, entry_id in enumerate(entry_ids)
        ]
        df_mapping = spark.createDataFrame(mapping_data)
        df_mapping.write.parquet(mapping_path, mode="overwrite")
        logger.info(f"Mapping saved to {mapping_path}")
    except Exception as e:
        logger.error(f"Failed FAISS or mapping saving : {e}")
        raise


def main():
    try:
        spark = initialize_spark(logger, "Embedding Generator")
        data = load_data(spark, INPUT_PATH)
        if not data:
            logger.warning("Empty data loaded, skipping embedding generation")
            return

        entry_ids, texts = zip(*data)
        entry_ids = list(entry_ids)
        texts = list(texts)

        embeddings = generate_embeddings(texts, EMBEDDINGS_MODEL)
        save_to_faiss(embeddings, entry_ids, FAISS_INDEX_PATH, MAPPING_PATH, spark)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
