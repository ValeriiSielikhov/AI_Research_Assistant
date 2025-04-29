from pyspark.sql import SparkSession


def initialize_spark(logger, sessian_name: str = "Spark session") -> SparkSession:
    try:
        spark = SparkSession.builder.appName(sessian_name).getOrCreate()
        logger.info("Spark session initialized successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialized Spark session: {e}")
        raise
