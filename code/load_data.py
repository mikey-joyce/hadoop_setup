import time
from hdfs_utils import clean_empty_parquet_files
import pyarrow as pa
import ray.data as rd

def load_and_prepare_dataset(spark, hdfs_path):
    """
    Reads training data from the given HDFS path using Spark,
    cleans up empty or error parquet files, loads the dataset into Ray,
    verifies the dataset counts, and returns the Ray dataset.
    """
    # Read with Spark
    train_spark_df = spark.read.parquet(hdfs_path)
    train_spark_df.show(5)

    time.sleep(10)
    
    # Clean up Parquet files
    cleanup_results = clean_empty_parquet_files(hdfs_path, spark=spark, verbose=True)
    print(f"Cleaned {cleanup_results['empty_files_removed']} empty files and {cleanup_results['error_files_removed']} error files")

    # Load Ray Dataset using a Hadoop FileSystem instance for pyarrow
    hdfs = pa.fs.HadoopFileSystem("localhost", 9000)

    # Ensure trailing slash for proper reading by Ray
    # path = hdfs_path if hdfs_path.endswith("/") else hdfs_path + "/"
    train_dataset = rd.read_parquet(hdfs_path, filesystem=hdfs)
    
    print("Dataset type:")
    print(type(train_dataset))
    
    print("Print sample data from Ray Dataset:")
    print(train_dataset.take(2))
    
    # Check if spark dataset to ray dataset worked
    spark_count = train_spark_df.count()
    ray_count = train_dataset.count()
    print(f"Spark DataFrame count: {spark_count}")
    print(f"Ray Dataset count: {ray_count}")

    # Optionally, you could add integrity checks here

    return train_dataset
