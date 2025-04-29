# io.py
from pyspark.sql import SparkSession
from typing import Optional, Dict, List, Any
import os

def clean_empty_parquet_files(
    hdfs_path: str, 
    spark: Optional[SparkSession] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Removes empty Parquet files from a HDFS directory that can cause 
    PyArrow/Ray Data reading errors.
    
    Args:
        hdfs_path: Path to HDFS directory containing Parquet files
        spark: Optional existing SparkSession (will create one if not provided)
        verbose: Whether to print information about removed files
    
    Returns:
        Dict with statistics about cleaned files
    """
    # Create SparkSession if not provided
    close_spark = False
    if spark is None:
        spark = SparkSession.builder.appName("CleanEmptyParquetFiles").getOrCreate()
        close_spark = True
    
    results = {
        "total_files_checked": 0,
        "empty_files_removed": 0,
        "error_files_removed": 0,
        "removed_files": []
    }
    
    try:
        # Get HDFS filesystem
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jsc.hadoopConfiguration()
        )
        
        # List files in directory
        files = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(hdfs_path))
        
        # Check each file
        for file_status in files:
            path = str(file_status.getPath())
            results["total_files_checked"] += 1
            
            # Skip _SUCCESS and non-parquet files
            if not path.endswith(".parquet") or os.path.basename(path) == "_SUCCESS":
                if verbose:
                    print(f"Skipping non-target file: {path}")
                continue
                
            # Check if file is empty
            try:
                df = spark.read.parquet(path)
                count = df.count()
                
                if count == 0:
                    if verbose:
                        print(f"Removing empty parquet file: {path}")
                    fs.delete(spark._jvm.org.apache.hadoop.fs.Path(path), False)
                    results["empty_files_removed"] += 1
                    results["removed_files"].append(path)
            except Exception as e:
                if verbose:
                    print(f"Error with file {path}, probably corrupted. Removing: {e}")
                fs.delete(spark._jvm.org.apache.hadoop.fs.Path(path), False)
                results["error_files_removed"] += 1
                results["removed_files"].append(path)
        
        if verbose:
            print(f"Clean-up complete! Removed {len(results['removed_files'])} problematic files.")
            
    finally:
        # Close SparkSession if we created it
        if close_spark:
            spark.stop()
            
    return results

