from pyspark import SparkContext, SparkConf
import pandas as pd
from io import StringIO

def main():
    conf = SparkConf().setAppName("ParseData").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    sc.setLogLevel("INFO")

    data_path = '/phase2/data/*'

    files = sc.wholeTextFiles(data_path)
    parsed = files.mapValues(process_with_pandas)
    dfs = parsed.collectAsMap()

    print(len(dfs))


def process_with_pandas(file_content):
    """Convert CSV content into a Pandas DataFrame."""
    return pd.read_csv(StringIO(file_content))

if __name__ == "__main__":
    main()