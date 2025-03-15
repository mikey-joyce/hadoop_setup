from pyspark import SparkContext, SparkConf
import pandas as pd
from io import StringIO, BytesIO

def main():
    conf = SparkConf().setAppName("ParseData").set("spark.driver.memory", "6g").set("spark.executor.memory", "6g")
    sc = SparkContext(conf=conf)

    sc.setLogLevel("OFF")

    data_path = 'hdfs://localhost:9000/phase2/data/'
    files = sc.wholeTextFiles(data_path)

    print()
    print("FILES: ", files)
    print()

    inp = []
    for filename, content in files.take(11):
        print(f"Detected filename: {filename}")
        print("Content: ", content)

    parsed = files.mapValues(lambda file: (file[0], detect_and_parse(file[1], file[0])))

    print()
    print("PARSED: ", parsed)
    print()

    dfs = parsed.collectAsMap()

    print(len(dfs))


def detect_and_parse(content, filename):
    filename = filename.lower().strip()
    print(f"Processing {filename}...")

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(StringIO(content), on_bad_lines="skip")
        elif filename.endswith(".json"):
            return pd.read_json(StringIO(content))
        elif filename.endswith(".xls") or filename.endswith(".xlsx"):
            return pd.read_excel(BytesIO(content.encode("utf-8")))
        else:
            print(f"Unsupported file format: {filename}")
            return None
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


if __name__ == "__main__":
    main()
