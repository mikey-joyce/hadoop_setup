from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName("WordCount").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    sc.setLogLevel("INFO")

    log_file = "hdfs://localhost:9000/phase1/log4j.properties"
    sc.addFile(log_file)

    input_files = "/phase1/hashtags_urls/*"
    text_files = sc.textFile(input_files)
    text_files = text_files.filter(lambda x: "_SUCCESS" not in x)

    words = text_files.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1))
    counts = words.reduceByKey(lambda x, y: x + y)

    counts.saveAsTextFile("/phase1/results/spark_wordcount")

    results = counts.collect()
    for word, count in results:
        print(f"{word}: {count}")

    sc.stop()

if __name__ == "__main__":
    main()