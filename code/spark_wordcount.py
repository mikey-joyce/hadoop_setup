from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName("WordCount").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    input_files = "/phase1/hashtags_urls/*.txt"
    text_files = sc.textFile(input_files)
    text_files = text_files.filter(lambda x: "_SUCCESS.txt" not in x)

    words = text_files.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1))
    counts = words.reduceByKey(lambda x, y: x + y)

    results = counts.collect()
    for word, count in results:
        print(f"{word}: {count}")

    results.saveAsTextFile("/phase1/results/spark_wordcount")

    sc.stop()

if __name__ == "__main__":
    main()