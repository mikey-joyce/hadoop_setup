from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col

def main():
    spark = SparkSession.builder.appName("Tweet Hashtag & URL Extraction").master("local[*]").getOrCreate()

    df = spark.read.json("/phase1/tweets.json")
    df.createOrReplaceTempView("tweets")
    hashtags = spark.sql("select explode(entities.hashtags.text) as hashtag from tweets")
    urls = spark.sql("select explode(entities.urls.expanded_url) as url from tweets")
    result = hashtags.union(urls).select(col("hashtag").alias("extracted_item"))
    result.rdd.map(lambda row: ','.join([str(field) for field in row])).saveAsTextFile("/phase1/hashtags_urls")
    spark.stop()

if __name__ == "__main__":
    main()