import ray
import ray.data as rd
from pyspark.sql import SparkSession
import time

def main():
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()
    ray.init()

    print("Session started...")
    time.sleep(5)

    rdd = spark.sparkContext.textFile("hdfs:///phase2/data/train")
    print(":)")
    time.sleep(5)
    train = rd.from_items(list(rdd.toLocalIterator()))
    train.show(5)
    time.sleep(60)



if __name__ == "__main__":
    main()
