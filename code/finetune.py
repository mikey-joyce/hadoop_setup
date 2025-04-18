import ray
import ray.data as rd
from pyspark.sql import SparkSession
import time

def main():
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()
    rdd = spark.sparkContext.textFile("hdfs:///phase2/data/train")
    print("HELLO!")
    time.sleep(10)
    train = rd.from_items(rdd.toLocalIterator())
    train.show(5)
    time.sleep(60)



if __name__ == "__main__":
    main()
