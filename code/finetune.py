import ray
import ray.data as rd
from pyspark.sql import SparkSession
import time

def main():
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()
    rdd = spark.sparkContext.textFile("hdfs:///phase2/data/train")
    train = rdd.collect()
    train = rd.from_items(train)
    train.show(5)
    time.sleep(60)



if __name__ == "__main__":
    main()
