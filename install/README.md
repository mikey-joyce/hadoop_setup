# Big Data Class Software Installation Automation

## Clone Repository
1. `git clone https://github.com/mikey-joyce/hadoop_setup.git`

Note: stay in the parent directory of hadoop_setup during the install process!

i.e. If the structure is like: `/home/ubuntu/hadoop_setup`, when running the following commands run it from `/home/ubuntu/`
 
## Hadoop Install Instructions
1. `. hadoop_setup/install/hadoop1.sh`
2. `. hadoop_setup/install/hadoop2.sh`

Note: if hdfs does not work after running the second command, re-run it

## Spark Install Instructions
1. `. hadoop_setup/install/spark.sh`

To start spark: `$SPARK_HOME/bin/spark-shell`

## PySpark and Ray Install Instructions
1. `. hadoop_setup/install/pyspark_ray.sh`
