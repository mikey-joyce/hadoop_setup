# Big Data Class Software Installation Automation

## Clone Repository
1. `git clone https://github.com/mikey-joyce/hadoop_setup.git`

Note: stay in the parent directory of hadoop_setup during the install process!

i.e. If the structure is like: `/home/ubuntu/hadoop_setup`, when running the following commands run it from `/home/ubuntu/`
 
## Hadoop Install Instructions
1. `. hadoop_setup/hadoop1.sh`
2. `. hadoop_setup/hadoop2.sh`

## Pig Install Instructions
1. `. hadoop_setup/pig.sh`

To start pig do: `$PIG_HOME/bin/pig`

## HBase Install Instructions
1. `. hadoop_setup/hbase.sh`

To start hbase do: `$HBASE_HOME/bin/start-hbase.sh`

To open hbase shell do: `$HBASE_HOME/bin/hbase shell`

## Hive Install Instructions
1. `. hadoop_setup/hive.sh`

To start the hive shell do: `$HIVE_HOME/bin/hive`

## Spark Install Instructions
1. `. hadoop_setup/spark.sh`

To start spark: `$SPARK_HOME/bin/spark-shell`

## PySpark Install Instructions
1. `. hadoop_setup/pyspark.sh`
