## Submitting Spark Job With Log

example: `$SPARK_HOME/bin/spark-submit <Python Script> > <Filename>.log 2>&1`

## Phase 1 Commands

- Task 1: Extract hashtags and urls from tweets.json:
  
`$SPARK_HOME/bin/spark-submit extract_hashurl.py > hashtags_urls.log 2>&1`

- Task 2: Use hadoop wordcount map reduce task and generate the log:

`hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar wordcount /phase1/hashtags_urls/ /phase1/results/hadoop_wordcount > hadoop_wordcount.log 2>&1`

- Task 3: Use spark to create a map reduce wordcount task and generate the log:

`$SPARK_HOME/bin/spark-submit spark_wordcount.py > spark_wordcount.log 2>&1`

## Phase 2 Commands

Note: Ensure you ran the to_hdfs.sh script located in the data directory of this project

- Format all of the data:
`$SPARK_HOME/bin/spark-submit parse_data.py > hashtags_urls.log 2>&1`
