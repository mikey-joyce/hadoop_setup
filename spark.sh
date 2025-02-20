#!/bin/bash

# Download Apache Spark 2.2.0
wget https://archive.apache.org/dist/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz

# Extract the downloaded archive
tar xvfz spark-2.2.0-bin-hadoop2.7.tgz

# Rename extracted directory
mv spark-2.2.0-bin-hadoop2.7 spark

# Add SPARK_HOME to bashrc and source it
echo 'export SPARK_HOME=/home/ubuntu/spark' >> ~/.bashrc
source ~/.bashrc

rm spark-2.2.0-bin-hadoop2.7.tgz

echo "Installing pyspark..."
pip install pyspark==2.2.1

echo 'Spark install complete!'
