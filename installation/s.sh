#!/bin/bash

# Download Spark 3.5.5
wget https://dlcdn.apache.org/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz

# Extract the Spark tar.gz file
sudo tar -xzvf spark-3.5.5-bin-hadoop3.tgz

mv spark-3.5.5-bin-hadoop3 spark

# Set up Spark environment variables
echo "export SPARK_HOME=/home/ubuntu/spark" >> ~/.bashrc
echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" >> ~/.bashrc
echo "export PYSPARK_PYTHON=python3" >> ~/.bashrc

# Reload bashrc to apply the changes
source ~/.bashrc

# Verify Spark installation
$SPARK_HOME/bin/spark-submit --version
