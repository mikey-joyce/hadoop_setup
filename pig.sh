#!/bin/bash

# Download the file
wget https://archive.apache.org/dist/pig/pig-0.16.0/pig-0.16.0.tar.gz

# Extract the tar.gz file
tar xvfz pig-0.16.0.tar.gz

# Rename the extracted directory
mv pig-0.16.0 pig

# Append environment variables to .bashrc
echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> ~/.bashrc
echo 'export PIG_HOME=/home/ubuntu/pig' >> ~/.bashrc

# Source the .bashrc to apply changes
source ~/.bashrc

rm pig-0.16.0-tar.gz

# Start the Hadoop job history server
$HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
