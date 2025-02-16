#!/bin/bash

set -e  # Exit if any command fails

echo "Downloading Hadoop 2.8.1 tarball..."
wget https://archive.apache.org/dist/hadoop/core/hadoop-2.8.1/hadoop-2.8.1.tar.gz

echo "Extracting Hadoop tarball..."
tar -xvzf hadoop-2.8.1.tar.gz

echo "Renaming Hadoop directory..."
mv hadoop-2.8.1 hadoop

echo "Appending environment variables to ~/.bashrc..."
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
echo 'export HADOOP_HOME=/home/ubuntu/hadoop' >> ~/.bashrc  # Updated HADOOP_HOME path
echo 'export HADOOP_INSTALL=$HADOOP_HOME' >> ~/.bashrc
echo 'export HADOOP_MAPRED_HOME=$HADOOP_HOME' >> ~/.bashrc
echo 'export HADOOP_COMMON_HOME=$HADOOP_HOME' >> ~/.bashrc
echo 'export HADOOP_HDFS_HOME=$HADOOP_HOME' >> ~/.bashrc
echo 'export HADOOP_YARN_HOME=$HADOOP_HOME' >> ~/.bashrc
echo 'export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native' >> ~/.bashrc
echo 'export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin' >> ~/.bashrc
echo 'export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"' >> ~/.bashrc

echo "Reloading .bashrc..."
source ~/.bashrc

echo "Part 2 Complete"
