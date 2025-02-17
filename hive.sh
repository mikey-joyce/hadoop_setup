#!/bin/bash

# Download Hive
wget https://archive.apache.org/dist/hive/hive-1.2.2/apache-hive-1.2.2-bin.tar.gz

# Extract the downloaded file
tar xvfz apache-hive-1.2.2-bin.tar.gz

# Move the tarball to the hive directory
mv apache-hive-1.2.2-bin hive

# Append the HIVE_HOME export to .bashrc
echo "export HIVE_HOME=/home/ubuntu/hive" >> ~/.bashrc

# Optionally, reload .bashrc to apply changes immediately
source ~/.bashrc

# Create necessary directories in HDFS
hdfs dfs -mkdir -p /user/hive/warehouse

# Change permissions on the directories
hdfs dfs -chmod g+w /user/hive/warehouse
hdfs dfs -chmod g+w /tmp

# Open bin/hive-config.sh and add the HADOOP_HOME export statement
echo "export HADOOP_HOME=\$HADOOP_HOME" >> $HIVE_HOME/bin/hive-config.sh

# Copy the template files to the conf directory
cp $HIVE_HOME/conf/hive-default.xml.template $HIVE_HOME/conf/hive-default.xml
cp $HIVE_HOME/conf/hive-env.sh.template $HIVE_HOME/conf/hive-env.sh

# Open conf/hive-env.sh and add the HADOOP_HOME export statement
echo "export HADOOP_HOME=\$HADOOP_HOME" >> $HIVE_HOME/conf/hive-env.sh

# Run schematool to initialize the schema with Derby
$HIVE_HOME/bin/schematool -dbType derby -initSchema

echo 'Hive install complete'
