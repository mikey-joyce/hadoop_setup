#!/bin/bash

# Download the file
wget https://archive.apache.org/dist/hbase/1.2.6/hbase-1.2.6-bin.tar.gz

# Extract the tar.gz file
tar xvfz hbase-1.2.6-bin.tar.gz

# Rename the extracted directory
mv hbase-1.2.6 hbase

# Append the HBASE_HOME environment variable to .bashrc
echo 'export HBASE_HOME=/home/ubuntu/hbase' >> ~/.bashrc

# Source the .bashrc to apply changes
source ~/.bashrc

# Update hbase-site.xml with new contents using sudo tee
sudo tee $HBASE_HOME/conf/hbase-site.xml > /dev/null <<EOL
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>hbase.rootdir</name>
    <value>hdfs://localhost:9000/hbase</value>
  </property>
</configuration>
EOL
