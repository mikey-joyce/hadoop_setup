#!/bin/bash

set -e  # Exit if any command fails


echo "Appending JAVA_HOME to hadoop-env.sh..."
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' | sudo tee -a $HADOOP_HOME/etc/hadoop/hadoop-env.sh

echo "Replacing contents of core-site.xml with the new configuration..."
sudo tee $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null <<EOL
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
EOL

echo "Replacing contents of hdfs-site.xml with the new configuration..."
sudo tee $HADOOP_HOME/etc/hadoop/hdfs-site.xml > /dev/null <<EOL
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file:///home/ubuntu/hadoopdata/hdfs/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///home/ubuntu/hadoopdata/hdfs/datanode</value>
  </property>
</configuration>
EOL

echo "Creating mapred-site.xml with the new configuration..."
sudo tee $HADOOP_HOME/etc/hadoop/mapred-site.xml > /dev/null <<EOL
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>yarn.app.mapreduce.am.env</name>
    <value>HADOOP_MAPRED_HOME=$HADOOP_HOME/home/ubuntu/hadoop/bin/hadoop</value>
  </property>
  <property>
    <name>mapreduce.map.env</name>
    <value>HADOOP_MAPRED_HOME=$HADOOP_HOME/home/ubuntu/hadoop/bin/hadoop</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///home/ubuntu/hadoopdata/hdfs/datanode</value>
  </property>
</configuration>
EOL

echo "Replacing contents of yarn-site.xml with the new configuration..."
sudo tee $HADOOP_HOME/etc/hadoop/yarn-site.xml > /dev/null <<EOL
<?xml version="1.0"?>
<configuration>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
</configuration>
EOL

echo "Formatting HDFS namenode..."
hdfs namenode -format

echo "Starting HDFS..."
$HADOOP_HOME/sbin/start-dfs.sh

echo "Starting YARN..."
$HADOOP_HOME/sbin/start-yarn.sh

echo "HDFS and YARN services have been started!"