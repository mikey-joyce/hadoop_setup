#!/bin/bash

# Download Hadoop 3.3.6
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz

# Extract the Hadoop tar.gz file
sudo tar -xzvf hadoop-3.3.6.tar.gz

# Create symbolic links for easy access
# sudo ln -s /opt/hadoop-3.3.6 /opt/hadoop
mv hadoop-3.3.6 hadoop

echo "export JAVA_HOME=$JAVA_HOME_PATH" >> ~/.bashrc
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc

# Set up Hadoop environment variables
echo "export HADOOP_HOME=/home/ubuntu/hadoop" >> ~/.bashrc
echo "export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop" >> ~/.bashrc
echo "export HADOOP_MAPRED_HOME=$HADOOP_HOME" >> ~/.bashrc
echo "export HADOOP_YARN_HOME=$HADOOP_HOME" >> ~/.bashrc
echo "export HADOOP_COMMON_HOME=$HADOOP_HOME" >> ~/.bashrc
echo "export HADOOP_HDFS_HOME=$HADOOP_HOME" >> ~/.bashrc
echo "export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin" >> ~/.bashrc
echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> ~/.bashrc

# Reload bashrc to apply the changes
source ~/.bashrc

# Configure core-site.xml
echo '<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>' > $HADOOP_HOME/etc/hadoop/core-site.xml

# Configure hdfs-site.xml
echo '<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.name.dir</name>
    <value>/home/ubuntu/hadoop/hdfs/namenode</value>
  </property>
  <property>
    <name>dfs.data.dir</name>
    <value>/home/ubuntu/hadoop/hdfs/datanode</value>
  </property>
</configuration>' > $HADOOP_HOME/etc/hadoop/hdfs-site.xml

# Configure yarn-site.xml
echo '<configuration>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>localhost</value>
  </property>
</configuration>' > $HADOOP_HOME/etc/hadoop/yarn-site.xml

# Format HDFS NameNode
hdfs namenode -format

# Start HDFS and YARN services
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh

# Verify Hadoop installation
hadoop version

rm hadoop-3.3.6.tar.gz
