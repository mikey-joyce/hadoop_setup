#!/bin/bash

set -e  # Exit if any command fails

echo "Updating package lists..."
sudo apt update

echo "Installing OpenJDK 8..."
sudo apt install -y openjdk-8-jdk

echo "Installing SSH..."
sudo apt install -y ssh

echo "Installing net-tools..."
sudo apt install -y net-tools

echo "Generating SSH key..."
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

echo "Adding public key to authorized_keys..."
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 640 ~/.ssh/authorized_keys  # Set the correct permissions for security

echo "Testing SSH connection to localhost..."
ssh -o StrictHostKeyChecking=no localhost  # Skip first-time host key prompt

echo "Downloading Hadoop 2.8.1 tarball..."
wget https://archive.apache.org/dist/hadoop/core/hadoop-2.8.1/hadoop-2.8.1.tar.gz

echo "Extracting Hadoop tarball..."
tar -xvzf hadoop-2.8.1.tar.gz

echo "Renaming Hadoop directory..."
mv hadoop-2.8.1 hadoop

echo "Appending environment variables to ~/.bashrc..."
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
echo 'export HADOOP_HOME=/home/ubuntu/hadoop/hadoop-2.8.1' >> ~/.bashrc  # Updated HADOOP_HOME path
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
    <value>HADOOP_MAPRED_HOME=$HADOOP_HOME/home/ubuntu/hadoop/hadoop-2.8.1/bin/hadoop</value>
  </property>
  <property>
    <name>mapreduce.map.env</name>
    <value>HADOOP_MAPRED_HOME=$HADOOP_HOME/home/ubuntu/hadoop/hadoop-2.8.1/bin/hadoop</value>
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
