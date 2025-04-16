#!/bin/bash
sudo apt install python3-pip

# Install PySpark
pip install pyspark

python3 -m pip install --upgrade setuptools pip
python3 -m pip install --upgrade pip

# Install Ray 2.5.0
pip install ray[default]==2.5.0

pip install pandas
pip install pyarrow

# install stuff we need for transformers and other stuffs
pip install "ray[train,air]" "transformers[torch]" datasets accelerate evaluate numpy scikit-learn

# Verify PySpark and Ray installations
python3 -c "import pyspark; print(pyspark.__version__)"
python3 -c "import ray; print(ray.__version__)"

# Optional: Set up the PySpark environment for Spark and Hadoop
echo "export PYSPARK_PYTHON=python3" >> ~/.bashrc
echo "export SPARK_HOME=/home/ubuntu/spark" >> ~/.bashrc
echo "export HADOOP_HOME=/home/ubuntu/hadoop" >> ~/.bashrc
source ~/.bashrc

# Verify everything works together
python3 -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.appName('Test').getOrCreate(); print(spark.version)"
