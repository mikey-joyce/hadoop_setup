#!/bin/bash
sudo apt install python3-pip

# Install PySpark
pip install pyspark

python3 -m pip install --upgrade setuptools pip
python3 -m pip install --upgrade pip

pip install pandas<2.1
pip install pyarrow

# Install Ray 2.5.0
# install stuff we need for transformers and other stuffs
pip install "ray[default,train,air]==2.5.0" --force-reinstall
pip install "transformers[torch]" datasets accelerate evaluate numpy==1.24 scikit-learn --force-reinstall

pip install top2vec

pip install "pydantic<2" --force-reinstall

python3 -m pip uninstall setuptools
python3 -m pip install setuptools

# Verify PySpark and Ray installations
echo 'Testing pyspark install..'
python3 -c "import pyspark; print(pyspark.__version__)"

echo 'Testing ray install..'
python3 -c "import ray; print(ray.__version__)"

echo 'Testing transformers install..'
python3 -c "import transformers; print(transformers.__version__)"

# Apparently top2vec doesn't have a version attribute ???
# echo 'Testing top2vec install..'
# python3 -c "from top2vec import top2vec; print(top2vec.__version__)"

# Optional: Set up the PySpark environment for Spark and Hadoop
echo "export PYSPARK_PYTHON=python3" >> ~/.bashrc
echo "export SPARK_HOME=/home/ubuntu/spark" >> ~/.bashrc
echo "export HADOOP_HOME=/home/ubuntu/hadoop" >> ~/.bashrc
source ~/.bashrc

# Verify everything works together
python3 -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.appName('Test').getOrCreate(); print(spark.version)"
