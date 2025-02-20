sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl

# Download Python 3.6 source
curl -O https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz

# Extract the downloaded tarball
tar -xf Python-3.6.15.tgz

# Navigate into the directory
cd Python-3.6.15

# Configure the build
./configure --enable-optimizations

# Build and install Python
sudo make -j 4
sudo make altinstall

sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.6 1
sudo update-alternatives --set python /usr/local/bin/python3.6
python --version

python -m pip install pyspark==2.2.1

cd ..
rm Python-3.6.15.tgz