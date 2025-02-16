#!/bin/bash

# Run the first script
echo "Running setup script..."
. hadoop_setup/install_hadoop1.sh

# Check if the first script was successful
if [ $? -ne 0 ]; then
    echo "Error: install_hadoop1.sh failed!"
    exit 1
fi

# Run the second script
echo "Running hadoop installation script..."
. hadoop_setup/install_hadoop2.sh

# Check if the second script was successful
if [ $? -ne 0 ]; then
    echo "Error: install_hadoop2.sh failed!"
    exit 1
fi

echo "Hadoop install complete!"
