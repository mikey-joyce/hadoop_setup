#!/bin/bash

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Java (Hadoop requires Java)
sudo apt install openjdk-11-jdk -y

sudo apt install -y ssh

sudo apt install -y net-tools

sudo apt install scala

sudo apt install unzip

# Verify Java installation
java -version

echo "Generating SSH key..."
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

echo "Adding public key to authorized_keys..."
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 640 ~/.ssh/authorized_keys  # Set the correct permissions for security

echo "Testing SSH connection to localhost..."
ssh -o StrictHostKeyChecking=no localhost  # Skip first-time host key prompt