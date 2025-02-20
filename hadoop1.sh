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

echo "Installing scala..."
sudo apt install scala

echo "Installing unzip..."
sudo apt install unzip

echo "Generating SSH key..."
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

echo "Adding public key to authorized_keys..."
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 640 ~/.ssh/authorized_keys  # Set the correct permissions for security

echo "Testing SSH connection to localhost..."
ssh -o StrictHostKeyChecking=no localhost  # Skip first-time host key prompt

echo "Part 1 Complete"
