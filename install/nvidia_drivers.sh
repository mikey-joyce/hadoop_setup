#!/bin/bash

sudo apt update
sudo apt install nvidia-driver-535 nvidia-utils-535
sudo apt install nvidia-cuda-toolkit
sudo modprobe nvidia

