#!/bin/bash

# Install snapd if necessary
sudo apt update
sudo apt install snapd

# Install docker from snap
sudo snap install docker