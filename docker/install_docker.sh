#!/bin/bash

DOCKER_PKGS=`sudo dpkg -la | grep docker`

if [ -z "${DOCKER_PKGS}" ]; then
    # Install snapd if necessary
    sudo apt update
    sudo apt install snapd
    # Install docker from snap
    sudo snap install docker
else
    echo "Docker is already installed on your system"
fi