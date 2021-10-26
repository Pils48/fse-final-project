#!/bin/bash

PROJECT_PATH=`cd ~ && find "$PWD" -name "fse-final-project" -type d`

echo $PROJECT_PATH
if [ -n "${PROJECT_PATH}" ]; then 
    sudo docker build -t runner_image -f "${PROJECT_PATH}/docker/Dockerfile-runner" --network=host "${PROJECT_PATH}/docker"
else
  echo "Unable to find project in /home/username directory"
fi