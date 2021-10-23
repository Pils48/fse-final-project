#!/bin/bash

PROJECT_PATH=`cd ~ && find "$PWD" -name "fse-final-project" -type d`

echo $PROJECT_PATH
if [ -n "${PROJECT_PATH}" ]; then 
    sudo docker build -t fse-final-project-image --network=host "${PROJECT_PATH}/docker"
else
  echo "Unable to find project in /home/username directory"
fi