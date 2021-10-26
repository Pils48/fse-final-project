#!/bin/bash

PROJECT_NAME="fse-final-project"

PROJECT_PATH=`cd ~ && find "$PWD" -name "$PROJECT_NAME" -type d`

FSE_PULLED_IMAGE=`sudo docker image list | grep -o piliushok48/fse-final-project`

FSE_LOCAL_IMAGE=`sudo docker image list | grep -o fse-final-project-image`

if [ -n ${PROJECT_PATH} ]; then 
  if [ -n "${FSE_PULLED_IMAGE}" ]; then
    echo "Running pulled image $FSE_PULLED_IMAGE"
    sudo docker run -it -v $PROJECT_PATH:/project \
                --net=host \
                $FSE_PULLED_IMAGE
  elif [ -n "${FSE_LOCAL_IMAGE}" ]; then
    echo "Running local image $FSE_LOCAL_IMAGE"
    sudo docker run -it -v $PROJECT_PATH:/project \
                --net=host \
                $FSE_LOCAL_IMAGE
  else
    echo "Unable to find any relevant docker image"
  fi
else
  echo "Unable to find project in /home/username directory"
fi

