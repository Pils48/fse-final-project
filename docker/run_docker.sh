#!/bin/bash

PROJECT_PATH=`cd ~ && find "$PWD" -name "fse-final-project" -type d`

if [ -n ${PROJECT_PATH} ]; then 
  sudo docker run  -it -v $PROJECT_PATH:/project \
              --net=host \
              fse_final_project_image
else
  echo "Unable to find project in /home/username directory"
fi

