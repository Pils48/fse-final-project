#!/bin/bash

PROJECT_PATH=`cd ~ && find "$PWD" -name "fse-final-project" -type d`

if [ -n ${PROJECT_PATH} ]; then 
  sudo docker run --rm -it -u 1000 -v /var/run/docker.sock:/var/run/docker.sock \
              -v /usr/bin/docker:/usr/bin/docker runner_image
else
  echo "Unable to find project in /home/username directory"
fi

