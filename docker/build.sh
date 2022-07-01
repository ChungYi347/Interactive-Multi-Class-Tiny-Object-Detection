#!/usr/bin/env bash
DIR_NAME=${1:?No input given directory image}
SRC_IMG_NAME="interactivedetection:${DIR_NAME%/}"
docker build $DIR_NAME --tag $SRC_IMG_NAME

# JUST run, nothing to change
USER_ID=$UID
USER_NAME=$USER
USER_HOME=/home/$USER_NAME
USER_SHELL=$SHELL
GROUP_ID=9001
GROUP_NAME=user
DST_IMG_NAME=$USER/$(echo $SRC_IMG_NAME | cut -d "_" -f 2)
docker build \
    --build-arg USER_ID=$USER_ID \
    --build-arg USER_NAME=$USER_NAME \
    --build-arg USER_HOME=$USER_HOME \
    --build-arg USER_SHELL=$USER_SHELL \
    --build-arg GROUP_ID=$GROUP_ID \
    --build-arg GROUP_NAME=$GROUP_NAME \
    --build-arg SRC_IMG_NAME=$SRC_IMG_NAME \
    --file Dockerfile-user ./ \
    --tag $DST_IMG_NAME
