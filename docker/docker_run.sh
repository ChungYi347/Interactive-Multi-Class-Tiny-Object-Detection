DockerName=$USER-interactivedetection
DockerImage=$USER/interactivedetection:ubuntu18.04-cuda9.2-cudnn7.6-python3.7-pt1.4.0

docker run -it --rm \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --shm-size 128G \
  --user $USER \
  -d \
  --name $DockerName \
  $DockerImage \
  /bin/bash
