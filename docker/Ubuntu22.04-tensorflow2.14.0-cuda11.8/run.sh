name=returnn_$1
docker run \
  --shm-size=2gb \
  --name=$name \
  --gpus all \
  --rm \
  -it \
  --user $UID \
  -e HOME \
  -v $HOME:$HOME \
  --ipc=host \
  -w $PWD \
  returnn:latest
  #/bin/bash
#  --expose 6006 \
# --runtime=nvidia \