name=returnn_$1
docker run \
  --shm-size=2gb \
  --name=$name \
  --gpus all \
  --rm \
  -it \
  -e HOME \
  -v $HOME:$HOME \
  --ipc=host \
  -w $PWD \
  returnn:latest
  #/bin/bash
#  --expose 6006 \
# --runtime=nvidia \
#  --user $UID \