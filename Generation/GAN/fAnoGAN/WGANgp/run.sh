#!/bin/sh

IMAGE_NAME=chainer5


if [ "${IMAGE_NAME}" = "" ]; then
exit 1
fi

 nvidia-docker run --rm -it \
 -v $(pwd):/usr/local/src/${IMAGE_NAME} \
 -w /usr/local/src/${IMAGE_NAME} \
 ${IMAGE_NAME} "bash"
