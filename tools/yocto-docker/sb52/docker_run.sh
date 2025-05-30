#!/bin/bash
container_name=$USER-sb52
usrid=`id -u`
grpid=`id -g`

### check container is ready
if [ `docker ps | grep -c $container_name` = 1 ]; then
    echo "set dev uid and gid"
    docker exec -d $container_name usermod -u $uesrid -g $grpid dev
else
    docker run -idt --name $container_name --rm -v $PWD:/home/dev/sb52-yocto sb52:dev
fi 