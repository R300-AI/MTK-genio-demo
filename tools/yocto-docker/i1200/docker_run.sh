#!/bin/bash
container_name=$USER-i1200
usrid=`id -u`
grpid=`id -g`

### check container is ready
echo "create $container_name container"
if [ `docker ps | grep -c $container_name` != 1 ]; then
    docker run -idt --name $container_name --rm \
        -v /home/$USER/.ssh:/home/dev/.ssh \
        -v $PWD:/home/dev/iot-yocto i1200:dev 
else
    echo "container $container_name existed"
fi 

if [ `docker ps | grep -c $container_name` = 1 ]; then
    echo "set dev uid and gid"
    docker exec -d $container_name usermod -u $usrid -g $grpid dev
    ####
else
    echo "container $container_name is not ready"
fi