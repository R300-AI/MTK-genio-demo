# MTK Genios i1200 Docker Build
This project is to build MTK Genios i1200 in docker container.
## _MTK Genios i1200_
_TODO_: CI/CD icons

## Features

- **build.sh** :  shell script to build MTK Genios i1200 yocto
- **docker_run.sh** : shell script to run container (TODO: replace with compose yaml file)
- **Dockerfile** :  to build docker image for MTK Genios i1200 build environment
- **env.sh** : MTK Genios i1200 yocto environment parameters which called by build.sh
- **repo.sh** : repo script to download MTK Genios i1200 yocto source codes


## Environments (Recommendations)
- **[Linux AI Server]** IP : 140.96.98.14 : create new sudo user for build personal environment
- **[tmux]** for background screen while long time build



## Installation
#### GIT Clone

```sh
git clone git@github.com:R300-AI/MTK-Docker.git
```
#### Move to Genios i1200 folder
```bash
cd i1200
```
#### download MTK Genios i1200 yocto source codes using repo script
```bash
#### IoT Yocto recipes for v23.1 release
./repo.sh

Downloading Repo source from https://gerrit.googlesource.com/git-repo

... A new version of repo (2.36) is available.
... New version is available at: /media/data1/anitawu/mtk/MTK-Docker/i1200/.repo/repo/repo
... The launcher is run from: /usr/sbin/repo
!!! The launcher is not writable.  Please talk to your sysadmin or distro
!!! to get an update installed.


Your identity is: Anita Wu <anita.wu@itri.org.tw>
If you want to change this, please re-run 'repo init' with --config-name

repo has been initialized in /media/data1/anitawu/mtk/MTK-Docker/i1200

... A new version of repo (2.36) is available.
... New version is available at: /media/data1/anitawu/mtk/MTK-Docker/i1200/.repo/repo/repo
... The launcher is run from: /usr/sbin/repo
!!! The launcher is not writable.  Please talk to your sysadmin or distro
!!! to get an update installed.

remote: Enumerating objects: 294, done.
remote: Counting objects: 100% (144/144), done.
remote: Compressing objects: 100% (48/48), done.
remote: Total 294 (delta 91), reused 138 (delta 90), pack-reused 150
Fetching: 100% (14/14), done in 21m18.725s
Checking out: 100% (14/14), done in 1.895s
repo sync has finished successfully.
```
## Build Sb52 Yocto Docker Images
Skip this step if sb52 docker images exists
```bash
## docker images name is sb52:dev
$ docker build -t i1200:dev . --no-cache
## check i1200:dev has been created
$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
i1200        dev       f11d9ff95194   6 days ago     749MB
ubuntu       18.04     f9a80a55f492   3 months ago   63.2MB
```

## Start to Build MTK Genios i1200 in Container
### 1. create new tmux screen (optional/recommendation)
```bash
$ tmux new -s i1200
```

### 2. create your own container with your username
```bash
### run docker_run.sh script
$ ./docker_run.sh
create anitawu-i1200 container
06461f40213e797322f8d5ac4732d293948873c5c6b9c9ee3042d87bc34f2488
set dev uid and gid
### check container exist
$ docker ps
CONTAINER ID   IMAGE       COMMAND       CREATED          STATUS          PORTS     NAMES
06461f40213e   i1200:dev   "/bin/bash"   23 seconds ago   Up 22 seconds             anitawu-i1200
```
### 3. enter your own container
```bash
$ docker exec -it anitawu-i1200 bash
dev@06461f40213e:~$ 
```
### 4. start to build sb52 in container
```bash
### 3. cd to iot-yocto folder
dev@7837edb4ee77:~$ cd iot-yocto/
### 4. start to build iot-yocto
./build.sh
```
### 5. exit tmux screen to do other thing with Ctrl+D (optional/recommendation)
```bash
[detached (from session i1200)]
$
```







