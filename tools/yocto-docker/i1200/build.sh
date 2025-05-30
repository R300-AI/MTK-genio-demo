source env.sh
source src/poky/oe-init-build-env
touch conf/sanity.conf
DISTRO=rity-demo MACHINE=genio-1200-evk bitbake rity-demo-image