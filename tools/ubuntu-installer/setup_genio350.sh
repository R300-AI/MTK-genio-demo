sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-pip
sudo pip install numpy

sudo apt-add-repository ppa:mediatek-genio/genio-public
# NeuronPilot
sudo apt install mediatek-vpud-genio350
sudo apt install mediatek-mdpd-genio350

# Gstreamer
sudo apt install pulseaudio
sudo apt install pulseaudio-utils
sudo apt install gstreamer1.0-alsa
sudo apt install gstreamer1.0-plugins-base
sudo apt install gstreamer1.0-plugins-good
sudo apt install gstreamer1.0-plugins-bad
sudo rm ~/.cache/gstreamer-1.0/registry.aarch64.bin
