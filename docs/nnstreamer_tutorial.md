### Requirements
* Ubuntu
* TFLite

### Step 1. Install Gstreamer
```
$ sudo apt update && sudo apt upgrade
$ sudo apt-get install v4l-utils
$ sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```

### Step 2. Install Gstreamer and Examples to `/usr/lib/nnstreamer/bin`
```
$ sudo add-apt-repository ppa:nnstreamer/ppa
$ sudo apt-get update
$ sudo apt-get install nnstreamer
$ sudo apt-get install nnstreamer-example
```
