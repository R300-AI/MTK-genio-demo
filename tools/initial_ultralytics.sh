# clone the source code of ultralytics framework
rm -rf ./ultralytics

git clone https://github.com/ultralytics/ultralytics.git
mv ./ultralytics ./ultralytics_old
mv ./ultralytics_old/ultralytics ./ultralytics
rm -rf ./ultralytics_old
