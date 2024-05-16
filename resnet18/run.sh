cd build
cmake ..
make

cd ..
#adb shell 'mkdir -p /data/ez826_demo/resnet18'
#adb shell 'mkdir -p /data/ez826_demo/resnet18/lib'
#adb shell 'rm -rf /data/ez826_demo/yolov5/images'
#adb shell 'rm -rf /data/ez826_demo/yolov5/labels'
#adb push images /data/ez826_demo/resnet18
#adb push labels /data/ez826_demo/resnet18
#adb push models /data/ez826_demo/resnet18
#adb push ../3rd-party/hcnn2/*.so /data/ez826_demo/resnet18/lib
adb push build/demo-resnet /data/ez826_demo/resnet18
adb push build/demo-resnet-power /data/ez826_demo/resnet18
#adb push test.jpg /data/ez826_demo/resnet18

