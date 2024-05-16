cd build
cmake ..
make

cd ..
# adb shell 'mkdir -p /data/ez826_demo/yolov5'
# adb shell 'mkdir -p /data/ez826_demo/yolov5/lib'
 #adb shell 'rm -rf /data/ez826_demo/yolov5/images'
 #adb shell 'rm -rf /data/ez826_demo/yolov5/labels'
 #adb shell 'rm -rf /data/ez826_demo/yolov5/result'
 #adb shell 'mkdir -p /data/ez826_demo/yolov5/result'
 #adb push images /data/ez826_demo/yolov5
 #adb push labels /data/ez826_demo/yolov5
# adb push models /data/ez826_demo/yolov5
# adb push ../3rd-party/hcnn2/*.so /data/ez826_demo/yolov5/lib
adb push build/demo-yolov5 /data/ez826_demo/yolov5
adb push build/demo-yolov5-power /data/ez826_demo/yolov5
# adb push test.jpg /data/ez826_demo/yolov5

