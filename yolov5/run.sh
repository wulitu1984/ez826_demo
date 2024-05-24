cd build
cmake .. -DPLATFORM=$1
make

cd ..
#adb shell 'mkdir -p /data/ez826_demo/yolov5'
#adb shell 'mkdir -p /data/ez826_demo/yolov5/lib'
#adb shell 'rm -rf /data/ez826_demo/yolov5/images'
#adb shell 'rm -rf /data/ez826_demo/yolov5/labels'
#adb shell 'rm -rf /data/ez826_demo/yolov5/result'
#adb shell 'mkdir -p /data/ez826_demo/yolov5/result'
#adb push images /data/ez826_demo/yolov5
#adb push labels /data/ez826_demo/yolov5
#if [ $1 == "82x" ]
#then
	#adb push ../3rd-party/hcnn2/lib82x/*.so /data/ez826_demo/yolov5/lib
	#adb push models/yolov5s.ezb /data/ez826_demo/yolov5/models/yolov5s.ezb
	#adb push models/yolov5s.bin /data/ez826_demo/yolov5/models/yolov5s.bin
#else
	#adb push ../3rd-party/hcnn2/lib51x/*.so /data/ez826_demo/yolov5/lib
	#adb push models/yolov5s_51x.ezb /data/ez826_demo/yolov5/models/yolov5s.ezb
	#adb push models/yolov5s_51x.bin /data/ez826_demo/yolov5/models/yolov5s.bin
#fi
adb push build/demo-yolov5 /data/ez826_demo/yolov5
adb push build/demo-yolov5-power /data/ez826_demo/yolov5

