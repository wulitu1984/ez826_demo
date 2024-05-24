cd build
cmake .. -DPLATFORM=$1
make

cd ..
#adb shell 'mkdir -p /data/ez826_demo/resnet18'
#adb shell 'mkdir -p /data/ez826_demo/resnet18/lib'
#adb shell 'rm -rf /data/ez826_demo/yolov5/images'
#adb shell 'rm -rf /data/ez826_demo/yolov5/labels'
#adb push images /data/ez826_demo/resnet18
#adb push labels /data/ez826_demo/resnet18
if [ $1 == "82x" ]
then
	adb push ../3rd-party/hcnn2/lib82x/*.so /data/ez826_demo/resnet18/lib
	adb push models/resnet18.ezb /data/ez826_demo/resnet18/models/resnet18.ezb
	adb push models/resnet18.bin /data/ez826_demo/resnet18/models/resnet18.bin
else
	adb push ../3rd-party/hcnn2/lib51x/*.so /data/ez826_demo/resnet18/lib
	adb push models/resnet18_51x.ezb /data/ez826_demo/resnet18/models/resnet18.ezb
	adb push models/resnet18_51x.bin /data/ez826_demo/resnet18/models/resnet18.bin
fi
adb push build/demo-resnet /data/ez826_demo/resnet18
adb push build/demo-resnet-power /data/ez826_demo/resnet18

