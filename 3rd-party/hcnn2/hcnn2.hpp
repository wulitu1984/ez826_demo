#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <typeinfo>
#include <cstring>

#include "timer.hpp"
#include "mat.hpp"

namespace hcnn2 {

struct Point {
	int x;
	int y;
};

struct ROI {
	int x;
	int y;
	int w;
	int h;
};

enum class BlobType {EMPTY, TENSOR, YUV420SP, YVU420SP, RGB24, BGR24};
//shape              null , custom, whc     , whc     , whc  , whc
//data               null , 0     , 0:y 1:uv, 0:y 1:vu, 0    , 0

//Blob only process the pointer
//manange the memory malloc/free yourself
class Blob {
public:
	~Blob(){}
	Blob(){}
	Blob(const std::string name_, BlobType type_, std::vector<int> shape_, std::vector<void*> data_)
		: name(name_), type(type_), shape(shape_), data(data_){}
public:
	std::string name{""};
	BlobType type{BlobType::EMPTY};
	std::vector<int> shape;
	std::vector<void*> data;
	//data should be copied before use
	bool needCopy{true};
};

enum class NetDataLayout{NCHW, NHWC, NWHC};

struct NetConfig {
	bool outputDeQuant;
	NetDataLayout outputLayout;
};

class NetImpl {
public:
	NetImpl(){}
	virtual ~NetImpl(){}
	virtual int load(const char* net, const char* weight)=0;
	virtual int unload()=0;
	void setConfig(hcnn2::NetConfig& config) {_config = config;};
	NetConfig getConfig(void) {return _config;}
	virtual int input(hcnn2::Blob&)=0;
	virtual int input(hcnn2::Blob&, hcnn2::ROI roi)=0;
	virtual int input(hcnn2::Blob&, std::vector<hcnn2::Point>& points, std::vector<hcnn2::Point>& ref_points)=0;
	virtual std::vector<hcnn2::Blob> forward()=0;
private:
	hcnn2::NetConfig _config;
};

}//namespace
