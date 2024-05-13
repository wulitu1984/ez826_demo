#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>
#include <unistd.h>

#include "spdlog/spdlog.h"
#include "hcnn2/hcnn2.hpp"
#include "hcnn2/mat.hpp"
#include "hcnn2/timer.hpp"
#include "hcnn2/netimpl.hpp"
#include "yolopost.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

int yolov5_demo(std::string path)
{
	int w, h, n;
	unsigned char* imgbuf = stbi_load(path.c_str(), &w, &h, &n, 0);
	spdlog::info("readpic w {}, h {}", w, h);

	std::vector<int> shape = {1, h, w, 3};
	hcnn2::Blob blob_in("images", hcnn2::BlobType::RGB24, shape, 
					{(void*)imgbuf, (void*)((unsigned char*)imgbuf+w*h)});

	hcnn2::NetEZ82x net;
	net.load("./models/yolov5.ezb", "./models/yolov5.bin");

	bool dequant = true;
	hcnn2::NetConfig config = {dequant, hcnn2::NetDataLayout::NWHC};
	net.setConfig(config);

	if (net.input(blob_in) != 0) {
		spdlog::error("set input error!");
		net.unload();
		free(imgbuf);
		return -1;
	}

	std::vector<hcnn2::Blob> blobs_out = net.forward();

	//post process
	YoloPost yolo_post(4);
	std::vector<std::vector<int>> mask{{0,1,2},{3,4,5},{6,7,8}};
	if (dequant) {
		float* conv_198_f32 = (float*)(blobs_out[0].data[0]);
		float* conv_205_f32 = (float*)(blobs_out[1].data[0]);
		float* conv_212_f32 = (float*)(blobs_out[2].data[0]);

		hcnn2::Mat<float> conv_198_mat(conv_198_f32, {80,48,255,1}, false);
		hcnn2::Mat<float> conv_205_mat(conv_205_f32, {40,24,255,1}, false);
		hcnn2::Mat<float> conv_212_mat(conv_212_f32, {20,12,255,1}, false);
		yolo_post.forward(conv_212_mat, mask[2], 32, 0.5);
		yolo_post.forward(conv_205_mat, mask[1], 16, 0.5);
		yolo_post.forward(conv_198_mat, mask[0],  8, 0.5);
	}
	std::vector<Bbox> bboxes = yolo_post.getBBoxes(0.45);
	std::cout << "bbox detected:" << bboxes.size() << std::endl;
	for(auto bbox:bboxes) {
		std::cout << "[" << bbox.x_left*w << "," << bbox.y_top*h << "," 
				<< bbox.x_right*w << "," << bbox.y_bottom*h << "]";
		std::cout << " " << bbox.max_conf;
		std::cout << " " << bbox.class_id << std::endl;
	}

	net.unload();
	free(imgbuf);
	return 0;
}

int main(int argc, char** argv)
{
	yolov5_demo(std::string(argv[1]));
	return 0;
}


