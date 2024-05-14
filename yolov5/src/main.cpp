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
#include "mAP.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

static int RGB2BGR(unsigned char* buf, int width, int height) {
	unsigned char tmp;
	for (int i = 0; i < width * height * 3; i += 3)
	{
		tmp = buf[i];
		buf[i] = buf[i + 2];
		buf[i + 2] = tmp;
	}
	return 0;
}

int yolov5_demo(hcnn2::NetEZ82x &net, hcnn2::Timer &timer, std::string image_path, std::string result_path)
{
	int w, h, n;
	unsigned char* imgbuf = stbi_load(image_path.c_str(), &w, &h, &n, 0);
	spdlog::info("image size: {}x{}x{}", w, h, n);
	if (n != 3) {
		spdlog::error("image size error!");
		free(imgbuf);
		return -1;
	}

	RGB2BGR((unsigned char*)imgbuf, w, h);

	std::vector<int> shape = {1, h, w, 3};
	hcnn2::Blob blob_in("input.1", hcnn2::BlobType::RGB24, shape, 
					{(void*)imgbuf, (void*)((unsigned char*)imgbuf+w*h)});

	if (net.input(blob_in) != 0) {
		spdlog::error("set input error!");
		free(imgbuf);
		return -1;
	}

	timer.tik();
	std::vector<hcnn2::Blob> blobs_out = net.forward();
	timer.tok();

	//post process
	YoloPost yolo_post(80);
	std::vector<std::vector<int>> mask{{0,1,2},{3,4,5},{6,7,8}};
	float* conv_198_f32 = (float*)(blobs_out[0].data[0]);
	float* conv_205_f32 = (float*)(blobs_out[1].data[0]);
	float* conv_212_f32 = (float*)(blobs_out[2].data[0]);

	hcnn2::Mat<float> conv_198_mat(conv_198_f32, {80,80,255,1}, false);
	hcnn2::Mat<float> conv_205_mat(conv_205_f32, {40,40,255,1}, false);
	hcnn2::Mat<float> conv_212_mat(conv_212_f32, {20,20,255,1}, false);
	yolo_post.forward(conv_212_mat, mask[2], 32, 0.5);
	yolo_post.forward(conv_205_mat, mask[1], 16, 0.5);
	yolo_post.forward(conv_198_mat, mask[0],  8, 0.5);
	std::vector<Bbox> bboxes = yolo_post.getBBoxes(0.45);
	std::cout << "bbox detected:" << bboxes.size() << std::endl;
	for(auto bbox:bboxes) {
		std::cout << "[" << bbox.x_left*w << "," << bbox.y_top*h << "," 
				<< bbox.x_right*w << "," << bbox.y_bottom*h << "]";
		std::cout << " " << bbox.max_conf;
		std::cout << " " << bbox.class_id << std::endl;
	}
	//write result
	std::ofstream result_file(result_path);
	for(auto bbox:bboxes) {
		result_file << bbox.class_id << " " << bbox.max_conf << " " << bbox.x_left*w << " " << bbox.y_top*h 
				<< " " << bbox.x_right*w << " " << bbox.y_bottom*h << std::endl;
	}

	free(imgbuf);
	return 0;
}


int main(int argc, char** argv)
{
	hcnn2::NetEZ82x net;
	net.load("./models/yolov5s.ezb", "./models/yolov5s.bin");

	bool dequant = true;
	hcnn2::NetConfig config = {dequant, hcnn2::NetDataLayout::NWHC};
	net.setConfig(config);

	hcnn2::Timer timer("forword");
	timer.clear();

	auto labels = get_files_in_path("./labels");
	for (auto l:labels) {
		std::string label = l.substr(9, l.size()-13);
		std::cout << label << std::endl;
		std::string image_path = "./images/" + label + ".jpg";
		std::string result_path = "./result/" + label + ".txt";
		yolov5_demo(net, timer, image_path, result_path);
	}

    std::string ground_truths_path = "./labels";
    std::string detection_results_path = "./result";
    std::vector<std::pair<int, float>> map = calc_mAP(ground_truths_path, detection_results_path);
	for (auto m:map) {
		std::cout << "class_id:" << m.first << " mAP:" << m.second << std::endl;
	}

	timer.stat();
	net.unload();
	return 0;
}


