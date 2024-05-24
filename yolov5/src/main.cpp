#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>
#include <unistd.h>
#include <cstdlib>

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

#define USE_CV2

#ifdef USE_CV2
#include "opencv2/opencv.hpp"
#endif

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
	#ifdef USE_CV2
	auto im = cv::imread(image_path);
	w = im.cols;
	h = im.rows;
	n = 3;
	unsigned char* imgbuf = im.data;
	#else
	unsigned char* imgbuf = stbi_load(image_path.c_str(), &w, &h, &n, 0);
	RGB2BGR((unsigned char*)imgbuf, w, h);
	#endif

	if (n != 3) {
		spdlog::error("image size error!");
		std::ofstream result_file(result_path);
		result_file << "#" << std::endl;
		result_file.close();
		#ifndef USE_CV2
		free(imgbuf);
		#endif
		return -1;
	}


	std::vector<int> shape = {1, h, w, 3};
	hcnn2::Blob blob_in("input.1", hcnn2::BlobType::RGB24, shape, 
					{(void*)imgbuf, (void*)((unsigned char*)imgbuf+w*h)});

	if (net.input(blob_in) != 0) {
		spdlog::error("set input error!");
		#ifndef USE_CV2
		free(imgbuf);
		#endif
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
	//spdlog::info("{}-{}-{}-{}", blobs_out[0].shape[0], blobs_out[0].shape[1], blobs_out[0].shape[2], blobs_out[0].shape[3]);
	//spdlog::info("{}-{}-{}-{}", blobs_out[1].shape[0], blobs_out[1].shape[1], blobs_out[1].shape[2], blobs_out[1].shape[3]);
	//spdlog::info("{}-{}-{}-{}", blobs_out[2].shape[0], blobs_out[2].shape[1], blobs_out[2].shape[2], blobs_out[2].shape[3]);

	hcnn2::Mat<float> conv_198_mat(conv_198_f32, {80,80,255,1}, false);
	hcnn2::Mat<float> conv_205_mat(conv_205_f32, {40,40,255,1}, false);
	hcnn2::Mat<float> conv_212_mat(conv_212_f32, {20,20,255,1}, false);
	yolo_post.forward(conv_212_mat, mask[2], 32, 0.1);
	yolo_post.forward(conv_205_mat, mask[1], 16, 0.1);
	yolo_post.forward(conv_198_mat, mask[0],  8, 0.1);
	std::vector<Bbox> bboxes = yolo_post.getBBoxes(0.6);
	//std::cout << "bbox detected:" << bboxes.size() << std::endl;
	//for(auto bbox:bboxes) {
		//std::cout << "[" << bbox.x_left*w << "," << bbox.y_top*h << "," 
				//<< bbox.x_right*w << "," << bbox.y_bottom*h << "]";
		//std::cout << " " << bbox.max_conf;
		//std::cout << " " << bbox.class_id << std::endl;
	//}
	//write result
	std::ofstream result_file(result_path);
	for(auto bbox:bboxes) {
		result_file << bbox.class_id << " " << bbox.max_conf << " " << bbox.x_left*w << " " << bbox.y_top*h 
				<< " " << bbox.x_right*w << " " << bbox.y_bottom*h << std::endl;
	}
	//add comment line to make sure the result file is the same as the ground truth file
	result_file << "#" << std::endl;
	result_file.close();

	#ifndef USE_CV2
	free(imgbuf);
	#endif
	return bboxes.size();
}


int main(int argc, char** argv)
{
	int cnt = std::atoi(argv[1]);

	hcnn2::NetEZ82x net;
	net.load("./models/yolov5s.ezb", "./models/yolov5s.bin");

	bool dequant = true;
	hcnn2::NetConfig config = {dequant, hcnn2::NetDataLayout::NWHC};
	net.setConfig(config);

	hcnn2::Timer timer("forword");
	timer.clear();

	auto labels = get_files_in_path("./labels");
	system("mkdir -p labels_1");
	system("mkdir -p result_1");
	system("rm -rf labels_1/*");
	system("rm -rf result_1/*");
	for (auto l:labels) {
		std::string label = l.substr(9, l.size()-13);
		std::string image_path = "./images/" + label + ".jpg";
		std::string cmd = "cp labels/" + label + ".txt" + " labels_1/" + label + ".txt";
		system(cmd.c_str());
		std::string result_path = "./result_1/" + label + ".txt";
		auto ret = yolov5_demo(net, timer, image_path, result_path);
		if (ret == -1) {
			break;
		}
		spdlog::info("cnt {}, image_path {}, result_path {}, bboxes detected {}", cnt, image_path, result_path, ret);
		cnt--;
		if (cnt == 0) {
			break;
		}
	}

    std::string ground_truths_path = "./labels_1";
    std::string detection_results_path = "./result_1";
	float map_all = 0;
	int cnt_all = 0;
	std::cout << "================ calc mAP ================" << std::endl;
	for (int i = 50; i < 100; i+=5) {
		float iou_threshold = i / 100.0;
		std::vector<std::pair<int, float>> mAPs = calc_mAP(ground_truths_path, detection_results_path, iou_threshold);
		float map = 0;
		//int j = 0;
		for (auto m:mAPs) {
			//std::cout << "class:" << m.first << " mAP:" << m.second << " ";
			map += m.second;
			//j++;
			//if ((j % 8) == 0) std::cout << std::endl;
		}
		spdlog::info("AP{}:{}", i, map/mAPs.size());
		cnt_all++;
		map_all += map/mAPs.size();
		break;
	}
	//std::cout << "mAP50:95: " << map_all/cnt_all << std::endl;


	timer.stat();
	net.unload();
	return 0;
}


