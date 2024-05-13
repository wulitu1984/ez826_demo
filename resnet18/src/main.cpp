#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <unistd.h>

#include "spdlog/spdlog.h"
#include "hcnn2/hcnn2.hpp"
#include "hcnn2/mat.hpp"
#include "hcnn2/timer.hpp"
#include "hcnn2/netimpl.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"


int resnet_demo(std::string path)
{
	int w, h, n;
	unsigned char* imgbuf = stbi_load(path.c_str(), &w, &h, &n, 0);
	spdlog::info("readpic w {}, h {}", w, h);

	std::vector<int> shape = {1, h, w, 3};
	hcnn2::Blob blob_in("data", hcnn2::BlobType::RGB24, shape, 
					{(void*)imgbuf, (void*)((unsigned char*)imgbuf+w*h)});

	hcnn2::NetEZ82x net;
	net.load("./models/resnet18.ezb", "./models/resnet18.bin");

	hcnn2::NetConfig config = {true, hcnn2::NetDataLayout::NWHC};
	net.setConfig(config);

	if (net.input(blob_in) != 0) {
		spdlog::error("set input error!");
		net.unload();
		free(imgbuf);
		return -1;
	}

	std::vector<hcnn2::Blob> blobs_out = net.forward();

	for (auto blob_out : blobs_out) {
		spdlog::info("net output:  {}-{}-{}-{}", 
					blob_out.shape[0], blob_out.shape[1], blob_out.shape[2], blob_out.shape[3]);
		hcnn2::Mat<float> output((float*)blob_out.data[0], {1,1,1,1000});
		output.tofile("output.bin");
		int max_id = -1;
		float max = 0;
		for(int i=0;i<1000;++i) {
			if (output(0,0,0,i) > max) {
				max_id = i;
				max = output(0,0,0,i);
			}
		}
		spdlog::info("max:  {}-{}", max_id, max);
	}


	net.unload();
	free(imgbuf);
	return 0;
}

int main(int argc, char** argv)
{
	resnet_demo(std::string(argv[1]));
	return 0;
}


