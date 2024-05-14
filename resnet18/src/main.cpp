#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
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

int resnet_demo(hcnn2::NetEZ82x &net, std::string path)
{
	int w, h, n;
	unsigned char* imgbuf = stbi_load(path.c_str(), &w, &h, &n, 0);
	// spdlog::info("image {}, w {}, h {}, n {}", path, w, h, n);	
	if (n == 3)
		RGB2BGR((unsigned char*)imgbuf, w, h);

	std::vector<int> shape = {1, h, w, 3};
	hcnn2::Blob blob_in("data", hcnn2::BlobType::RGB24, shape, 
					{(void*)imgbuf, (void*)((unsigned char*)imgbuf+w*h)});

	if (net.input(blob_in) != 0) {
		spdlog::error("set input error!");
		free(imgbuf);
		return -1;
	}

	std::vector<hcnn2::Blob> blobs_out = net.forward();

	auto blob_out = blobs_out[0];
	hcnn2::Mat<float> output((float*)blob_out.data[0], {1,1,1,1000});
	// output.tofile("output.bin");
	int max_id = -1;
	float max = 0;
	for(int i=0;i<1000;++i) {
		if (output(0,0,0,i) > max) {
			max_id = i;
			max = output(0,0,0,i);
		}
	}

	free(imgbuf);
	return max_id;
}

int main(int argc, char** argv)
{
	hcnn2::NetEZ82x net;
	net.load("./models/resnet18.ezb", "./models/resnet18.bin");
	hcnn2::NetConfig config = {true, hcnn2::NetDataLayout::NWHC};
	net.setConfig(config);

	std::ifstream ifs("./labels/val.txt");
	std::string line;
	int total = 0;
	int correct = 0;
	float acc_top1 = 0;

	while (std::getline(ifs, line))
    {
    	std::istringstream iss(line);
		std::string fname;
		iss >> fname;
		int cls;
		iss >> cls;
		int pred = resnet_demo(net, "./images/"+fname);
		spdlog::info("image {}, label {}, pred {}", fname, cls, pred);	
		if (pred == cls) {
			correct++;
		}
		total++;
	}

	acc_top1 = (float)correct / total;
	spdlog::info("total {}, correct {}, acc_top1 {}", total, correct, acc_top1);

	net.unload();
	return 0;
}


