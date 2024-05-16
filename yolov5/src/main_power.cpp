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


int main(int argc, char** argv)
{
	int cnt = std::atoi(argv[1]);

	hcnn2::NetEZ82x net;
	net.load("./models/yolov5s.ezb", "./models/yolov5s.bin");

	bool dequant = false;
	hcnn2::NetConfig config = {dequant, hcnn2::NetDataLayout::NWHC};
	net.setConfig(config);

	hcnn2::Timer timer("forword");
	timer.clear();

	for (int i = 0; i < cnt; i++) {
		timer.tik();
		net.forward();
		timer.tok();
	}


	timer.stat();
	net.unload();
	return 0;
}


