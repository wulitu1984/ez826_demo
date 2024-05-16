#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <thread>

#include "spdlog/spdlog.h"
#include "hcnn2/hcnn2.hpp"
#include "hcnn2/mat.hpp"
#include "hcnn2/timer.hpp"
#include "hcnn2/netimpl.hpp"


int main(int argc, char** argv)
{
	int cnt = std::atoi(argv[1]);

	hcnn2::NetEZ82x net;
	net.load("./models/resnet18.ezb", "./models/resnet18.bin");

	bool dequant = false;
	hcnn2::NetConfig config = {dequant, hcnn2::NetDataLayout::NWHC};
	net.setConfig(config);

	hcnn2::Timer timer("forword");
	timer.clear();

	for (int i = 0; i < cnt; i++) {
		timer.tik();
		net.forward();
		timer.tok();
		std::this_thread::sleep_for(std::chrono::milliseconds(25));
	}


	timer.stat();
	net.unload();
	return 0;
}


