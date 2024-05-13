
#pragma once

#include <chrono>
#include <iostream>
#include <string>


namespace hcnn2 {

/*
	timer that stat the task avg runing time, not thread safe
	usage:
		timer.clear();
		for(){
			timer.tik();
			task();
			timer.tok();
		}
		timer.stat();
*/

class Timer
{
public:
	Timer(){
		_name = "";
	}
	Timer(std::string name){
		_name = name;
	}
	~Timer(){}

	void set_name(std::string name){
		_name = name;
	}

	void clear()
	{
		_cnt = 0;
		_total = 0;
		_max = 0;
		_min = 0;
		_tiked = false;
	}
	void tik()
	{
		_tiked = true;
		_start = std::chrono::steady_clock::now();
	}
	void tok()
	{
		if(_tiked){
			_end = std::chrono::steady_clock::now();
			_cnt++;
			double elps = (double)std::chrono::duration_cast<std::chrono::duration<double>>(_end - _start).count()*1000;
			if(elps > _max) _max = elps;
			if((elps < _min)||(_min == 0)) _min = elps;
			_total += elps;
		}
	}
	int get_cnt()
	{
		return _cnt;
	}
	void stat()
	{
		std::cout << "====================" << _name << "====================" << std::endl;
		std::cout << "total " << _total << "ms" << std::endl;
		std::cout << "cnt " << _cnt << std::endl;
		std::cout << "avg " << _total / _cnt << "ms" << std::endl;
		std::cout << "max " << _max << "ms" << std::endl;
		std::cout << "min " << _min << "ms" << std::endl;
	}
private:

	std::string _name;
	int _cnt{0};
	bool _tiked{false};
	std::chrono::steady_clock::time_point _start;
	std::chrono::steady_clock::time_point _end;
	double _total{0};
	double _max{0};
	double _min{0};
};

}//end namespace

#define HCNN2_PROFILE(name, code, loop) { \
	timer.set_name(name); \
	timer.clear(); \
	for ( int i = 0; i < loop; i++) { \
		timer.tik(); \
		(code); \
		timer.tok(); \
	} \
	timer.stat(); \
	} \

