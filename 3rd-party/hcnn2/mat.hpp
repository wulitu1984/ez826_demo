#pragma once
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <assert.h>

namespace hcnn2 {
template<typename T>
class Mat{
public:
	~Mat() {};
	Mat(T* data, std::vector<size_t> shape, bool idx_reverse=false) {
		_shape= shape;
		_data = data;
		_idx_reverse = idx_reverse;
		_calc_stride();
	}
	Mat(const Mat& lhs) {
		_shape= lhs._shape;
		_data = lhs._data;
		_stride = lhs._stride;
		_idx_reverse = lhs._idx_reverse;
	}
	Mat& operator=(const Mat& lhs) {
		_shape= lhs._shape;
		_data = lhs._data;
		_stride = lhs._stride;
		_idx_reverse = lhs._idx_reverse;
		return *this;
	}
	Mat& operator=(Mat&& rhs) {
		this->_shape= rhs._shape;
		this->_data = rhs._data;
		this->_stride = rhs._stride;
		this->_idx_reverse = rhs._idx_reverse;
		return *this;
	}
	std::vector<size_t> shape() {
		return _shape;
	}

	bool reverse() {
		return _idx_reverse;
	}

	T& operator()(size_t i, size_t j=0, size_t k=0, size_t h=0){
		if(_idx_reverse) {
			return _data[h*_stride[0]+k*_stride[1]+j*_stride[2]+ i*_stride[3]];
		} else {
			return _data[i*_stride[0]+j*_stride[1]+k*_stride[2]+ h*_stride[3]];
		}
	}

	void tofile(const std::string& inFilename)
	{
		std::ofstream ofile(inFilename, std::ios::binary);
		if (!ofile.good())
		{
			spdlog::error("Unable to open the input file: {}", inFilename);
		}
		if (_data != nullptr)
		{
			ofile.write(reinterpret_cast<const char*>(_data), _size * sizeof(T));
		}
		ofile.close();
	}

private:
	bool _idx_reverse{false};
	std::vector<size_t> _shape;
	std::vector<size_t> _stride;
	size_t _size;
	T* _data{nullptr};

	void _calc_stride() {
		size_t dims = _shape.size();
		_stride.resize(4);
		_size = 1;
		for(size_t t=dims; t>0; --t)
		{
			if(t==dims)
			{
				_stride[0] = 1;
				continue;
			}
			_stride[3] = _stride[2];
			_stride[2] = _stride[1];
			_stride[1] = _stride[0];
			_stride[0] *= _shape[t];
		}
		_size *= _stride[0] * _shape[0];
	}
};
}
