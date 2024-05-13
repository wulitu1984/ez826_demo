#include <iostream>
#include "spdlog/spdlog.h"
#include "yolopost.hpp"

template<typename T>
double sigmoid(T x) { return 1/(1+exp(-x));}
template<typename T>
double resigmoid(T y) { return -1*log(1/y-1);}

static int nms(const std::vector<Bbox> &bboxes, std::vector<Bbox> &keep_contianer, float nms_threshold){
	size_t count = bboxes.size();
	std::vector<unsigned char> suppress(count,0);
	std::vector<float> area(count, 0);

	for(size_t i=0; i<count; ++i)
	{
		const Bbox &bbox = bboxes[i];
		float w = bbox.x_right-bbox.x_left;
		float h = bbox.y_bottom - bbox.y_top;
		area[i] = (w*h);
	}

	for(size_t i=0; i<count; i++) 
	{
		if(suppress.at(i))
			continue;
		const Bbox &_l = bboxes[i];
		for( size_t j=i+1; j<count; j++)
		{
			if (suppress.at(j))
				continue;
			const Bbox &_r = bboxes[j];
			float x_1 = std::max(_l.x_left, _r.x_left);
			float x_2 = std::min(_l.x_right, _r.x_right);


			float y_1 = std::max(_l.y_top, _r.y_top);
			float y_2 = std::min(_l.y_bottom, _r.y_bottom);

			float w = x_2 - x_1;
			float h = y_2 - y_1;

			if(w<0 ||h<0) 
			{
				w=0;
				h=0;
			}
			float overlap = w*h;
			float iou = overlap/(area[i]+area[j]-overlap);
			if(iou>nms_threshold)
			{
				suppress[j] = 1;
			}
			// keep.push_back(keep_idx[i]);
		}
	}
	// keep.reserve(count/2);
	int cnt = 0;
	for(size_t i=0; i< suppress.size(); ++i)
	{
		if(suppress[i]==0)
		{
			keep_contianer.push_back(bboxes[i]);
			cnt++;
		}
	}

	return cnt;
}
void YoloPost::det_conf_process(hcnn2::Mat<float>& features, size_t begin, float threshold, int flag_preproces,std::vector<float> &anchor_wh)
{
	std::vector<size_t> shape = features.shape();
	hcnn2::Mat<float> output = features;
	size_t hs = (output.reverse()) ? shape[2] : shape[1];
	size_t ws = (output.reverse()) ? shape[3] : shape[0];
	float rethreshold = resigmoid(threshold);
	for(size_t w=0; w<ws; ++w)
	{
		for(size_t h=0; h<hs; ++h)
		{
			if(output(w, h, begin+0)<rethreshold)
				continue;
			// for def conf
			output(w, h, begin+0) = sigmoid(output(w, h, begin+0));
			int max_id = 0;
			float max_conf = 0.0;
			for(size_t c=begin+1; c<begin+_num_classes+1; c++)
			{
				// for classes conf
				output(w, h, c) = sigmoid(output(w, h, c));
				output(w, h, c) *= output(w, h, begin+0);
				if( max_conf < output(w, h, c))
				{
					max_conf = output(w, h, c);
					max_id = c-begin-1;
				}
			}
			if(max_conf>threshold)
			{
				// opt. todo
				float x_left,y_top, x_right,y_bottom;
				float &x_c	= output(w, h, begin-4);
				float &y_c	= output(w, h, begin-3);
				float &weight = output(w, h, begin-2);
				float &height = output(w, h, begin-1);
				if( flag_preproces!=1 )
				{
					x_c = (2*sigmoid(x_c)-0.5+w)/ws;
					y_c = (2*sigmoid(y_c)-0.5+h)/hs;
					weight = ((sigmoid(weight)*2)*(sigmoid(weight)*2)*anchor_wh[0])/ws;
					height = ((sigmoid(height)*2)*(sigmoid(height)*2)*anchor_wh[1])/hs;
					x_c = x_c-weight*0.5;
					y_c = y_c-height*0.5;
					weight = x_c+weight;
					height = y_c+height;

				}
				
				x_left = x_c;
				y_top = y_c;
				x_right = weight;
				y_bottom = height;
				
				Bbox tmp{x_left,y_top, x_right, y_bottom,max_conf,max_id};
				_bboxes[max_id].push_back(tmp);
			}
		}
	}
}

void YoloPost::forward(hcnn2::Mat<float>& yolo_head, std::vector<int> masked_anchors, int stride, float conf_thresh){
	std::vector<float> anchors_tmp;
	for(auto idx : masked_anchors)
	{
		anchors_tmp.push_back((float)anchors[idx*2]/stride);
		anchors_tmp.push_back((float)anchors[idx*2+1]/stride);
	}
	for(size_t i=0; i<masked_anchors.size(); ++i)
	{
		int begin = i*(5+_num_classes);
		std::vector<float> anchor_wh{anchors_tmp[i*2], anchors_tmp[i*2+1]};
		det_conf_process(yolo_head, begin+4,conf_thresh, 0, anchor_wh);
	}
}

std::vector<Bbox> YoloPost::getBBoxes(float nms_thresh){
	std::vector<Bbox> res;
	for(size_t idx=0; idx<_bboxes.size(); ++idx)
	{
		std::sort(_bboxes[idx].begin(), _bboxes[idx].end(), [](Bbox _l, Bbox _r){return _l.max_conf> _r.max_conf;});
		nms(_bboxes[idx], res, nms_thresh);
	}
	for(auto elem: _bboxes) elem.clear();
	return res;
}

