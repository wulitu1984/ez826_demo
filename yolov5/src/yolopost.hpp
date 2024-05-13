#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "hcnn2/mat.hpp"


struct Bbox{
    float x_left,y_top,x_right,y_bottom;
    float max_conf;
    int class_id;
    Bbox(){}
    Bbox(float x_l, float y_t, float x_r, float y_b, float m_c, int c_id): x_left(x_l), y_top(y_t), x_right(x_r), y_bottom(y_b), max_conf(m_c), class_id(c_id){}
};

class YoloPost{
public:
    YoloPost(int num_classes):_num_classes(num_classes)
    {
        _bboxes.resize(_num_classes);
    }
    void forward(hcnn2::Mat<float>& output, std::vector<int> masked_anchors, int stride, float conf_thresh);
    std::vector<Bbox> getBBoxes(float nms_thresh);

private:
    // prior anchors
    std::vector<int> anchors {  10, 13, 16, 30, 33, 23,
                                30, 61, 62, 45, 59, 119,
                                116, 90, 156, 198, 373, 326};

    std::vector<int> _anchors_mask;
    std::vector<float> _masked_anchor;
    std::vector<int> _anchors;
    std::vector<std::vector<Bbox>> _bboxes;

    int _num_classes;
    int _num_anchors;
    int _stride;
    int _anchor_step;
    float _thresh = 0.6;

    // helper funcs
    void det_conf_process(hcnn2::Mat<float> &output, size_t begin, float threshold, int flag_preprocess,std::vector<float> &anchor_wh);
    void bbox_transform(hcnn2::Mat<float> &output, size_t begin);
};

