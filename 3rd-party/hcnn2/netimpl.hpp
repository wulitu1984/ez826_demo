#pragma once
#include "hcnn2.hpp"
#include "timer.hpp"

namespace hcnn2 {

class Device;
class NetEZ82x : public NetImpl
{
public:
	NetEZ82x();
	virtual ~NetEZ82x() override;
	virtual int load(const char* net, const char* weight) override;
	virtual int unload() override;
	virtual int input(hcnn2::Blob& blob) override;
	virtual int input(hcnn2::Blob& blob, hcnn2::ROI roi) override;
	virtual int input(hcnn2::Blob& blob, std::vector<hcnn2::Point>& points, std::vector<hcnn2::Point>& ref_points) override;
	virtual std::vector<hcnn2::Blob> forward() override;
private:
	Device* _device{nullptr};
};
}
