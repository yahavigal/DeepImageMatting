#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include "caffe/net.hpp"
#include "CaffeLayerShape.h"

namespace caffe
{

	class CaffeWrapper
	{
	public:
		enum CaffeMode
		{
			CaffeMode_CPU = 0,
			CaffeMode_GPU
		};

		enum CaffeStatus
		{
			CaffeStatus_Success,
			CaffeStatus_FailedToLoadModel,
			CaffeStatus_ModelNotSupported,
			CaffeStatus_BadInput
		};

		enum LayerShapeIndex
		{
			Num = 0,
			Channels,
			Height,
			Width
		};

		CaffeWrapper(CaffeMode mode);
		CaffeStatus LoadModel(const std::string& model_file, const std::string& trained_file);
		CaffeStatus RunModel(std::vector<std::string>& input_layers, float* input, int input_size);
		CaffeStatus GetOutput(std::vector<std::string>& output_layers, float* output, int output_size);
		CaffeLayerShape GetLayerShape(const std::string& layer_name);

	private:
		std::shared_ptr<caffe::Net<float>> m_net;
	};

}