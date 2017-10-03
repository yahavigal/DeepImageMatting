#include "CaffeWrapper.h"

namespace caffe
{

	CaffeWrapper::CaffeWrapper(CaffeMode mode)
	{
#ifdef CPU_ONLY
		if (mode != CaffeMode_CPU)
			throw std::runtime_error("Mode not supported. Work with CPU mode");

		Caffe::set_mode(Caffe::Brew(mode));
#else
		if (mode != CaffeMode_GPU)
			throw std::runtime_error("Mode not supported. Work with GPU mode");

		Caffe::set_mode(Caffe::Brew(mode));
#endif // CPU_ONLY
	}

	CaffeWrapper::CaffeStatus CaffeWrapper::LoadModel(const std::string& model_file, const std::string& trained_file)
	{
		try
		{
			m_net.reset(new Net<float>(model_file, TEST));
			m_net->CopyTrainedLayersFrom(trained_file);

			if (m_net->num_inputs() != 1 || m_net->num_outputs() != 1)
				return CaffeStatus_ModelNotSupported;
		}
		catch (const std::exception&)
		{
			return CaffeStatus_FailedToLoadModel;
		}

		return CaffeWrapper::CaffeStatus_Success;
	}

	CaffeWrapper::CaffeStatus CaffeWrapper::RunModel(std::vector<std::string>& input_layers, float* input, int input_size)
	{
		if (m_net->num_inputs() != input_layers.size())
			return CaffeStatus_BadInput;

		boost::shared_ptr<Blob<float>> input_layer = m_net->blob_by_name(input_layers[0]);
		if(input_layer->count() != input_size)
			return CaffeStatus_BadInput;

		float* input_data = input_layer->mutable_cpu_data();
		input_data = input;

		m_net->Forward();

		return CaffeStatus_Success;
	}

	CaffeWrapper::CaffeStatus CaffeWrapper::GetOutput(std::vector<std::string>& output_layers, float* output, int output_size)
	{
		if (m_net->num_outputs() != output_layers.size())
			return CaffeStatus_BadInput;

		boost::shared_ptr<Blob<float>> output_layer = m_net->blob_by_name(output_layers[0]);
		if (output_layer->count() != output_size)
			return CaffeStatus_BadInput;

		const float* output_data = output_layer->cpu_data();
		float* output_ptr = output;
		for (int i = 0; i < output_size; i++)
		{
			*output_ptr = *output_data;
			output_data++;
			output_ptr++;
		}

		return CaffeStatus_Success;
	}

	CaffeLayerShape CaffeWrapper::GetLayerShape(const std::string& layer_name)
	{
		boost::shared_ptr<Blob<float>> layer = m_net->blob_by_name(layer_name);
		CaffeLayerShape shape;
		shape.num = layer->shape(Num);
		shape.channels = layer->shape(Channels);
		shape.height = layer->shape(Height);
		shape.width = layer->shape(Width);
		return shape;
	}

}