#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/score_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

	template <typename Dtype>
	void ScoreAccuracyLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		
	}


	template <typename Dtype>
	void ScoreAccuracyLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		vector<int> top_shape(0);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void ScoreAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Blob<Dtype>* scores = bottom[0];
		Blob<Dtype>* labels = bottom[1];
		int num = bottom[0]->shape(0);
		Dtype scoreAccuracy = 0;

		for (int i = 0; i < num; i++)
		{

			Dtype label = labels->data_at(i, 0, 0, 0);
			LOG_IF(FATAL, label != 1.0f && label != -1.0f) << "Unexpected labels " << label;
			
			Dtype score = scores->data_at(i, 0, 0, 0);
			
			if (score*label >= 0.0f)
				scoreAccuracy++;
		}
		top[0]->mutable_cpu_data()[0] = scoreAccuracy / num;
	}


#ifdef CPU_ONLY
STUB_GPU(ScoreAccuracyLayer);
#endif
	INSTANTIATE_CLASS(ScoreAccuracyLayer);
	REGISTER_LAYER_CLASS(ScoreAccuracy);

}  // namespace caffe
