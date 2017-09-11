#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/mask_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe 
{
void printMask(const cv::Mat& mask)
{
	for (int i=0;i<mask.rows;i++)
	{
		std::cout<<std::endl;
		for (int j=0;j<mask.cols;j++)
		{
			float value = mask.at<float>(i,j);
			std::cout<<value<<" ";
		}
	}
	std::cout<<"\n---------------------------------------------------------------------------------------------\n";
}
	template <typename Dtype>
	void MaskAccuracyLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
    
	}


template <typename Dtype>
void MaskAccuracyLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  vector<int> top_shape(0);
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MaskAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	CHECK_EQ(bottom[0]->shape(1), 1);
	CHECK_EQ(bottom[0]->shape(3), 224);
	CHECK_EQ(bottom[0]->shape(2), 224);

	CHECK_EQ(bottom[1]->shape(2), 224);
	CHECK_EQ(bottom[1]->shape(3), 224);
	CHECK_EQ(bottom[1]->shape(1), 1);

	Blob<Dtype>* predictions = bottom[0];
	Blob<Dtype>* masks = bottom[1];
  Blob<Dtype>* labels = bottom[2];

	int num = bottom[0]->shape(0);
	Dtype maskAccuracy = 0;
  Dtype rellevantMasks = 0;
	Dtype threshold = 0.0f;

	for (int i = 0; i < num; i++)
	{
    Dtype label = labels->data_at(i, 0, 0, 0);
    if (label==-1.0f)
      continue;
    rellevantMasks++;
		int offsetPred = predictions->offset(i);
		int offsetMask = masks->offset(i);
		CHECK_EQ(offsetPred, 224 * 224 * i);
		CHECK_EQ(offsetMask, 224 * 224 * i);

		cv::Mat cvPrediction(predictions->shape(2), predictions->shape(3), CV_32FC1, (void*)(predictions->cpu_data() + offsetPred)), cvPredictionCopy, cvPredictionTemp;
		cvPrediction.copyTo(cvPredictionCopy);
		//printMask(cvPredictionCopy);
		cvPredictionCopy.setTo(-1, cvPredictionCopy <= threshold);
		cvPredictionCopy.setTo(1, cvPredictionCopy > threshold);
		CHECK_EQ(cvPredictionCopy.rows, 224);
		CHECK_EQ(cvPredictionCopy.cols, 224);


		cv::Mat cvMask(masks->shape(2), masks->shape(3), CV_32FC1, (void*)(masks->cpu_data() + offsetMask)), cvMaskCopy;
		cvMask.copyTo(cvMaskCopy);
	//	cv::resize(cvMaskCopy, cvMaskCopy, cv::Size(56, 56), cv::INTER_NEAREST);

		cvMaskCopy.setTo(1, cvMaskCopy > 0);
		cvMaskCopy.setTo(-1, cvMaskCopy == 0);

		Dtype mistakesCouter = 0;
		//printMask(cvMaskCopy);
		//LOG_IF(FATAL ,true);
		for (int k = 0; k<224; k++)
		{
			for (int j = 0; j<224; j++)
			{
				Dtype mask_kj = cvMaskCopy.at<Dtype>(k, j);
				Dtype pred_kj = cvPredictionCopy.at<Dtype>(k, j);
				if (mask_kj != pred_kj )
					mistakesCouter++;
			}
		
		}
		//printMask(cvMaskCopy);
		//printMask(cvPredictionCopy);
		mistakesCouter /= 50176.0f;
		//std::cout<<"mistakesCouter= "<<mistakesCouter<<"\n";
		maskAccuracy += (1.0f - mistakesCouter);
	}

	top[0]->mutable_cpu_data()[0] = maskAccuracy/rellevantMasks;
}

#ifdef CPU_ONLY
STUB_GPU(MaskAccuracyLayer);
#endif

INSTANTIATE_CLASS(MaskAccuracyLayer);
REGISTER_LAYER_CLASS(MaskAccuracy);

}  // namespace caffe
