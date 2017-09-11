#include <algorithm>
#include <vector>

#include "caffe/layers/mask_soft_plus_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void MaskSoftPlusLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    Blob<Dtype>* predictions = bottom[0];
    Blob<Dtype>* masks = bottom[1];

    m_predictionHeight = predictions->shape(2);	
    m_predictionWidth = predictions->shape(3);

    m_maskHeight = masks->shape(2);	
    m_maskWidth = masks->shape(3);
	
    m_numericalStableValue = Dtype(15);
    CHECK_EQ(bottom[1]->shape(1), 1);

}

template <typename Dtype>
void MaskSoftPlusLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom[1]->shape(2), m_maskHeight);
  CHECK_EQ(bottom[1]->shape(3),m_maskWidth);
	
	Blob<Dtype>* predictions = bottom[0];
	Blob<Dtype>* masks = bottom[1];
		
	int num = bottom[0]->shape(0);
	Dtype lossPerAllBatch = 0;


	for (int i = 0; i < num; i++)
	{
		int offsetPred = predictions->offset(i);
		int offsetMask = masks->offset(i);
		CHECK_EQ(offsetPred,m_predictionHeight*m_predictionWidth*i);
		CHECK_EQ(offsetMask,m_maskWidth*m_maskHeight*i);

		cv::Mat cvPrediction(m_predictionHeight, m_predictionWidth, CV_32FC1, (void*)(predictions->cpu_data()+offsetPred)),cvPredictionCopy,cvPredictionTemp;
		cvPrediction.copyTo(cvPredictionCopy);

		CHECK_EQ(cvPredictionCopy.rows,m_predictionHeight);
		CHECK_EQ(cvPredictionCopy.cols,m_predictionWidth);		


		cv::Mat cvMask(m_maskHeight,m_maskWidth, CV_32FC1, (void*)(masks->cpu_data()+offsetMask)), cvMaskCopy;
		
		cvMask.copyTo(cvMaskCopy);
		if (m_maskWidth!=m_predictionWidth ||m_maskHeight!=m_predictionHeight)  
			cv::resize(cvMaskCopy, cvMaskCopy, cv::Size(m_predictionHeight, m_predictionWidth), cv::INTER_NEAREST);
		
		cvMaskCopy.setTo(1,cvMaskCopy > 0);
		cvMaskCopy.setTo(-1, cvMaskCopy == 0);

		Dtype sumOfElementWiseLog = 0;
		for (int i=0;i<m_predictionHeight;i++)
    {
			for (int j=0;j<m_predictionWidth;j++)
			{
				Dtype mask_ij = cvMaskCopy.at<Dtype>(i,j);
				Dtype pred_ij = cvPredictionCopy.at<Dtype>(i,j);
				if(-mask_ij*pred_ij > m_numericalStableValue)         
		 			sumOfElementWiseLog += (-mask_ij*pred_ij);     
				else if (-mask_ij*pred_ij < -m_numericalStableValue)
 					sumOfElementWiseLog += 0.0f;
				else
					sumOfElementWiseLog += std::log(1 + std::exp(-mask_ij*pred_ij));

			}
    }

		sumOfElementWiseLog /= Dtype(m_predictionHeight*m_predictionWidth);
		lossPerAllBatch += sumOfElementWiseLog;		
	}

	top[0]->mutable_cpu_data()[0] = lossPerAllBatch;
}

template <typename Dtype>
void MaskSoftPlusLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	Blob<Dtype>* predictions = bottom[0];
	Blob<Dtype>* masks = bottom[1];
	

	CHECK_GT(predictions->shape(0), 0) << "Zero blobs.";
	CHECK_EQ(predictions->shape(1), 1) << "Icorrect size of channels.";
	

	CHECK_EQ(propagate_down[0],true);
	CHECK_EQ(propagate_down[1],false);
	
	int num = bottom[0]->shape(0);
	int dim = predictions->shape(1)*predictions->shape(2)*predictions->shape(3);
	CHECK_EQ(predictions->count(),dim*num);
	int stride = predictions->shape(2);
	

	for (int j = 0; j < num; j++)
	{
		cv::Mat cvMask(m_maskHeight,m_maskWidth, CV_32FC1, (void*)(masks->cpu_data() + masks->offset(j))), cvMaskCopy;

		cvMask.copyTo(cvMaskCopy);
		if (m_maskWidth!=m_predictionWidth ||m_maskHeight!=m_predictionHeight)  
			cv::resize(cvMaskCopy, cvMaskCopy, cv::Size(m_predictionHeight, m_predictionWidth), cv::INTER_NEAREST);

		cvMaskCopy.setTo(1,cvMaskCopy > 0);
		cvMaskCopy.setTo(-1, cvMaskCopy == 0);

		for (int height = 0; height < predictions->shape(2); height++)
		{
			for (int width = 0; width < predictions->shape(3); width++)
			{
				Dtype pred = predictions->data_at(j, 0, height, width);
				Dtype mask = cvMaskCopy.at<Dtype>(height, width);

				LOG_IF(FATAL,mask != 1.0f && mask != -1.0f)<<"Unexpected mask value "<< mask;

				int index = j*dim + height*stride + width;
				LOG_IF(FATAL,index>=predictions->count())<<"index :"<<index<<" j: "<<j<<" dim: "<<dim<<" height :"<<height<<" stride: "<<stride<<" width :"<<width<<" count: "<<predictions->count();

				if (-mask*pred >m_numericalStableValue) 
					predictions->mutable_cpu_diff()[index] = (1.0f/Dtype(m_predictionHeight*m_predictionWidth))*(-mask);
				else if (-mask*pred <-m_numericalStableValue) 
					predictions->mutable_cpu_diff()[index] = 0.0f;
				else 
					predictions->mutable_cpu_diff()[index] = (1.0f/Dtype(m_predictionHeight*m_predictionWidth))*(-mask)*(std::exp(-mask*pred)*(1.0f / (1 + std::exp(-mask*pred))));
				
			}
		}
	}
			
	
}

#ifdef CPU_ONLY
STUB_GPU(MaskSoftPlusLossLayer);
#endif

INSTANTIATE_CLASS(MaskSoftPlusLossLayer);
REGISTER_LAYER_CLASS(MaskSoftPlusLoss);

}  // namespace caffe
