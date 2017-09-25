#include "caffe/layers/mask_iou_layer.hpp"
#include <opencv2/core/core.hpp>

namespace caffe 
{
template <typename Dtype>
MaskIOULayer<Dtype>::MaskIOULayer(const LayerParameter& param):Layer<Dtype>(param) 
{
  m_thresh = param.threshold_param().threshold();
  if (param.has_loss_param() == true && param.loss_param().has_ignore_label() == true)
    m_ignore_label = param.loss_param().ignore_label();
  else
    m_ignore_label = int(0);
}
	template <typename Dtype>
	void MaskIOULayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    m_use_trimap = bottom.size() == 3;
}


template <typename Dtype>
void MaskIOULayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  vector<int> top_shape(0);
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MaskIOULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	CHECK_EQ(bottom[0]->shape(1), 1)<<"prediction must has one channel";
	CHECK_EQ(bottom[1]->shape(1), 1)<<"mask must has one channel";

  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))<<"mask and prediction must have the same height";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))<<"mask and prediction must have the same width";

	Blob<Dtype>* predictions = bottom[0];
	Blob<Dtype>* masks = bottom[1];
  Blob<Dtype>* trimaps = NULL;
  if (m_use_trimap == true)
    trimaps = bottom[2];

	int num = bottom[0]->shape(0);
	Dtype batchIOU = 0;

	for (int i = 0; i < num; i++)
	{
   
		int offsetPred = predictions->offset(i);
		int offsetMask = masks->offset(i);
		
		cv::Mat cvPrediction(predictions->shape(2), predictions->shape(3), CV_32FC1, (void*)(predictions->cpu_data() + offsetPred)), cvPredictionCopy, cvPredictionTemp;
		cvPrediction.copyTo(cvPredictionCopy);

		cv::Mat cvMask(masks->shape(2), masks->shape(3), CV_32FC1, (void*)(masks->cpu_data() + offsetMask)), cvMaskCopy;
		cvMask.copyTo(cvMaskCopy);

		Dtype intersectionPixels = 0;
    Dtype unionPixels = 0;

		for (int k = 0; k<predictions->shape(2); k++)
		{
			for (int j = 0; j<predictions->shape(3); j++)
			{
				Dtype mask_kj = cvMaskCopy.at<Dtype>(k, j);
				Dtype pred_kj = cvPredictionCopy.at<Dtype>(k, j);
        if (m_use_trimap == true)
        {
          Dtype trimap_val = trimaps->data_at(i,3,k,j);
          if (trimap_val != m_ignore_label)
            continue;
        }
				if (mask_kj > m_thresh || pred_kj > m_thresh)
          unionPixels++;
        if (mask_kj > m_thresh && pred_kj > m_thresh)
          intersectionPixels++;
			}
		
		}
    
    Dtype iou = intersectionPixels/unionPixels;
		//std::cout<<"iou= "<<iou<<"\n";
		batchIOU += iou;
	}

	top[0]->mutable_cpu_data()[0] = batchIOU/num;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(MaskIOULayer,Forward);
#endif

INSTANTIATE_CLASS(MaskIOULayer);
REGISTER_LAYER_CLASS(MaskIOU);

}  // namespace caffe
