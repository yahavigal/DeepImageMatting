#include <algorithm>
#include <vector>

#include "caffe/layers/mask_focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
MaskFocalLossLayer<Dtype>::MaskFocalLossLayer(const LayerParameter& param):LossLayer<Dtype>(param)
{
    if (param.has_loss_param())
    {
        m_is_segmentation = param.loss_param().is_segmentation();
    }
    else 
    {
        m_is_segmentation = false;
    }
}
template <typename Dtype>
void MaskFocalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    Blob<Dtype>* predictions = bottom[0];
    Blob<Dtype>* masks = bottom[1];

    m_predictionHeight = predictions->shape(2);	
    m_predictionWidth = predictions->shape(3);

    m_maskHeight = masks->shape(2);	
    m_maskWidth = masks->shape(3);

	  CHECK_EQ(m_maskHeight, m_predictionHeight) <<"prediction and mask must have the same height";
    CHECK_EQ(m_maskWidth, m_predictionWidth) <<"prediction and mask must have the same widht";
    CHECK_EQ(bottom[1]->shape(1), 1);

    m_numerical_stable_value = 1e-7;
    m_low_loss = -16.1180957;
    m_epsilon = 1e-10;
    m_gamma = Dtype(2);
    m_alpha =  Dtype(0.25);
}

template <typename Dtype>
void MaskFocalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
	

  CHECK_EQ(bottom[1]->shape(2), m_maskHeight);
  CHECK_EQ(bottom[1]->shape(3),m_maskWidth);
  CHECK_EQ(bottom[0]->shape(2), m_maskHeight);
  CHECK_EQ(bottom[0]->shape(3),m_maskWidth);
	
  CHECK_EQ(bottom[0]->shape(1), 1)<<"prediction must have 1 channel";
  CHECK_EQ(bottom[1]->shape(1),1)<<"mask must have 1 channel";

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

        cvMaskCopy/=Dtype(255);
        if(m_is_segmentation)
        {
            cvMaskCopy.setTo(0,cvMaskCopy<Dtype(0.5));
            cvMaskCopy.setTo(1,cvMaskCopy>=Dtype(0.5));
        }

        Dtype sumOfElementWiseLoss = 0;
        for (int i=0;i<m_predictionHeight;i++)
        {
            for (int j=0;j<m_predictionWidth;j++)
            {
                Dtype mask_ij = cvMaskCopy.at<Dtype>(i,j);
                Dtype pred_ij = cvPredictionCopy.at<Dtype>(i,j);
                if(m_is_segmentation)
                {
                    LOG_IF(FATAL, mask_ij != 1 && mask_ij != 0)<<"Inavlid mask "<< mask_ij;
                }

                LOG_IF(FATAL, pred_ij < 0 || pred_ij > 1)<<"Invalid pred "<< pred_ij;
                Dtype attenuation_factor = m_alpha*std::pow(pred_ij - mask_ij,m_gamma);
                Dtype loss = 0;
                if (pred_ij < 1e-7)
                    loss = mask_ij*m_low_loss + (Dtype(1) - mask_ij)*std::log(Dtype(1) - pred_ij);
                else if(Dtype(1) - pred_ij < 1e-7)
                    loss = mask_ij*std::log(pred_ij) + (Dtype(1) - mask_ij)*m_low_loss;
                else
                    loss = mask_ij*std::log(pred_ij) + (Dtype(1) - mask_ij)*std::log(Dtype(1) - pred_ij);
                loss = -loss*attenuation_factor; 
                sumOfElementWiseLoss += loss;

            }
        }

        sumOfElementWiseLoss /= Dtype(m_predictionHeight*m_predictionWidth);
        lossPerAllBatch += sumOfElementWiseLoss;		
        }

        top[0]->mutable_cpu_data()[0] = lossPerAllBatch;
}

template <typename Dtype>
void MaskFocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
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

        cvMaskCopy/=Dtype(255);
        if(m_is_segmentation)
        {
            cvMaskCopy.setTo(0,cvMaskCopy<Dtype(0.5));
            cvMaskCopy.setTo(1,cvMaskCopy>=Dtype(0.5));
        }

        for (int height = 0; height < predictions->shape(2); height++)
        {
            for (int width = 0; width < predictions->shape(3); width++)
            {
                Dtype pred = predictions->data_at(j, 0, height, width);
                Dtype mask = cvMaskCopy.at<Dtype>(height, width);
                // since focal loss is product of two functions need to do
                // f'(x)*g(x) + g'(x)f(x)
                // when f(x) = alpha*(pred-gt)^gamma
                // g(x) = gt*log(p) + (1-gt)*log(1-p)
       
                int index = j*dim + height*stride + width;
                LOG_IF(FATAL,index>=predictions->count())<<"index :"<<index<<" j: "<<j<<" dim: "<<dim<<" height :"<<height<<" stride: "<<stride<<" width :"<<width<<" count: "<<predictions->count();
       
                Dtype grad_focal  = m_alpha*m_gamma*std::pow(pred - mask,m_gamma -Dtype(1));
                Dtype grad_CE =  (-mask)/(pred + m_epsilon) + (Dtype(1) - mask)/(Dtype(1) - pred + m_epsilon);
                Dtype func_focal = m_alpha*std::pow(pred - mask,m_gamma);
                
                Dtype func_CE = Dtype(0);
                if (pred < 1e-7)
                    func_CE = mask*m_low_loss + (Dtype(1) - mask)*std::log(Dtype(1) - pred);
                else if(Dtype(1) - pred < 1e-7)
                    func_CE = mask*std::log(pred) + (Dtype(1) - mask)*m_low_loss;
                else
                    func_CE = mask*std::log(pred) + (Dtype(1) - mask)*std::log(Dtype(1) - pred);
                func_CE = -func_CE;

                
                Dtype grad = grad_focal*func_CE + grad_CE*func_focal;
                //if (std::signbit(grad_CE) != std::signbit(grad))
                //{
                //    std::cout<<"mask: "<<mask<<" pred: "<<pred;
                //    std::cout<<" grad_focal: "<<grad_focal<<" func_CE: "<<func_CE<<" grad_CE: "<<grad_CE<<" func_focal: "<<func_focal;
                //    std::cout<<" grad: "<<grad<<std::endl;
                //}

                predictions->mutable_cpu_diff()[index] = (Dtype(1)/Dtype(m_predictionHeight*m_predictionWidth))*grad;
            }
        }
	}
}

#ifdef CPU_ONLY
STUB_GPU(MaskFocalLossLayer);
#else
NO_GPU_CODE(MaskFocalLossLayer);
#endif

INSTANTIATE_CLASS(MaskFocalLossLayer);
REGISTER_LAYER_CLASS(MaskFocalLoss);

}  // namespace caffe
