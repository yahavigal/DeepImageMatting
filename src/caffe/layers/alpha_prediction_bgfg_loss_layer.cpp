#include <algorithm>
#include <vector>

#include "caffe/layers/alpha_prediction_bgfg_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {
template <typename Dtype>
AlphaPredictionBGFGLossLayer<Dtype>::AlphaPredictionBGFGLossLayer(const LayerParameter& param):LossLayer<Dtype>(param) 
{
   m_epsilonSquare = (1e-6)*(1e-6);
   if (param.has_loss_param() == true && param.loss_param().has_ignore_label() == true)
       m_ignore_label = param.loss_param().ignore_label();
   else
       m_ignore_label = int(0);
}



template <typename Dtype>
void AlphaPredictionBGFGLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    Blob<Dtype>* predictions = bottom[0];
    Blob<Dtype>* masks = bottom[2];

    m_predictionHeight = predictions->shape(2);	
    m_predictionWidth = predictions->shape(3);

    m_maskHeight = masks->shape(2);	
    m_maskWidth = masks->shape(3);

    CHECK_EQ(m_maskHeight, m_predictionHeight) <<"prediction and mask must have the same height";
    CHECK_EQ(m_maskWidth, m_predictionWidth) <<"prediction and mask must have the same widht";
    CHECK_EQ(bottom[2]->shape(2),bottom[1]->shape(2)) <<"bkg and mask must have the same height";
    CHECK_EQ(bottom[2]->shape(3),bottom[1]->shape(3)) <<"bkg and mask must have the same widht";
    CHECK_EQ(bottom[1]->shape(1), 1);
    CHECK_EQ(bottom[0]->shape(1), 1);
    CHECK_EQ(bottom[2]->shape(1), 1);

   
    m_use_trimap = bottom.size() == 4;
    

}

template <typename Dtype>
void AlphaPredictionBGFGLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    Blob<Dtype>* predictions_fg = bottom[0];
    Blob<Dtype>* masks = bottom[2];
    Blob<Dtype>* trimaps = NULL;
    if (m_use_trimap == true)
        trimaps = bottom[3];

    int num = bottom[0]->shape(0);
    Dtype lossPerAllBatch = 0;
    Dtype norm_factor = m_predictionHeight*m_predictionWidth;

    for (int i = 0; i < num; i++)
    {
        int offsetPred = predictions_fg->offset(i);
        int offsetMask = masks->offset(i);
        CHECK_EQ(offsetPred,m_predictionHeight*m_predictionWidth*i);
        CHECK_EQ(offsetMask,m_maskWidth*m_maskHeight*i);

        cv::Mat cvPrediction(m_predictionHeight, m_predictionWidth, CV_32FC1, (void*)(predictions_fg->cpu_data()+offsetPred)),cvPredictionCopy,cvPredictionTemp;
        cvPrediction.copyTo(cvPredictionCopy);

        CHECK_EQ(cvPredictionCopy.rows,m_predictionHeight);
        CHECK_EQ(cvPredictionCopy.cols,m_predictionWidth);		

        cv::Mat cvMask(m_maskHeight,m_maskWidth, CV_32FC1, (void*)(masks->cpu_data()+offsetMask)), cvMaskCopy;
        cvMask.copyTo(cvMaskCopy);
        if (m_maskWidth!=m_predictionWidth ||m_maskHeight!=m_predictionHeight)  
            cv::resize(cvMaskCopy, cvMaskCopy, cv::Size(m_predictionHeight, m_predictionWidth), cv::INTER_NEAREST);
                        
        cvMaskCopy/=Dtype(255);
        if (m_use_trimap == true)
            norm_factor = 0;

     
        Dtype sumOfElementWiseLoss = 0;
        for (int k=0;k<m_predictionHeight;k++)
        {
            for (int j=0;j<m_predictionWidth;j++)
            {
                Dtype mask_kj = cvMaskCopy.at<Dtype>(k,j);
                Dtype pred_kj = cvPredictionCopy.at<Dtype>(k,j);
                Dtype square = (pred_kj - mask_kj)*(pred_kj - mask_kj);
                //std::cout<<"square: "<<square<<'\n';
                if (m_use_trimap == true)
                {
                  Dtype trimap_val = trimaps->data_at(i,3,k,j);
                  if (trimap_val != m_ignore_label)
                    continue;
                  else
                    norm_factor++;
                }  
                sumOfElementWiseLoss += std::sqrt(square + m_epsilonSquare);
        
            }
        }
    
        sumOfElementWiseLoss /= norm_factor;
        lossPerAllBatch += sumOfElementWiseLoss;		
    }

    top[0]->mutable_cpu_data()[0] = lossPerAllBatch;
}

template <typename Dtype>
void AlphaPredictionBGFGLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    Blob<Dtype>* predictions_fg = bottom[0];
    Blob<Dtype>* predictions_bg = bottom[1];
    Blob<Dtype>* masks = bottom[2];
    Blob<Dtype>* trimaps = NULL;
    if (m_use_trimap == true)
        trimaps = bottom[3];

    CHECK_GT(predictions_fg->shape(0), 0) << "Zero blobs.";
    CHECK_GT(predictions_bg->shape(0), 0) << "Zero blobs.";
    CHECK_EQ(predictions_fg->shape(1), 1) << "Icorrect size of channels.";
    CHECK_EQ(predictions_bg->shape(1), 1) << "Icorrect size of channels.";
    
    CHECK_EQ(propagate_down[0],true);
    CHECK_EQ(propagate_down[1],true);
    CHECK_EQ(propagate_down[2],false);
    
    int num = bottom[0]->shape(0);
    int dim = predictions_fg->shape(1)*predictions_fg->shape(2)*predictions_fg->shape(3);
    CHECK_EQ(predictions_fg->count(),dim*num);
    int stride = predictions_fg->shape(2);
    Dtype norm_factor = m_predictionHeight*m_predictionWidth;

    for (int j = 0; j < num; j++)
    {
        cv::Mat cvMask(m_maskHeight,m_maskWidth, CV_32FC1, (void*)(masks->cpu_data() + masks->offset(j))), cvMaskCopy;

        cvMask.copyTo(cvMaskCopy);
        if (m_maskWidth!=m_predictionWidth ||m_maskHeight!=m_predictionHeight)  
            cv::resize(cvMaskCopy, cvMaskCopy, cv::Size(m_predictionHeight, m_predictionWidth), cv::INTER_NEAREST);

        cvMaskCopy/=Dtype(255);

        for (int height = 0; height < predictions_fg->shape(2); height++)
        {
            for (int width = 0; width < predictions_fg->shape(3); width++)
            {
                Dtype pred_fg = predictions_fg->data_at(j, 0, height, width);
                Dtype pred_bg = predictions_bg->data_at(j, 0, height, width);
                Dtype mask = cvMaskCopy.at<Dtype>(height, width);

                int index = j*dim + height*stride + width;
                LOG_IF(FATAL,index>=predictions_fg->count())<<"index :"<<index<<" j: "<<j<<" dim: "<<dim<<" height :"<<height<<" stride: "<<stride<<" width :"<<width<<" count: "<<predictions_fg->count();
                LOG_IF(FATAL,index>=predictions_bg->count())<<"index :"<<index<<" j: "<<j<<" dim: "<<dim<<" height :"<<height<<" stride: "<<stride<<" width :"<<width<<" count: "<<predictions_bg->count();

                Dtype square_fg = (pred_fg - mask)*(pred_fg - mask);
                Dtype grad_fg =  (pred_fg - mask)/std::sqrt(square_fg + m_epsilonSquare);
                predictions_fg->mutable_cpu_diff()[index] = grad_fg/norm_factor;
                

                Dtype square_bg = (pred_bg -(1 - mask))*(pred_bg -(1- mask));
                Dtype grad_bg =  (pred_bg - (1 - mask))/std::sqrt(square_bg + m_epsilonSquare);
                predictions_bg->mutable_cpu_diff()[index] = grad_bg/norm_factor;
                
                if (m_use_trimap == true)
                {
                     Dtype trimap_val = trimaps->data_at(j, 3, height, width);
                     if (trimap_val != m_ignore_label)
                     {
                         predictions_fg->mutable_cpu_diff()[index] = Dtype(0);
                         predictions_bg->mutable_cpu_diff()[index] = Dtype(0);
                     }
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(AlphaPredictionBGFGLossLayer);
#else
NO_GPU_CODE(AlphaPredictionBGFGLossLayer);
#endif

INSTANTIATE_CLASS(AlphaPredictionBGFGLossLayer);
REGISTER_LAYER_CLASS(AlphaPredictionBGFGLoss);

}  // namespace caffe
