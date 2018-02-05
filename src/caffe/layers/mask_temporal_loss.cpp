#include <algorithm>
#include <vector>

#include "caffe/layers/mask_temporal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {
template <typename Dtype>
MaskTemporalLossLayer<Dtype>::MaskTemporalLossLayer(const LayerParameter& param):LossLayer<Dtype>(param) 
{
}

template <typename Dtype>
void MaskTemporalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
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

}

template <typename Dtype>
void MaskTemporalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
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
    Dtype norm_factor = m_predictionHeight*m_predictionWidth;

    for (int i = 1; i < num; i++)
    {
        Dtype sumOfElementWiseLoss = 0;
        for (int height = 0; height < predictions->shape(2); height++)
        {
            for (int width = 0; width < predictions->shape(3); width++)
            {
                
                Dtype pred = predictions->data_at(i, 0, height, width);
                Dtype prev_pred = predictions->data_at(i-1, 0, height, width);
                Dtype mask = masks->data_at(i, 0, height, width);
                Dtype prev_mask = masks->data_at(i-1, 0, height, width);
                
                Dtype diff_pred = pred - prev_pred;
                Dtype diff_mask = mask - prev_mask;
                
                Dtype square = (diff_pred - diff_mask)*(diff_pred - diff_mask);
                sumOfElementWiseLoss += square;

            }
        }

        sumOfElementWiseLoss /= norm_factor;
        lossPerAllBatch += sumOfElementWiseLoss;		
    }

    top[0]->mutable_cpu_data()[0] = lossPerAllBatch/num;
}

    template <typename Dtype>
void MaskTemporalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
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
    Dtype norm_factor = m_predictionHeight*m_predictionWidth;
    caffe_set(dim, Dtype(0), predictions->mutable_cpu_diff());

    for (int i = 1; i < num; i++)
    {

        for (int height = 0; height < predictions->shape(2); height++)
        {
            for (int width = 0; width < predictions->shape(3); width++)
            {
                Dtype pred = predictions->data_at(i, 0, height, width);
                Dtype prev_pred = predictions->data_at(i-1, 0, height, width);
                Dtype mask = masks->data_at(i, 0, height, width);
                Dtype prev_mask = masks->data_at(i-1, 0, height, width);
                
                Dtype diff_pred = pred - prev_pred;
                Dtype diff_mask = mask - prev_mask;

                int index = i*dim + height*stride + width;
                LOG_IF(FATAL,index>=predictions->count())<<"index :"<<index<<" i: "<<i<<" dim: "<<dim<<" height :"<<height<<" stride: "<<stride<<" width :"<<width<<" count: "<<predictions->count();

                Dtype grad = Dtype(2)*(diff_pred - diff_mask);
                predictions->mutable_cpu_diff()[index] = grad/norm_factor;
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(MaskTemporalLossLayer);
#else
NO_GPU_CODE(MaskTemporalLossLayer);
#endif

INSTANTIATE_CLASS(MaskTemporalLossLayer);
REGISTER_LAYER_CLASS(MaskTemporalLoss);

}  // namespace caffe
