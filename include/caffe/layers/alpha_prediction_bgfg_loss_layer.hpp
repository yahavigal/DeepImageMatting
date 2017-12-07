#ifndef CAFFE_ALPH_PREDICTION_LOSS_LAYER_HPP_
#define CAFFE_ALPH_PREDICTION_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class AlphaPredictionBGFGLossLayer : public LossLayer<Dtype> {
public:
   explicit AlphaPredictionBGFGLossLayer(const LayerParameter& param);
      
   virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

   virtual inline int ExactNumBottomBlobs() const { return 3; }
   virtual inline const char* type() const { return "AlphaPredictionBGFGLoss"; }
 
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int m_predictionHeight;	
    int m_predictionWidth;

    int m_maskHeight;	
    int m_maskWidth;

    Dtype m_epsilonSquare;
    bool m_use_trimap;
    int m_ignore_label;

};

}  // namespace caffe

#endif  // CAFFE_ALPH_PREDICTION_LOSS_LAYER_HPP_
