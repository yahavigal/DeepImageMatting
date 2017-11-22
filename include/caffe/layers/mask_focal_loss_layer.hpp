#ifndef CAFFE_MAKS_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_MASK_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class MaskFocalLossLayer : public LossLayer<Dtype> {
 public:
  explicit MaskFocalLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "MaskFocalLoss"; }
 
 protected:
  /// @copydoc ContrastiveLossLayer
	 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		 const vector<Blob<Dtype>*>& top);

	 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		 const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int m_predictionHeight;	
	int m_predictionWidth;

	int m_maskHeight;	
	int m_maskWidth;

   Dtype m_numerical_stable_value;
   Dtype m_low_loss;
   Dtype m_epsilon;
   Dtype m_gamma;
   Dtype m_alpha;
};

}  // namespace caffe

#endif  // CAFFE_ALPH_PREDICTION_LOSS_LAYER_HPP_
