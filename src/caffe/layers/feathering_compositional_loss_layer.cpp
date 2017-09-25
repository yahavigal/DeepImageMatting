#include <algorithm>
#include <vector>

#include "caffe/layers/feathering_compositional_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace caffe {

template <typename Dtype>
void FeatheringCompositionalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    Blob<Dtype>* predictions = bottom[0];
    Blob<Dtype>* masks = bottom[1];

    m_predictionHeight = predictions->shape(2);	
    m_predictionWidth = predictions->shape(3);

    m_maskHeight = masks->shape(2);	
    m_maskWidth = masks->shape(3);
	  
    CHECK_EQ(m_predictionHeight, m_maskHeight);
    CHECK_EQ(m_predictionWidth, m_maskWidth);    

    CHECK_EQ(bottom[1]->shape(1), 1);
    
    m_epsilonSquare = 1e-6*1e-6;

}

template <typename Dtype>
void FeatheringCompositionalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom[1]->shape(2), m_maskHeight);
  CHECK_EQ(bottom[1]->shape(3),m_maskWidth);
	
	Blob<Dtype>* predictions = bottom[0];
	Blob<Dtype>* masks = bottom[1];
	Blob<Dtype>* images = bottom[2];
		
	int num = bottom[0]->shape(0);
	Dtype lossPerAllBatch = 0;
  
  for (int imageInd = 0 ;imageInd < num; imageInd++)
  {
    Dtype lossPerImage = 0;
    for (int heightInd = 0 ;heightInd < m_predictionHeight; heightInd++)
    {
      for (int widthInd = 0 ;widthInd < m_predictionWidth; widthInd++)
      {
        Dtype imageR = images->data_at(imageInd, 0, heightInd, widthInd);
        Dtype imageG = images->data_at(imageInd, 1, heightInd, widthInd);
        Dtype imageB = images->data_at(imageInd, 2, heightInd, widthInd);

        Dtype predictedAlpha = predictions->data_at(imageInd, 0, heightInd, widthInd);
        Dtype gtAlpha = masks->data_at(imageInd, 0, heightInd, widthInd); 

        Dtype squareR = (imageR*predictedAlpha - imageR*gtAlpha)*(imageR*predictedAlpha - imageR*gtAlpha);
        Dtype squareG = (imageG*predictedAlpha - imageG*gtAlpha)*(imageG*predictedAlpha - imageG*gtAlpha);
        Dtype squareB = (imageB*predictedAlpha - imageB*gtAlpha)*(imageB*predictedAlpha - imageB*gtAlpha);

        Dtype lossPixel = std::sqrt(squareR + m_epsilonSquare);
        lossPixel += std::sqrt(squareG + m_epsilonSquare);
        lossPixel += std::sqrt(squareB + m_epsilonSquare);
        lossPerImage += lossPixel;
      } 
    }
    lossPerAllBatch += (Dtype(1)/(m_predictionHeight*m_predictionWidth))*lossPerImage;
  }

	top[0]->mutable_cpu_data()[0] = lossPerAllBatch;
}

template <typename Dtype>
void FeatheringCompositionalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	Blob<Dtype>* predictions = bottom[0];
	Blob<Dtype>* masks = bottom[1];
	Blob<Dtype>* images = bottom[2];

	CHECK_GT(predictions->shape(0), 0) << "Zero blobs.";
	CHECK_EQ(predictions->shape(1), 1) << "Icorrect size of channels.";
	int num = predictions->shape(0);

	CHECK_EQ(propagate_down[0], true);
	CHECK_EQ(propagate_down[1], false);
	CHECK_EQ(propagate_down[2], false);

  int dim = predictions->shape(1)*predictions->shape(2)*predictions->shape(3);
	CHECK_EQ(predictions->count(),dim*num);
	int stride = predictions->shape(2);
	
	for (int imageInd = 0 ;imageInd < num; imageInd++)
  {
    for (int heightInd = 0 ;heightInd < m_predictionHeight; heightInd++)
    {
      for (int widthInd = 0 ;widthInd < m_predictionWidth; widthInd++)
      {
        Dtype imageR = images->data_at(imageInd, 0, heightInd, widthInd);
        Dtype imageG = images->data_at(imageInd, 1, heightInd, widthInd);
        Dtype imageB = images->data_at(imageInd, 2, heightInd, widthInd);

        Dtype predictedAlpha = predictions->data_at(imageInd, 0, heightInd, widthInd);
        Dtype gtAlpha = masks->data_at(imageInd, 0, heightInd, widthInd);

        Dtype squareR = (imageR*predictedAlpha - imageR*gtAlpha)*(imageR*predictedAlpha - imageR*gtAlpha);
        Dtype squareG = (imageG*predictedAlpha - imageG*gtAlpha)*(imageG*predictedAlpha - imageG*gtAlpha);
        Dtype squareB = (imageB*predictedAlpha - imageB*gtAlpha)*(imageB*predictedAlpha - imageB*gtAlpha);

        Dtype diffR = imageR*(imageR*predictedAlpha - imageR*gtAlpha)/std::sqrt(squareR + m_epsilonSquare);
        Dtype diffG = imageG*(imageG*predictedAlpha - imageG*gtAlpha)/std::sqrt(squareG + m_epsilonSquare);
        Dtype diffB = imageB*(imageB*predictedAlpha - imageB*gtAlpha)/std::sqrt(squareB + m_epsilonSquare);
        
        int index = imageInd*dim + heightInd*stride + widthInd;
        predictions->mutable_cpu_diff()[index] = (diffR + diffG + diffB)/(m_predictionHeight*m_predictionWidth);
      } 
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FeatheringCompositionalLossLayer);
#else
NO_GPU_CODE(FeatheringCompositionalLossLayer);
#endif

INSTANTIATE_CLASS(FeatheringCompositionalLossLayer);
REGISTER_LAYER_CLASS(FeatheringCompositionalLoss);

}  // namespace caffe
