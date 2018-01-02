#include "caffe/layers/mask_gradient_accuracy_layer.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe 
{
    template <typename Dtype>
        MaskGradientAccuracyLayer<Dtype>::MaskGradientAccuracyLayer(const LayerParameter& param):Layer<Dtype>(param) 
    {
    }
    template <typename Dtype>
        void MaskGradientAccuracyLayer<Dtype>::LayerSetUp(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
        {
        }

    template <typename Dtype>
        void MaskGradientAccuracyLayer<Dtype>::Reshape(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
        {
            vector<int> top_shape(0);
            top[0]->Reshape(top_shape);
        }

    template <typename Dtype>
        void MaskGradientAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) 
        {
            CHECK_EQ(bottom[0]->shape(1), 1)<<"prediction must has one channel";
            CHECK_EQ(bottom[1]->shape(1), 1)<<"mask must has one channel";

            CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))<<"mask and prediction must have the same height";
            CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))<<"mask and prediction must have the same width";

            Blob<Dtype>* predictions = bottom[0];
            Blob<Dtype>* masks = bottom[1];

            int num = bottom[0]->shape(0);
            Dtype batch_gradient = 0;
            Dtype epsilon = 1e-2;

            for (int i = 0; i < num; i++)
            {
                int offsetPred = predictions->offset(i);
                int offsetMask = masks->offset(i);

                cv::Mat cvPrediction(predictions->shape(2), predictions->shape(3), CV_32FC1, (void*)(predictions->cpu_data() + offsetPred)), cvPredictionCopy, cvPredictionTemp;
                cvPrediction.copyTo(cvPredictionCopy);

                cv::Mat cvMask(masks->shape(2), masks->shape(3), CV_32FC1, (void*)(masks->cpu_data() + offsetMask)), cvMaskCopy;
                cvMask.copyTo(cvMaskCopy);
                cv::Mat gt_gauss,pred_gauss,gt_grad_x,gt_grad_y,pred_grad_x,
                        pred_grad_y,gt_mag,pred_mag;
                
                cv::GaussianBlur(cvMaskCopy, gt_gauss, cv::Size(5,5), 1.4,1.4);
                cv::GaussianBlur(cvPredictionCopy, pred_gauss, cv::Size(5,5), 1.4,1.4);
                cv::Sobel(gt_gauss, gt_grad_x, -1, 1, 0);
                cv::Sobel(gt_gauss, gt_grad_y, -1, 0, 1);
                cv::Sobel(pred_gauss, pred_grad_x, -1, 1, 0);
                cv::Sobel(pred_gauss, pred_grad_y, -1, 0, 1);
                cv::magnitude(gt_grad_x, gt_grad_y, gt_mag);
                cv::magnitude(pred_grad_x, pred_grad_y, pred_mag);
                
                cv::divide(gt_grad_x,gt_mag,gt_grad_x);
                cv::divide(gt_grad_y,gt_mag,gt_grad_y);
                pred_grad_x.setTo(0, pred_grad_x < epsilon);
                pred_grad_y.setTo(0, pred_grad_y < epsilon);
                gt_grad_x.setTo(0, gt_grad_x < epsilon);
                gt_grad_y.setTo(0, gt_grad_y < epsilon);
                cv::divide(pred_grad_x, pred_mag, pred_grad_x);
                cv::divide(pred_grad_y, pred_mag, pred_grad_y);
                cv::Mat normalized_magnitude;
                cv::magnitude(gt_grad_x - pred_grad_x, gt_grad_y - pred_grad_y, normalized_magnitude);
                batch_gradient += cv::mean(normalized_magnitude)[0];
            }

            top[0]->mutable_cpu_data()[0] = batch_gradient/num;
        }

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(MaskGradientAccuracyLayer,Forward);
#else
    NO_GPU_CODE_FORWARD(MaskGradientAccuracyLayer);
#endif

    INSTANTIATE_CLASS(MaskGradientAccuracyLayer);
    REGISTER_LAYER_CLASS(MaskGradientAccuracy);

}  // namespace caffe
