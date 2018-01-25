#include "caffe/layers/mask_connect_accuracy_layer.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe 
{
    template <typename Dtype>
        MaskConnectAccuracyLayer<Dtype>::MaskConnectAccuracyLayer(const LayerParameter& param):Layer<Dtype>(param) 
    {
    }

    template <typename Dtype>
        Dtype MaskConnectAccuracyLayer<Dtype>::calc_connectivity(const cv::Mat& mask)
    {
        float theta = 0.15;
        float quantization = 0.1;
        cv::Mat mask_thresh = mask >= 0.9;
        cv::Mat mask_thresh_int = mask_thresh*255;
        mask_thresh_int.convertTo(mask_thresh_int,CV_8UC1);
        cv::Mat cc_top,dt;
        auto num_cc_top = cv::connectedComponents(mask_thresh_int,cc_top);
        cv::distanceTransform(cc_top!=1, dt, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        double min,max;
        cv::minMaxLoc(dt,&min,&max);
        dt/=max;
        //cv::imshow("dt",dt);
        //cv::waitKey();
        cv::Mat max_penalty = cv::Mat::zeros(mask.rows, mask.cols,CV_32FC1);

        for (float current_thresh = 0.1; current_thresh<0.9;current_thresh+=quantization)
        {
            cv::Mat current_thresh_mask = mask >= current_thresh;
            cv::Mat current_thresh_int = mask_thresh*255;
            current_thresh_int.convertTo(current_thresh_int,CV_8UC1);
            cv::Mat current_cc_res;
            auto current_num_cc = cv::connectedComponents(current_thresh_int,current_cc_res);
            if (current_num_cc == num_cc_top)
                continue;
            cv::Mat current_d = mask - (current_thresh - quantization);
            //cv::waitKey();
            current_d.setTo(0,current_cc_res == 1);
            current_d.setTo(0,current_cc_res == 0);
            current_d.setTo(0,current_cc_res < theta);

            //cv::imshow("current d",current_d);
            cv::Mat mutipication;
            cv::multiply(current_d,dt,mutipication);
            cv::Mat current_conncet = 1 - mutipication;
            cv::max(current_conncet,max_penalty,max_penalty);
        }
        
        auto res = cv::mean(cv::abs(max_penalty))[0];
        std::cout<<res<<"\n";
        return res > 0.0 ? res : Dtype(1);
    }
    
    template <typename Dtype>
        void MaskConnectAccuracyLayer<Dtype>::LayerSetUp(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
        {
        }

    template <typename Dtype>
        void MaskConnectAccuracyLayer<Dtype>::Reshape(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
        {
            vector<int> top_shape(0);
            top[0]->Reshape(top_shape);
        }

    template <typename Dtype>
        void MaskConnectAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) 
        {
            CHECK_EQ(bottom[0]->shape(1), 1)<<"prediction must has one channel";
            CHECK_EQ(bottom[1]->shape(1), 1)<<"mask must has one channel";

            CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))<<"mask and prediction must have the same height";
            CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))<<"mask and prediction must have the same width";

            Blob<Dtype>* predictions = bottom[0];
            Blob<Dtype>* masks = bottom[1];

            int num = bottom[0]->shape(0);
            Dtype batch_connect = 0;

            for (int i = 0; i < num; i++)
            {
                int offsetPred = predictions->offset(i);
                int offsetMask = masks->offset(i);

                cv::Mat cvPrediction(predictions->shape(2), predictions->shape(3), CV_32FC1, (void*)(predictions->cpu_data() + offsetPred)), cvPredictionCopy, cvPredictionTemp;
                cvPrediction.copyTo(cvPredictionCopy);

                cv::Mat cvMask(masks->shape(2), masks->shape(3), CV_32FC1, (void*)(masks->cpu_data() + offsetMask)), cvMaskCopy;
                cvMask.copyTo(cvMaskCopy);

                batch_connect += std::abs(calc_connectivity(cvMaskCopy)- calc_connectivity(cvPredictionCopy));
                
            }

            top[0]->mutable_cpu_data()[0] = batch_connect/num;
        }

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(MaskConnectAccuracyLayer,Forward);
#else
    NO_GPU_CODE_FORWARD(MaskConnectAccuracyLayer);
#endif

    INSTANTIATE_CLASS(MaskConnectAccuracyLayer);
    REGISTER_LAYER_CLASS(MaskConnectAccuracy);

}  // namespace caffe
