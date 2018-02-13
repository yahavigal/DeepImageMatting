#include "caffe/layers/mask_connect_accuracy_layer.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
int find_largst_cc(const cv::Mat& cc_stat)
{
    int max_area = int(0);
    int max_area_ind = 0;
    for (int i=1; i<cc_stat.rows;i++)
    {
        auto area = cc_stat.ptr<int>(i)[cv::CC_STAT_AREA];
        if (area > max_area)
        {
            max_area = area;
            max_area_ind = i;
        }
    }
    return max_area_ind;
}
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
        cv::Mat mask_thresh_int = mask >= 0.9;
        cv::Mat cc_top,dt,cc_top_stats,centeroids;
        auto num_cc_top = cv::connectedComponentsWithStats(mask_thresh_int,cc_top,cc_top_stats,centeroids,4);
        int largest_cc_top = find_largst_cc(cc_top_stats);
        cv::distanceTransform(cc_top!=largest_cc_top, dt, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        auto norm_factor =  std::sqrt(mask.cols*mask.cols + mask.rows*mask.rows); 
        dt/=norm_factor;
        //cv::imshow("dt",dt);
        //cv::waitKey();
        cv::Mat max_penalty = cv::Mat::zeros(mask.rows, mask.cols,CV_32FC1);

        for (float current_thresh = 0.1; current_thresh<0.9;current_thresh+=quantization)
        {
            cv::Mat current_thresh_int = mask >= current_thresh;
            cv::Mat current_cc_res,current_cc_stats;
            auto current_num_cc = cv::connectedComponentsWithStats(current_thresh_int,current_cc_res,current_cc_stats,centeroids,4);
            if (current_num_cc == num_cc_top)
                continue;
            cv::Mat current_d = mask - (current_thresh + Dtype(0.5)*quantization);
            //cv::waitKey();
            int current_largest_cc = find_largst_cc(current_cc_stats);
            current_d.setTo(0,current_cc_res == current_largest_cc);
            current_d.setTo(0,current_cc_res == 0);
            current_d.setTo(0,current_d < theta);

            //cv::imshow("current d",current_d);
            cv::Mat mutipication;
            cv::multiply(current_d,dt,mutipication);
            //cv::Mat current_conncet =  mutipication;
            cv::max(mutipication,max_penalty,max_penalty);
        }
        
        auto res = cv::sum(max_penalty)[0]/
                   (cc_top_stats.ptr<int>(largest_cc_top)[cv::CC_STAT_AREA]);
        //std::cout<<res<<"\n";
        return res > 0.0 ? res : Dtype(0);
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
