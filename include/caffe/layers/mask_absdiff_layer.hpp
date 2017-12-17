#ifndef CAFFE_MaskAbsDiff_LAYER_HPP_
#define CAFFE_MaskAbsDiff_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
    class MaskAbsDiffLayer : public Layer<Dtype>
{
public:
        explicit MaskAbsDiffLayer(const LayerParameter& param);

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "MaskAbsDiff"; }
        virtual inline int MinNumBottomBlobs() const { return 2; }

        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlos() const { return 1; }

protected:

                virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

                virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

                virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
                    for (int i = 0; i < propagate_down.size(); ++i) {
                        if (propagate_down[i]) { NOT_IMPLEMENTED; }
                    }
                }

                virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
                    for (int i = 0; i < propagate_down.size(); ++i) {
                        if (propagate_down[i]) { NOT_IMPLEMENTED; }
                    }
                }
                Dtype m_thresh;
                bool m_use_trimap;
                int m_ignore_label;
        };

}  // namespace caffe

#endif  // CAFFE_MaskAbsDiff_LAYER_HPP_
#pragma once
