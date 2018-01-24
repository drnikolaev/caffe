#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Ftype, typename Btype>
class ImageDataLayer : public BasePrefetchingDataLayer<Ftype, Btype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Ftype, Btype>(param) {}
  virtual ~ImageDataLayer();
  void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  const char* type() const override { return "ImageData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }

 protected:
//  void ShuffleImages();
  void load_batch(Batch* batch, int thread_id, size_t queue_id = 0UL) override;
  void start_reading() override {}
  void InitializePrefetch() override;
  bool auto_mode() const override {
    return false;
  }

  //size_t queue_id(size_t thread_id) const override;

  Flag* layer_inititialized_flag() override {
    return this->phase_ == TRAIN ? &layer_inititialized_flag_ : nullptr;
  }

  shared_ptr<Caffe::RNG> prefetch_rng_;
//  size_t lines_id_;
  Flag layer_inititialized_flag_;
  vector<size_t> line_ids_;
  vector<std::pair<std::string, int>> lines_;
};

//template <typename Ftype, typename Btype>
//vector<std::pair<std::string, int>> ImageDataLayer<Ftype, Btype>::lines_;

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
