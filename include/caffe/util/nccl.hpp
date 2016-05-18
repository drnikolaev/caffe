#ifndef CAFFE_UTIL_NCCL_H_
#define CAFFE_UTIL_NCCL_H_
#ifdef USE_NCCL

#include <nccl.h>

#include "caffe/common.hpp"

#define NCCL_CHECK(condition) \
  { \
    ncclResult_t result = condition; \
    CHECK_EQ(result, ncclSuccess) << " "\
      << ncclGetErrorString(result); \
  }

#if 0
inline const char* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return "ncclSuccess";
    case ncclUnhandledCudaError:
      return "ncclUnhandledCudaError";
    case ncclSystemError:
      return "ncclSystemError";
    case ncclInternalError:
      return "ncclInternalError";
    case ncclInvalidDevicePointer:
      return "ncclInvalidDevicePointer";
    case ncclInvalidRank:
      return "ncclInvalidRank";
    case ncclUnsupportedDeviceCount:
      return "ncclUnsupportedDeviceCount";
    case ncclDeviceNotFound:
      return "ncclDeviceNotFound";
    case ncclInvalidDeviceIndex:
      return "ncclInvalidDeviceIndex";
    case ncclLibWrapperNotSet:
      return "ncclLibWrapperNotSet";
    case ncclCudaMallocFailed:
      return "ncclCudaMallocFailed";
    case ncclRankMismatch:
      return "ncclRankMismatch";
    case ncclInvalidArgument:
      return "ncclInvalidArgument";
    case ncclInvalidType:
      return "ncclInvalidType";
    case ncclInvalidOperation:
      return "ncclInvalidOperation";
    case nccl_NUM_RESULTS:
      break;
  }
  return "Unknown NCCL status";
}
#endif

namespace caffe {

namespace nccl {

template <typename Dtype> class dataType;

template<> class dataType<float> {
 public:
  static const ncclDataType_t type = ncclFloat;
};
template<> class dataType<double> {
 public:
  static const ncclDataType_t type = ncclDouble;
};

}  // namespace nccl

}  // namespace caffe

#endif  // end USE_NCCL

#endif  // CAFFE_UTIL_NCCL_H_
