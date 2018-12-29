#pragma once

#include <c10/util/Exception.h>
#include "c10/util/Registry.h"

namespace at {

class Context;

struct CAFFE2_API BFloat16HooksInterface {
  virtual ~BFloat16HooksInterface() {}

  virtual void registerBFloat16Type(Context*) const {
    AT_ERROR("Cannot register bfloat16 type without loading a library with bfloat16 support");
  }
};

struct CAFFE2_API BFloat16HooksArgs {};
C10_DECLARE_REGISTRY(
    BFloat16HooksRegistry,
    BFloat16HooksInterface,
    BFloat16HooksArgs);
#define REGISTER_BFLOAT16_HOOKS(clsname) \
  C10_REGISTER_CLASS(BFloat16HooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const BFloat16HooksInterface& getBFloat16Hooks();
}

}
