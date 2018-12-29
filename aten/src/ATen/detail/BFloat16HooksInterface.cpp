#include <ATen/detail/BFloat16HooksInterface.h>

namespace at {

namespace detail {
const BFloat16HooksInterface& getBFloat16Hooks() {
  static std::unique_ptr<BFloat16HooksInterface> bfloat16_hooks;
  // NB: The once_flag here implies that if you try to call any BFloat16
  // functionality before you load the BFloat16 library, you're toast.
  // Same restriction as in getCUDAHooks()
  static std::once_flag once;
  std::call_once(once, [] {
    bfloat16_hooks = BFloat16HooksRegistry()->Create("BFloat16Hooks", BFloat16HooksArgs{});
    if (!bfloat16_hooks) {
      bfloat16_hooks =
          std::unique_ptr<BFloat16HooksInterface>(new BFloat16HooksInterface());
    }
  });
  return *bfloat16_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(
    BFloat16HooksRegistry,
    BFloat16HooksInterface,
    BFloat16HooksArgs)
}
