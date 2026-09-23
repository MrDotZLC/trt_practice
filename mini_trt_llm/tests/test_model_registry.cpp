#include "mini_trt_llm/core/imodel_builder.hpp"
#include "mini_trt_llm/core/model_registry.hpp"
#include <gtest/gtest.h>

namespace mini_trt_llm {

class DummyBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "dummy"; }
    bool Build(nvinfer1::INetworkDefinition*, const WeightLoader&,
               const ModelConfig&) override {
        return true;
    }
};

TEST(ModelRegistryTest, RegisterAndGet) {
    ModelRegistry registry;
    auto builder = std::make_shared<DummyBuilder>();
    registry.Register("dummy", builder);

    EXPECT_TRUE(registry.Has("dummy"));
    auto got = registry.Get("dummy");
    EXPECT_NE(got, nullptr);
    EXPECT_EQ(got->Name(), "dummy");
    EXPECT_EQ(registry.List().size(), 1u);
}

}  // namespace mini_trt_llm
