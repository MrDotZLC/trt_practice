#include "mini_trt_llm/plugins/iplugin_v3_base.hpp"

namespace mini_trt_llm {

// =============================================================================
// Inner capability classes: delegate everything to the parent plugin.
// =============================================================================

class IPluginV3Base::Core : public nvinfer1::IPluginV3OneCore {
 public:
    explicit Core(IPluginV3Base* plugin) : plugin_(plugin) {}

    nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override {
        return nvinfer1::InterfaceInfo{"PLUGIN_V3ONE_CORE", 1, 0};
    }

    const char* getPluginName() const noexcept override {
        return plugin_->getPluginType();
    }

    const char* getPluginVersion() const noexcept override {
        return plugin_->getPluginVersion();
    }

    const char* getPluginNamespace() const noexcept override {
        return plugin_->GetPluginNamespaceInternal();
    }

 private:
    IPluginV3Base* plugin_;
};

class IPluginV3Base::Build : public nvinfer1::IPluginV3OneBuild {
 public:
    explicit Build(IPluginV3Base* plugin) : plugin_(plugin) {}

    nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override {
        return nvinfer1::InterfaceInfo{"PLUGIN_V3ONE_BUILD", 1, 0};
    }

    int32_t getNbOutputs() const noexcept override {
        return plugin_->getNbOutputs();
    }

    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                            int32_t nbInputs,
                            const nvinfer1::DynamicPluginTensorDesc* out,
                            int32_t nbOutputs) noexcept override {
        return plugin_->configurePlugin(in, nbInputs, out, nbOutputs);
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
                               int32_t nbOutputs,
                               const nvinfer1::DataType* inputTypes,
                               int32_t nbInputs) const noexcept override {
        return plugin_->getOutputDataTypes(outputTypes, nbOutputs, inputTypes, nbInputs);
    }

    int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs,
                            int32_t nbInputs,
                            const nvinfer1::DimsExprs* shapeInputs,
                            int32_t nbShapeInputs,
                            nvinfer1::DimsExprs* outputs,
                            int32_t nbOutputs,
                            nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        return plugin_->getOutputShapes(inputs, nbInputs, shapeInputs, nbShapeInputs,
                                        outputs, nbOutputs, exprBuilder);
    }

    bool supportsFormatCombination(int32_t pos,
                                   const nvinfer1::DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override {
        return plugin_->supportsFormatCombination(pos, inOut, nbInputs, nbOutputs);
    }

    size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const nvinfer1::DynamicPluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override {
        return plugin_->getWorkspaceSize(inputs, nbInputs, outputs, nbOutputs);
    }

 private:
    IPluginV3Base* plugin_;
};

class IPluginV3Base::Runtime : public nvinfer1::IPluginV3OneRuntime {
 public:
    explicit Runtime(IPluginV3Base* plugin) : plugin_(plugin) {}

    nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override {
        return nvinfer1::InterfaceInfo{"PLUGIN_V3ONE_RUNTIME", 1, 0};
    }

    int32_t onShapeChange(const nvinfer1::PluginTensorDesc* in,
                          int32_t nbInputs,
                          const nvinfer1::PluginTensorDesc* out,
                          int32_t nbOutputs) noexcept override {
        return plugin_->onShapeChange(in, nbInputs, out, nbOutputs);
    }

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override {
        return plugin_->enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override {
        // Phase 1 不需要 per-context 资源，直接 clone 自身。
        (void)context;
        return plugin_->clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return plugin_->getFieldsToSerialize();
    }

 private:
    IPluginV3Base* plugin_;
};

// =============================================================================
// IPluginV3Base implementation
// =============================================================================

IPluginV3Base::IPluginV3Base()
    : core_(std::make_unique<Core>(this)),
      build_(std::make_unique<Build>(this)),
      runtime_(std::make_unique<Runtime>(this)) {}

IPluginV3Base::~IPluginV3Base() = default;

nvinfer1::IPluginCapability* IPluginV3Base::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return core_.get();
        case nvinfer1::PluginCapabilityType::kBUILD:
            return build_.get();
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return runtime_.get();
        default:
            return nullptr;
    }
}

std::string IPluginV3Base::ReadStringFromBuffer(const char*& buffer) const {
    uint32_t length = ReadFromBuffer<uint32_t>(buffer);
    std::string str(buffer, length);
    buffer += length;
    return str;
}

void IPluginV3Base::WriteStringToBuffer(char*& buffer, const std::string& str) const {
    uint32_t length = static_cast<uint32_t>(str.size());
    WriteToBuffer(buffer, length);
    std::memcpy(buffer, str.data(), length);
    buffer += length;
}

void IPluginV3Base::ResetFields() noexcept {
    fields_.clear();
    field_collection_.nbFields = 0;
    field_collection_.fields = nullptr;
}

void IPluginV3Base::AddField(const char* name,
                             const void* data,
                             int32_t type,
                             uint32_t length) noexcept {
    fields_.push_back(nvinfer1::PluginField{name, data,
                                            static_cast<nvinfer1::PluginFieldType>(type),
                                            static_cast<int32_t>(length)});
    field_collection_.nbFields = static_cast<int32_t>(fields_.size());
    field_collection_.fields = fields_.data();
}

nvinfer1::PluginFieldCollection* IPluginV3Base::GetFields() noexcept {
    return &field_collection_;
}

void IPluginV3Base::SetPluginNamespaceInternal(const char* pluginNamespace) noexcept {
    plugin_namespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* IPluginV3Base::GetPluginNamespaceInternal() const noexcept {
    return plugin_namespace_.c_str();
}

}  // namespace mini_trt_llm
