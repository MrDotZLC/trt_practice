#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace mini_trt_llm {

// IPluginV3 插件基类封装，适配 TensorRT 10.x。
//
// TensorRT 10.x 将 IPluginV3 拆分为三个能力接口：
//   - IPluginV3OneCore：插件元数据（name / version / namespace）
//   - IPluginV3OneBuild：构建期行为（shape、dtype、format、workspace、output count）
//   - IPluginV3OneRuntime：运行期行为（enqueue、序列化）
//
// 本基类把三个能力接口收敛到统一的纯虚函数中，子类只需实现业务逻辑，
// 无需重复实现能力接口的分发。
class IPluginV3Base : public nvinfer1::IPluginV3 {
 public:
    virtual ~IPluginV3Base();

    // -----------------------------------------------------------------------
    // IPluginV3 必需接口
    // -----------------------------------------------------------------------

    // 返回指定类型的能力接口指针。
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;

    // 克隆插件。子类必须实现，返回一个携带相同属性的新插件对象。
    IPluginV3Base* clone() noexcept override = 0;

    // -----------------------------------------------------------------------
    // Core 能力：子类必须实现
    // -----------------------------------------------------------------------

    virtual const char* getPluginType() const noexcept = 0;
    virtual const char* getPluginVersion() const noexcept = 0;

    // -----------------------------------------------------------------------
    // Build 能力：子类必须实现
    // -----------------------------------------------------------------------

    virtual int32_t getNbOutputs() const noexcept = 0;

    virtual int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
                                       int32_t nbOutputs,
                                       const nvinfer1::DataType* inputTypes,
                                       int32_t nbInputs) const noexcept = 0;

    virtual int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs,
                                    int32_t nbInputs,
                                    const nvinfer1::DimsExprs* shapeInputs,
                                    int32_t nbShapeInputs,
                                    nvinfer1::DimsExprs* outputs,
                                    int32_t nbOutputs,
                                    nvinfer1::IExprBuilder& exprBuilder) noexcept = 0;

    virtual bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::DynamicPluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept = 0;

    virtual int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                    int32_t nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc* out,
                                    int32_t nbOutputs) noexcept = 0;

    virtual size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                    int32_t nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc* outputs,
                                    int32_t nbOutputs) const noexcept = 0;

    // -----------------------------------------------------------------------
    // Runtime 能力：子类必须实现
    // -----------------------------------------------------------------------

    virtual int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                            const nvinfer1::PluginTensorDesc* outputDesc,
                            const void* const* inputs,
                            void* const* outputs,
                            void* workspace,
                            cudaStream_t stream) noexcept = 0;

    virtual int32_t onShapeChange(const nvinfer1::PluginTensorDesc* in,
                                  int32_t nbInputs,
                                  const nvinfer1::PluginTensorDesc* out,
                                  int32_t nbOutputs) noexcept = 0;

    virtual nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept = 0;

 protected:
    IPluginV3Base();

    // -----------------------------------------------------------------------
    // 序列化/反序列化辅助函数
    // -----------------------------------------------------------------------

    template <typename T>
    void WriteToBuffer(char*& buffer, const T& value) const noexcept {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        std::memcpy(buffer, &value, sizeof(T));
        buffer += sizeof(T);
    }

    template <typename T>
    T ReadFromBuffer(const char*& buffer) const noexcept {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        T value{};
        std::memcpy(&value, buffer, sizeof(T));
        buffer += sizeof(T);
        return value;
    }

    // 从 buffer 读取 std::string（先读长度，再读内容）。
    std::string ReadStringFromBuffer(const char*& buffer) const;
    void WriteStringToBuffer(char*& buffer, const std::string& str) const;

    // -----------------------------------------------------------------------
    // PluginFieldCollection 辅助
    // -----------------------------------------------------------------------

    // 清空并重新填充 fields_，返回指向 fields_ 的 PluginFieldCollection。
    // 子类在 getFieldsToSerialize() 中先调用 ResetFields()，再 AddField()。
    void ResetFields() noexcept;
    void AddField(const char* name, const void* data, int32_t type, uint32_t length) noexcept;
    nvinfer1::PluginFieldCollection* GetFields() noexcept;

    // -----------------------------------------------------------------------
    // namespace 辅助
    // -----------------------------------------------------------------------

    void SetPluginNamespaceInternal(const char* pluginNamespace) noexcept;
    const char* GetPluginNamespaceInternal() const noexcept;

 private:
    class Core;
    class Build;
    class Runtime;

    std::unique_ptr<Core> core_;
    std::unique_ptr<Build> build_;
    std::unique_ptr<Runtime> runtime_;

    std::string plugin_namespace_;

    // PluginFieldCollection 内部缓存。
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
};

}  // namespace mini_trt_llm
