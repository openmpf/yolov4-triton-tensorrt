// The majority of this code is copyright Isarsoft and/or Wang Xinyu. It has been
// modified by MITRE to use unique names to allow multiple plugins for different
// dimensions to be loaded into Triton at the same time.

#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "../utils/profiler.h"
#include <iostream>

namespace Yolo416
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 416;
    static constexpr int INPUT_W = 416;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 8,
        INPUT_H / 8,
        {12,16, 19,36, 40,28}
    };
    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {36,75, 76,55, 72,146}
    };
    static constexpr YoloKernel yolo3 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {142,110, 192,243, 459,401}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}


namespace nvinfer1
{
    class Yolo416LayerPlugin: public IPluginV2IOExt
    {
        public:
            explicit Yolo416LayerPlugin();
            Yolo416LayerPlugin(const void* data, size_t length);

            ~Yolo416LayerPlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            int initialize() override;

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            virtual size_t getSerializationSize() const override;

            virtual void serialize(void* buffer) const override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override;

            const char* getPluginVersion() const override;

            void destroy() override;

            IPluginV2IOExt* clone() const override;

            void setPluginNamespace(const char* pluginNamespace) override;

            const char* getPluginNamespace() const override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

            void detachFromContext() override;

        private:
            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
            int mClassCount;
            int mKernelCount;
            std::vector<Yolo416::YoloKernel> mYoloKernel;
            int mThreadCount = 256;
            void** mAnchor;
            const char* mPluginNamespace;
    };

    class Yolo416PluginCreator : public IPluginCreator
    {
        public:
            Yolo416PluginCreator();

            ~Yolo416PluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };

    REGISTER_TENSORRT_PLUGIN(Yolo416PluginCreator);
};

#endif