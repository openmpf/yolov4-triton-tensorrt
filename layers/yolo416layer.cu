#include "yolo416layer.h"

using namespace Yolo416;

namespace nvinfer1
{
    Yolo416LayerPlugin::Yolo416LayerPlugin()
    {
        mClassCount = CLASS_NUM;
        mYoloKernel.clear();
        mYoloKernel.push_back(yolo1);
        mYoloKernel.push_back(yolo2);
        mYoloKernel.push_back(yolo3);

        mKernelCount = mYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        for(int ii = 0; ii < mKernelCount; ii ++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii],AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    
    Yolo416LayerPlugin::~Yolo416LayerPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    Yolo416LayerPlugin::Yolo416LayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(mYoloKernel.data(),d,kernelSize);
        d += kernelSize;

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        for(int ii = 0; ii < mKernelCount; ii ++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii],AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }

        assert(d == a + length);
    }

    void Yolo416LayerPlugin::serialize(void* buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(),kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }
    
    size_t Yolo416LayerPlugin::getSerializationSize() const
    {  
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(Yolo416::YoloKernel) * mYoloKernel.size();
    }

    int Yolo416LayerPlugin::initialize()
    { 
        return 0;
    }
    
    Dims Yolo416LayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void Yolo416LayerPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* Yolo416LayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType Yolo416LayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool Yolo416LayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool Yolo416LayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void Yolo416LayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void Yolo416LayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void Yolo416LayerPlugin::detachFromContext() {}

    const char* Yolo416LayerPlugin::getPluginType() const
    {
        return "Yolo416Layer_TRT";
    }

    const char* Yolo416LayerPlugin::getPluginVersion() const
    {
        return "1";
    }

    void Yolo416LayerPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* Yolo416LayerPlugin::clone() const
    {
        Yolo416LayerPlugin *p = new Yolo416LayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1./(1. + exp(-data)); };

    __global__ void CalYolo416Detection(const float *input, float *output,int noElements,
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

            float *res_count = output + bnIdx*outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
            det->bbox[2] = exp(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = exp(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }

    void Yolo416LayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {

        int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CalYolo416Detection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount ,outputElem);
        }

    }


    int Yolo416LayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);

        return 0;
    }

    PluginFieldCollection Yolo416PluginCreator::mFC{};
    std::vector<PluginField> Yolo416PluginCreator::mPluginAttributes;

    Yolo416PluginCreator::Yolo416PluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* Yolo416PluginCreator::getPluginName() const
    {
            return "Yolo416Layer_TRT";
    }

    const char* Yolo416PluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* Yolo416PluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* Yolo416PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        Yolo416LayerPlugin* obj = new Yolo416LayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* Yolo416PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        Yolo416LayerPlugin* obj = new Yolo416LayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}