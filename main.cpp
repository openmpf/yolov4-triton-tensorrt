// The majority of this code is copyright Isarsoft and/or Wang Xinyu. It has been
// modified by MITRE to generate the engine file on a Triton server in a specific
// directory with a specific name, and to exit if the file already exists.

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "NvInferVersion.h"
#include "version.h"

#include "networks/yolov4.h"

#include "utils/logging.h"
static Logger gLogger;

#include <iostream>
#include <sys/stat.h>
#include <math.h>

#define DEVICE 0
#define BATCH_SIZE 16

using namespace nvinfer1;

int main(int argc, char** argv) {

    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);
    std::string cudaVersionMajor = std::to_string(cudaVersion / 1000);
    std::string cudaVersionMinor = std::to_string((int) fmod(cudaVersion, 1000) / 10);

    cudaDeviceProp gpuProps;
    cudaGetDeviceProperties(&gpuProps, DEVICE);
    std::string gpuName = gpuProps.name;
    // replace ' ' with '_' and convert to lower-case
    std::replace(gpuName.begin(), gpuName.end(), ' ', '_');
    std::transform(gpuName.begin(), gpuName.end(), gpuName.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    std::string engineDir = std::string("/models/yolo-")
                            + std::to_string(Yolo416::INPUT_W)
                            + "/1";
    std::string enginePath = engineDir + "/" + std::string("yolov4")
                             + ".dim" + std::to_string(Yolo416::INPUT_W)
                             + ".bs" + std::to_string(BATCH_SIZE)
                             + ".cuda" + cudaVersionMajor
                             + "." + cudaVersionMinor
                             + ".trt" + std::to_string(NV_TENSORRT_MAJOR)
                             + "." + std::to_string(NV_TENSORRT_MINOR)
                             + "." + std::to_string(NV_TENSORRT_PATCH)
                             + "." + gpuName
                             + ".engine"
                             + "." + std::to_string(YOLOV4_VERSION_MAJOR)
                             + "." + std::to_string(YOLOV4_VERSION_MINOR)
                             + "." + std::to_string(YOLOV4_VERSION_PATCH);

    struct stat statBuffer;
    if (stat(enginePath.c_str(), &statBuffer) == 0) {
        std::cout << "Skipping engine file generation. File exists: " << enginePath << std::endl;
        return 0;
    }

    std::string command = std::string("mkdir -p ") + engineDir;
    if (std::system(command.c_str()) != 0) {
        std::cerr << "Could not create directory: " << engineDir << std::endl;
        return -1;
    }

    std::cout << "Generating " << enginePath << std::endl;
    cudaSetDevice(DEVICE);

    std::cout << "Creating builder" << std::endl;
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    std::cout << "Creating model" << std::endl;
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = yolov4::createEngine(BATCH_SIZE, builder, config, DataType::kFLOAT, "yolov4.wts");
    assert(engine != nullptr);

    std::cout << "Serializing model to engine file" << std::endl;
    // Serialize the engine
    IHostMemory* modelStream{nullptr};
    modelStream = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();

    assert(modelStream != nullptr);
    std::ofstream p(enginePath.c_str(), std::ios::binary);
    if (!p) {
        std::cerr << "Could not open engine output file: " << enginePath << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();

    std::cout << "Done" << std::endl;

    return 0;
}
