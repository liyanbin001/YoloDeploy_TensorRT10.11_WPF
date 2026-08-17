#include "YoloBridge.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#define NOMINMAX
#include <windows.h>

namespace
{
    class BuildLogger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override
        {
            if (!msg)
                return;

            // Keep warnings/errors in the UI log; TensorRT INFO can be extremely verbose.
            if (severity <= Severity::kWARNING)
            {
                mLog << severityName(severity) << ": " << msg << "\n";

                OutputDebugStringA("[TensorRT Builder] ");
                OutputDebugStringA(msg);
                OutputDebugStringA("\n");
            }
        }

        std::string text() const
        {
            return mLog.str();
        }

    private:
        static const char* severityName(Severity severity) noexcept
        {
            switch (severity)
            {
            case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            case Severity::kERROR:          return "ERROR";
            case Severity::kWARNING:        return "WARNING";
            case Severity::kINFO:           return "INFO";
            case Severity::kVERBOSE:        return "VERBOSE";
            default:                        return "UNKNOWN";
            }
        }

        std::ostringstream mLog;
    };

    template <typename T>
    struct TrtDeleter
    {
        void operator()(T* p) const noexcept
        {
            delete p;
        }
    };

    template <typename T>
    using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

    static std::wstring widenUtf8(const std::string& text)
    {
        if (text.empty())
            return L"";

        int count = MultiByteToWideChar(
            CP_UTF8, 0,
            text.data(), static_cast<int>(text.size()),
            nullptr, 0);

        if (count <= 0)
            return L"";

        std::wstring result(static_cast<size_t>(count), L'\0');

        MultiByteToWideChar(
            CP_UTF8, 0,
            text.data(), static_cast<int>(text.size()),
            result.data(), count);

        return result;
    }

    static std::string narrowUtf8(const std::wstring& text)
    {
        if (text.empty())
            return "";

        int count = WideCharToMultiByte(
            CP_UTF8, 0,
            text.data(), static_cast<int>(text.size()),
            nullptr, 0, nullptr, nullptr);

        if (count <= 0)
            return "";

        std::string result(static_cast<size_t>(count), '\0');

        WideCharToMultiByte(
            CP_UTF8, 0,
            text.data(), static_cast<int>(text.size()),
            result.data(), count, nullptr, nullptr);

        return result;
    }

    static void setWideBuffer(
        const std::wstring& text,
        wchar_t* buffer,
        int32_t capacity)
    {
        if (!buffer || capacity <= 0)
            return;

        wcsncpy_s(
            buffer,
            static_cast<size_t>(capacity),
            text.c_str(),
            _TRUNCATE);
    }

    static std::string dimsToString(const nvinfer1::Dims& dims)
    {
        std::ostringstream oss;
        oss << "[";

        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (i)
                oss << ",";
            oss << dims.d[i];
        }

        oss << "]";
        return oss.str();
    }

    static bool hasDynamicDimension(const nvinfer1::Dims& dims)
    {
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] < 0)
                return true;
        }
        return false;
    }

    static nvinfer1::Dims makeProfileDims(
        nvinfer1::Dims dims,
        int inputWidth,
        int inputHeight)
    {
        if (dims.nbDims != 4)
            throw std::runtime_error(
                "Only 4D NCHW model inputs are supported.");

        // Standard YOLO input: N,C,H,W.
        if (dims.d[0] < 0) dims.d[0] = 1;
        if (dims.d[1] < 0) dims.d[1] = 3;
        if (dims.d[2] < 0) dims.d[2] = inputHeight;
        if (dims.d[3] < 0) dims.d[3] = inputWidth;

        if (dims.d[0] != 1)
            throw std::runtime_error("This deployment supports batch=1 only.");

        if (dims.d[1] != 3)
            throw std::runtime_error(
                "This deployment expects a 3-channel NCHW image input.");

        if (dims.d[2] <= 0 || dims.d[3] <= 0)
            throw std::runtime_error(
                "Failed to resolve ONNX input height/width.");

        return dims;
    }

    static std::string collectParserErrors(nvonnxparser::IParser& parser)
    {
        std::ostringstream oss;

        const int32_t count = parser.getNbErrors();
        for (int32_t i = 0; i < count; ++i)
        {
            const auto* error = parser.getError(i);
            if (!error)
                continue;

            oss << "ONNX parser [" << i << "]: "
                << error->desc() << "\n";
        }

        return oss.str();
    }
}

int32_t __cdecl YoloBuildEngineFromOnnx(
    const wchar_t* onnxPath,
    const wchar_t* enginePath,
    int32_t inputWidth,
    int32_t inputHeight,
    int32_t enableFp16,
    int32_t workspaceMiB,
    wchar_t* logBuffer,
    int32_t logCapacity,
    wchar_t* errorBuffer,
    int32_t errorCapacity)
{
    BuildLogger logger;
    std::ostringstream report;

    try
    {
        if (!onnxPath || !*onnxPath)
            throw std::runtime_error("ONNX path is empty.");

        if (!enginePath || !*enginePath)
            throw std::runtime_error("Engine output path is empty.");

        if (inputWidth <= 0 || inputHeight <= 0)
            throw std::runtime_error(
                "Input width/height must be positive.");

        if (workspaceMiB < 64)
            throw std::runtime_error(
                "Workspace should be at least 64 MiB.");

        const std::filesystem::path onnxFile(onnxPath);
        const std::filesystem::path engineFile(enginePath);

        if (!std::filesystem::exists(onnxFile))
            throw std::runtime_error("ONNX file does not exist.");

        if (engineFile.has_parent_path())
            std::filesystem::create_directories(engineFile.parent_path());

        int cudaDevice = 0;
        cudaDeviceProp deviceProp{};

        const cudaError_t getDeviceResult = cudaGetDevice(&cudaDevice);
        if (getDeviceResult == cudaSuccess)
        {
            const cudaError_t propResult =
                cudaGetDeviceProperties(&deviceProp, cudaDevice);

            if (propResult == cudaSuccess)
            {
                report
                    << "GPU: " << deviceProp.name
                    << " | Compute Capability "
                    << deviceProp.major << "." << deviceProp.minor
                    << "\n";
            }
        }

        report << "ONNX: " << narrowUtf8(onnxFile.wstring()) << "\n";
        report << "Engine: " << narrowUtf8(engineFile.wstring()) << "\n";
        report << "Requested input: [1,3,"
               << inputHeight << "," << inputWidth << "]\n";
        report << "Precision: "
               << (enableFp16 ? "FP16 builder mode" : "FP32")
               << "\n";
        report << "Workspace: " << workspaceMiB << " MiB\n";

        const auto start = std::chrono::steady_clock::now();

        TrtUniquePtr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(logger));

        if (!builder)
            throw std::runtime_error(
                "TensorRT createInferBuilder failed.");

        TrtUniquePtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(0U));

        if (!network)
            throw std::runtime_error(
                "TensorRT createNetworkV2 failed.");

        TrtUniquePtr<nvinfer1::IBuilderConfig> config(
            builder->createBuilderConfig());

        if (!config)
            throw std::runtime_error(
                "TensorRT createBuilderConfig failed.");

        TrtUniquePtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, logger));

        if (!parser)
            throw std::runtime_error(
                "nvonnxparser::createParser failed.");

        const std::string onnxUtf8 =
            narrowUtf8(onnxFile.wstring());

        const bool parsed = parser->parseFromFile(
            onnxUtf8.c_str(),
            static_cast<int32_t>(
                nvinfer1::ILogger::Severity::kWARNING));

        const std::string parserMessages =
            collectParserErrors(*parser);

        if (!parserMessages.empty())
            report << parserMessages;

        if (!parsed)
            throw std::runtime_error(
                "TensorRT ONNX parser failed. See build log.");

        if (network->getNbInputs() != 1)
        {
            std::ostringstream oss;
            oss << "This deployment requires exactly one ONNX input, but model has "
                << network->getNbInputs() << ".";
            throw std::runtime_error(oss.str());
        }

        const int outputCount =
            network->getNbOutputs();

        if (outputCount < 1 ||
            outputCount > 2)
        {
            std::ostringstream oss;
            oss
                << "This deployment supports one ONNX output "
                << "(Detect/OBB) or two outputs "
                << "(YOLO26 Seg prediction + proto), but model has "
                << outputCount
                << ".";

            throw std::runtime_error(
                oss.str());
        }

        nvinfer1::ITensor* input =
            network->getInput(0);

        if (!input)
        {
            throw std::runtime_error(
                "Unable to read ONNX input tensor.");
        }

        const nvinfer1::Dims originalInputDims =
            input->getDimensions();

        report
            << "Parsed input: "
            << (input->getName()
                    ? input->getName()
                    : "<unnamed>")
            << " "
            << dimsToString(
                originalInputDims)
            << "\n";

        for (int i = 0;
             i < outputCount;
             ++i)
        {
            nvinfer1::ITensor* output =
                network->getOutput(i);

            if (!output)
            {
                throw std::runtime_error(
                    "Unable to read an ONNX output tensor.");
            }

            report
                << "Parsed output["
                << i
                << "]: "
                << (output->getName()
                        ? output->getName()
                        : "<unnamed>")
                << " "
                << dimsToString(
                    output->getDimensions())
                << "\n";
        }

        if (originalInputDims.nbDims != 4)
            throw std::runtime_error(
                "This deployment supports only NCHW 4D image input.");

        if (originalInputDims.d[0] > 0 &&
            originalInputDims.d[0] != 1)
        {
            throw std::runtime_error(
                "This deployment supports batch=1 only.");
        }

        if (originalInputDims.d[1] > 0 &&
            originalInputDims.d[1] != 3)
        {
            throw std::runtime_error(
                "This deployment expects 3 input channels.");
        }

        if (hasDynamicDimension(originalInputDims))
        {
            nvinfer1::IOptimizationProfile* profile =
                builder->createOptimizationProfile();

            if (!profile)
                throw std::runtime_error(
                    "createOptimizationProfile failed.");

            const nvinfer1::Dims profileDims =
                makeProfileDims(
                    originalInputDims,
                    inputWidth,
                    inputHeight);

            const char* inputName = input->getName();
            if (!inputName)
                throw std::runtime_error(
                    "Dynamic ONNX input has no tensor name.");

            // Phase 1 intentionally uses a single fixed profile shape.
            // This makes the generated engine predictable and directly
            // compatible with the existing WPF inference path.
            if (!profile->setDimensions(
                    inputName,
                    nvinfer1::OptProfileSelector::kMIN,
                    profileDims) ||
                !profile->setDimensions(
                    inputName,
                    nvinfer1::OptProfileSelector::kOPT,
                    profileDims) ||
                !profile->setDimensions(
                    inputName,
                    nvinfer1::OptProfileSelector::kMAX,
                    profileDims))
            {
                throw std::runtime_error(
                    "Failed to set optimization profile dimensions.");
            }

            if (!profile->isValid())
                throw std::runtime_error(
                    "TensorRT optimization profile is invalid.");

            const int32_t profileIndex =
                config->addOptimizationProfile(profile);

            if (profileIndex < 0)
                throw std::runtime_error(
                    "addOptimizationProfile failed.");

            report << "Dynamic profile: "
                   << dimsToString(profileDims)
                   << " (MIN=OPT=MAX)\n";
        }
        else
        {
            const int fixedHeight = originalInputDims.d[2];
            const int fixedWidth = originalInputDims.d[3];

            if (fixedHeight <= 0 || fixedWidth <= 0)
            {
                throw std::runtime_error(
                    "Fixed ONNX input has invalid height/width.");
            }

            if (fixedWidth != inputWidth ||
                fixedHeight != inputHeight)
            {
                std::ostringstream oss;
                oss
                    << "Fixed ONNX input shape mismatch. "
                    << "ONNX is [1,3,"
                    << fixedHeight << "," << fixedWidth
                    << "], but UI requested [1,3,"
                    << inputHeight << "," << inputWidth
                    << "]. Re-export the ONNX with the desired fixed size "
                    << "or set the UI width/height to match the ONNX model.";

                throw std::runtime_error(
                    oss.str());
            }

            report
                << "Fixed ONNX input verified: [1,3,"
                << fixedHeight << "," << fixedWidth
                << "]\n";
        }

        const size_t workspaceBytes =
            static_cast<size_t>(workspaceMiB)
            * 1024ULL * 1024ULL;

        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE,
            workspaceBytes);

        if (enableFp16)
        {
            // TensorRT 10.x supports BuilderFlag::kFP16.
            // TensorRT may still keep layers in FP32 where required.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        TrtUniquePtr<nvinfer1::IHostMemory> serializedEngine(
            builder->buildSerializedNetwork(
                *network,
                *config));

        if (!serializedEngine)
            throw std::runtime_error(
                "buildSerializedNetwork failed. "
                "Check TensorRT builder log, workspace and ONNX support.");

        std::ofstream outputFile(
            engineFile,
            std::ios::binary | std::ios::trunc);

        if (!outputFile)
            throw std::runtime_error(
                "Cannot create output .engine file.");

        outputFile.write(
            static_cast<const char*>(serializedEngine->data()),
            static_cast<std::streamsize>(serializedEngine->size()));

        if (!outputFile.good())
            throw std::runtime_error(
                "Failed while writing .engine file.");

        outputFile.close();

        const auto finish = std::chrono::steady_clock::now();
        const auto elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                finish - start).count();

        report << "Serialized engine size: "
               << serializedEngine->size()
               << " bytes\n";

        report << "Build time: "
               << elapsedMs << " ms\n";

        report << "Result: SUCCESS\n";

        const std::string trtLog = logger.text();
        if (!trtLog.empty())
        {
            report << "\nTensorRT warnings/errors:\n"
                   << trtLog;
        }

        setWideBuffer(
            widenUtf8(report.str()),
            logBuffer,
            logCapacity);

        setWideBuffer(
            L"",
            errorBuffer,
            errorCapacity);

        return 0;
    }
    catch (const std::exception& ex)
    {
        const std::string trtLog = logger.text();

        if (!trtLog.empty())
        {
            report << "\nTensorRT warnings/errors:\n"
                   << trtLog;
        }

        report << "\nResult: FAILED\n";

        setWideBuffer(
            widenUtf8(report.str()),
            logBuffer,
            logCapacity);

        setWideBuffer(
            widenUtf8(ex.what()),
            errorBuffer,
            errorCapacity);

        return -1;
    }
    catch (...)
    {
        report << "\nResult: FAILED\n";

        setWideBuffer(
            widenUtf8(report.str()),
            logBuffer,
            logCapacity);

        setWideBuffer(
            L"Unknown error while building TensorRT engine.",
            errorBuffer,
            errorCapacity);

        return -1;
    }
}
