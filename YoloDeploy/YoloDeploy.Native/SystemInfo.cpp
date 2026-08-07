#include "YoloBridge.h"

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#define NOMINMAX
#include <windows.h>

namespace
{
    static std::wstring widenUtf8(const std::string& text)
    {
        if (text.empty())
            return L"";

        const int count = MultiByteToWideChar(
            CP_UTF8, 0,
            text.data(), static_cast<int>(text.size()),
            nullptr, 0);

        if (count <= 0)
            return L"";

        std::wstring result(
            static_cast<size_t>(count),
            L'\0');

        MultiByteToWideChar(
            CP_UTF8, 0,
            text.data(), static_cast<int>(text.size()),
            result.data(), count);

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

    static std::string jsonEscape(
        const std::string& text)
    {
        std::ostringstream oss;

        for (unsigned char ch : text)
        {
            switch (ch)
            {
            case '\"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (ch < 0x20)
                {
                    static const char* hex =
                        "0123456789abcdef";

                    oss << "\\u00"
                        << hex[(ch >> 4) & 0xF]
                        << hex[ch & 0xF];
                }
                else
                {
                    oss << static_cast<char>(ch);
                }
                break;
            }
        }

        return oss.str();
    }

    static void checkCuda(
        cudaError_t code,
        const char* operation)
    {
        if (code == cudaSuccess)
            return;

        std::ostringstream oss;
        oss << operation
            << " failed: "
            << cudaGetErrorString(code);

        throw std::runtime_error(
            oss.str());
    }
}

int32_t __cdecl YoloGetGpuInfoJson(
    wchar_t* jsonBuffer,
    int32_t jsonCapacity,
    wchar_t* errorBuffer,
    int32_t errorCapacity)
{
    try
    {
        int deviceCount = 0;

        checkCuda(
            cudaGetDeviceCount(&deviceCount),
            "cudaGetDeviceCount");

        if (deviceCount <= 0)
        {
            throw std::runtime_error(
                "No CUDA-capable NVIDIA GPU was detected.");
        }

        int deviceIndex = 0;

        checkCuda(
            cudaGetDevice(&deviceIndex),
            "cudaGetDevice");

        if (deviceIndex < 0 ||
            deviceIndex >= deviceCount)
        {
            deviceIndex = 0;
        }

        cudaDeviceProp prop{};

        checkCuda(
            cudaGetDeviceProperties(
                &prop,
                deviceIndex),
            "cudaGetDeviceProperties");

        int cudaRuntimeVersion = 0;
        int cudaDriverVersion = 0;

        checkCuda(
            cudaRuntimeGetVersion(
                &cudaRuntimeVersion),
            "cudaRuntimeGetVersion");

        checkCuda(
            cudaDriverGetVersion(
                &cudaDriverVersion),
            "cudaDriverGetVersion");

        std::ostringstream json;

        json
            << "{"
            << "\"deviceIndex\":" << deviceIndex << ","
            << "\"deviceCount\":" << deviceCount << ","
            << "\"name\":\"" << jsonEscape(prop.name) << "\","
            << "\"computeCapabilityMajor\":" << prop.major << ","
            << "\"computeCapabilityMinor\":" << prop.minor << ","
            << "\"totalGlobalMemoryBytes\":"
            << static_cast<unsigned long long>(
                prop.totalGlobalMem) << ","
            << "\"multiProcessorCount\":"
            << prop.multiProcessorCount << ","
            << "\"cudaRuntimeVersion\":"
            << cudaRuntimeVersion << ","
            << "\"cudaDriverVersion\":"
            << cudaDriverVersion << ","
            << "\"tensorRtMajor\":"
            << NV_TENSORRT_MAJOR << ","
            << "\"tensorRtMinor\":"
            << NV_TENSORRT_MINOR << ","
            << "\"tensorRtPatch\":"
            << NV_TENSORRT_PATCH << ","
            << "\"tensorRtBuild\":"
            << NV_TENSORRT_BUILD
            << "}";

        setWideBuffer(
            widenUtf8(json.str()),
            jsonBuffer,
            jsonCapacity);

        setWideBuffer(
            L"",
            errorBuffer,
            errorCapacity);

        return 0;
    }
    catch (const std::exception& ex)
    {
        setWideBuffer(
            L"",
            jsonBuffer,
            jsonCapacity);

        setWideBuffer(
            widenUtf8(ex.what()),
            errorBuffer,
            errorCapacity);

        return -1;
    }
    catch (...)
    {
        setWideBuffer(
            L"",
            jsonBuffer,
            jsonCapacity);

        setWideBuffer(
            L"Unknown error while querying GPU information.",
            errorBuffer,
            errorCapacity);

        return -1;
    }
}
