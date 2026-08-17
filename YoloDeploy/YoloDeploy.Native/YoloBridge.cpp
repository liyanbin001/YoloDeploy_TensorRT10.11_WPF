#include "YoloBridge.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define NOMINMAX
#include <windows.h>

namespace
{
    class TrtLogger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override
        {
            if (severity <= Severity::kWARNING)
            {
                OutputDebugStringA("[TensorRT] ");
                OutputDebugStringA(msg);
                OutputDebugStringA("\n");
            }
        }
    };

    TrtLogger gLogger;

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

    struct DeviceBuffer
    {
        void* ptr = nullptr;
        size_t bytes = 0;

        ~DeviceBuffer()
        {
            if (ptr)
                cudaFree(ptr);
        }

        DeviceBuffer() = default;
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    };

    struct StreamHolder
    {
        cudaStream_t stream = nullptr;
        ~StreamHolder()
        {
            if (stream)
                cudaStreamDestroy(stream);
        }
    };

    struct Candidate
    {
        float x1 = 0;
        float y1 = 0;
        float x2 = 0;
        float y2 = 0;
        float score = 0;
        int classId = -1;
    };

    struct ObbCandidate
    {
        float centerX = 0;
        float centerY = 0;
        float width = 0;
        float height = 0;
        float angle = 0;
        float score = 0;
        int classId = -1;
    };

    struct ObbCorners
    {
        float p1x = 0;
        float p1y = 0;
        float p2x = 0;
        float p2y = 0;
        float p3x = 0;
        float p3y = 0;
        float p4x = 0;
        float p4y = 0;
    };

    static std::wstring widen(const std::string& text)
    {
        if (text.empty()) return L"";
        int count = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
        if (count <= 0) return L"";
        std::wstring result(count, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), count);
        return result;
    }

    static std::string narrow(const std::wstring& text)
    {
        if (text.empty()) return "";
        int count = WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
        if (count <= 0) return "";
        std::string result(count, '\0');
        WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), count, nullptr, nullptr);
        return result;
    }

    static void setError(const std::wstring& msg, wchar_t* buffer, int32_t capacity)
    {
        if (!buffer || capacity <= 0) return;
        wcsncpy_s(buffer, static_cast<size_t>(capacity), msg.c_str(), _TRUNCATE);
    }

    static void checkCuda(cudaError_t code, const char* what)
    {
        if (code != cudaSuccess)
        {
            std::ostringstream oss;
            oss << what << ": " << cudaGetErrorString(code);
            throw std::runtime_error(oss.str());
        }
    }

    static size_t dataTypeSize(nvinfer1::DataType type)
    {
        switch (type)
        {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kFP8:   return 1;
        case nvinfer1::DataType::kBF16:  return 2;
        case nvinfer1::DataType::kINT64: return 8;
#endif
        default: throw std::runtime_error("Unsupported TensorRT data type.");
        }
    }

    static const char* dataTypeName(nvinfer1::DataType type)
    {
        switch (type)
        {
        case nvinfer1::DataType::kFLOAT: return "FP32";
        case nvinfer1::DataType::kHALF: return "FP16";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kBOOL: return "BOOL";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kUINT8: return "UINT8";
        case nvinfer1::DataType::kFP8: return "FP8";
        case nvinfer1::DataType::kBF16: return "BF16";
        case nvinfer1::DataType::kINT64: return "INT64";
#endif
        default: return "UNKNOWN";
        }
    }

    static int64_t volume(const nvinfer1::Dims& dims)
    {
        int64_t v = 1;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] <= 0)
                throw std::runtime_error("Tensor has unresolved or invalid dimensions.");
            v *= static_cast<int64_t>(dims.d[i]);
        }
        return v;
    }

    static std::string dimsToString(const nvinfer1::Dims& dims)
    {
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (i) oss << ",";
            oss << dims.d[i];
        }
        oss << "]";
        return oss.str();
    }

    static std::vector<char> readAllBytes(const std::wstring& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Cannot open engine file.");

        std::streamsize size = file.tellg();
        if (size <= 0)
            throw std::runtime_error("Engine file is empty.");

        file.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(size));
        if (!file.read(bytes.data(), size))
            throw std::runtime_error("Failed to read engine file.");
        return bytes;
    }

    static float clampf(float value, float low, float high)
    {
        return std::max(low, std::min(high, value));
    }

    static float iou(const Candidate& a, const Candidate& b)
    {
        const float xx1 = std::max(a.x1, b.x1);
        const float yy1 = std::max(a.y1, b.y1);
        const float xx2 = std::min(a.x2, b.x2);
        const float yy2 = std::min(a.y2, b.y2);

        const float w = std::max(0.0f, xx2 - xx1);
        const float h = std::max(0.0f, yy2 - yy1);
        const float inter = w * h;
        const float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
        const float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
        const float denom = areaA + areaB - inter;
        return denom > 0.0f ? inter / denom : 0.0f;
    }

    static std::vector<Candidate> classAwareNms(
        std::vector<Candidate> candidates,
        float iouThreshold)
    {
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b)
            {
                return a.score > b.score;
            });

        std::vector<Candidate> kept;
        std::vector<uint8_t> removed(candidates.size(), 0);

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (removed[i]) continue;
            kept.push_back(candidates[i]);

            for (size_t j = i + 1; j < candidates.size(); ++j)
            {
                if (removed[j]) continue;
                if (candidates[i].classId != candidates[j].classId) continue;
                if (iou(candidates[i], candidates[j]) > iouThreshold)
                    removed[j] = 1;
            }
        }
        return kept;
    }


    static float probabilisticObbIou(
        const ObbCandidate& a,
        const ObbCandidate& b)
    {
        // Scalar equivalent of Ultralytics ProbIoU used by rotated NMS.
        // OBB format: center-x, center-y, width, height, angle (radians).
        constexpr float eps = 1e-7f;

        auto covariance = [](const ObbCandidate& box, float& ca, float& cb, float& cc)
        {
            const float w2 = box.width * box.width / 12.0f;
            const float h2 = box.height * box.height / 12.0f;

            const float c = std::cos(box.angle);
            const float s = std::sin(box.angle);

            const float c2 = c * c;
            const float s2 = s * s;

            ca = w2 * c2 + h2 * s2;
            cb = w2 * s2 + h2 * c2;
            cc = (w2 - h2) * c * s;
        };

        float a1 = 0, b1 = 0, c1 = 0;
        float a2 = 0, b2 = 0, c2 = 0;

        covariance(a, a1, b1, c1);
        covariance(b, a2, b2, c2);

        const float dx = a.centerX - b.centerX;
        const float dy = a.centerY - b.centerY;

        const float aa = a1 + a2;
        const float bb = b1 + b2;
        const float cc = c1 + c2;

        const float denom = aa * bb - cc * cc + eps;

        const float t1 =
            ((aa * dy * dy) + (bb * dx * dx))
            / denom
            * 0.25f;

        const float t2 =
            (cc * (b.centerX - a.centerX) * (a.centerY - b.centerY))
            / denom
            * 0.5f;

        const float det1 = std::max(a1 * b1 - c1 * c1, 0.0f);
        const float det2 = std::max(a2 * b2 - c2 * c2, 0.0f);

        const float logArg =
            (aa * bb - cc * cc)
            / (4.0f * std::sqrt(det1 * det2) + eps)
            + eps;

        const float t3 =
            std::log(std::max(logArg, eps))
            * 0.5f;

        float bd = t1 + t2 + t3;
        bd = clampf(bd, eps, 100.0f);

        const float hd =
            std::sqrt(std::max(1.0f - std::exp(-bd) + eps, 0.0f));

        return clampf(1.0f - hd, 0.0f, 1.0f);
    }

    static std::vector<ObbCandidate> classAwareRotatedNms(
        std::vector<ObbCandidate> candidates,
        float threshold)
    {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const ObbCandidate& a, const ObbCandidate& b)
            {
                return a.score > b.score;
            });

        std::vector<ObbCandidate> kept;
        kept.reserve(candidates.size());

        // Ultralytics rotated NMS evaluates each candidate against all
        // higher-confidence boxes (upper-triangle ProbIoU matrix).
        // Restrict the comparison to the same class to preserve the existing
        // class-aware behavior of this deployment application.
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            bool suppress = false;

            for (size_t j = 0; j < i; ++j)
            {
                if (candidates[i].classId != candidates[j].classId)
                    continue;

                if (probabilisticObbIou(
                        candidates[j],
                        candidates[i]) >= threshold)
                {
                    suppress = true;
                    break;
                }
            }

            if (!suppress)
                kept.push_back(candidates[i]);
        }

        return kept;
    }

    static ObbCorners obbToCorners(
        const ObbCandidate& box)
    {
        const float halfW = box.width * 0.5f;
        const float halfH = box.height * 0.5f;

        const float c = std::cos(box.angle);
        const float s = std::sin(box.angle);

        auto rotatePoint =
            [&](float dx, float dy, float& x, float& y)
            {
                // Image coordinates use +Y downward, therefore the normal
                // 2D rotation formula visually follows Ultralytics' clockwise
                // positive-angle convention.
                x = box.centerX + dx * c - dy * s;
                y = box.centerY + dx * s + dy * c;
            };

        ObbCorners points;

        rotatePoint(-halfW, -halfH, points.p1x, points.p1y);
        rotatePoint( halfW, -halfH, points.p2x, points.p2y);
        rotatePoint( halfW,  halfH, points.p3x, points.p3y);
        rotatePoint(-halfW,  halfH, points.p4x, points.p4y);

        return points;
    }

    struct LetterboxInfo
    {
        float scale = 1.0f;
        int left = 0;
        int top = 0;
        int resizedW = 0;
        int resizedH = 0;
    };

    static inline float sampleBilinearChannel(
        const uint8_t* bgra,
        int srcW,
        int srcH,
        int stride,
        float sx,
        float sy,
        int bgraChannel)
    {
        sx = clampf(sx, 0.0f, static_cast<float>(srcW - 1));
        sy = clampf(sy, 0.0f, static_cast<float>(srcH - 1));

        int x0 = static_cast<int>(std::floor(sx));
        int y0 = static_cast<int>(std::floor(sy));
        int x1 = std::min(x0 + 1, srcW - 1);
        int y1 = std::min(y0 + 1, srcH - 1);

        float dx = sx - static_cast<float>(x0);
        float dy = sy - static_cast<float>(y0);

        auto at = [&](int x, int y) -> float
        {
            return static_cast<float>(bgra[y * stride + x * 4 + bgraChannel]);
        };

        float v00 = at(x0, y0);
        float v01 = at(x1, y0);
        float v10 = at(x0, y1);
        float v11 = at(x1, y1);

        float top = v00 + (v01 - v00) * dx;
        float bottom = v10 + (v11 - v10) * dx;
        return top + (bottom - top) * dy;
    }

    static LetterboxInfo preprocessBgra(
        const uint8_t* bgra,
        int srcW,
        int srcH,
        int stride,
        int dstW,
        int dstH,
        std::vector<float>& chw)
    {
        if (!bgra || srcW <= 0 || srcH <= 0 || stride < srcW * 4)
            throw std::runtime_error("Invalid BGRA image.");

        const float r = std::min(
            static_cast<float>(dstW) / static_cast<float>(srcW),
            static_cast<float>(dstH) / static_cast<float>(srcH));

        const int resizedW = static_cast<int>(std::round(srcW * r));
        const int resizedH = static_cast<int>(std::round(srcH * r));

        const float dw = static_cast<float>(dstW - resizedW) / 2.0f;
        const float dh = static_cast<float>(dstH - resizedH) / 2.0f;

        // Same practical centering convention used by Ultralytics LetterBox.
        const int left = static_cast<int>(std::round(dw - 0.1f));
        const int top = static_cast<int>(std::round(dh - 0.1f));

        const size_t plane = static_cast<size_t>(dstW) * dstH;
        chw.assign(plane * 3, 114.0f / 255.0f);

        for (int y = 0; y < resizedH; ++y)
        {
            // OpenCV-style half-pixel resize coordinate.
            float sy = (static_cast<float>(y) + 0.5f) / r - 0.5f;
            int dy = top + y;
            if (dy < 0 || dy >= dstH) continue;

            for (int x = 0; x < resizedW; ++x)
            {
                float sx = (static_cast<float>(x) + 0.5f) / r - 0.5f;
                int dx = left + x;
                if (dx < 0 || dx >= dstW) continue;

                // BGRA -> RGB
                float red   = sampleBilinearChannel(bgra, srcW, srcH, stride, sx, sy, 2) / 255.0f;
                float green = sampleBilinearChannel(bgra, srcW, srcH, stride, sx, sy, 1) / 255.0f;
                float blue  = sampleBilinearChannel(bgra, srcW, srcH, stride, sx, sy, 0) / 255.0f;

                size_t p = static_cast<size_t>(dy) * dstW + dx;
                chw[p] = red;
                chw[plane + p] = green;
                chw[plane * 2 + p] = blue;
            }
        }

        return LetterboxInfo{ r, left, top, resizedW, resizedH };
    }


    struct SegCandidate
    {
        float x1 = 0;
        float y1 = 0;
        float x2 = 0;
        float y2 = 0;
        float score = 0;
        int classId = -1;
        std::vector<float> coefficients;
    };

    struct PointF
    {
        float x = 0;
        float y = 0;
    };

    struct MinAreaRectResult
    {
        bool valid = false;
        float centerX = 0;
        float centerY = 0;
        float width = 0;
        float height = 0;
        float angle = 0;
        ObbCorners corners{};
    };

    static float pointCross(
        const PointF& o,
        const PointF& a,
        const PointF& b)
    {
        return
            (a.x - o.x) * (b.y - o.y)
            - (a.y - o.y) * (b.x - o.x);
    }

    static std::vector<PointF> convexHull(
        std::vector<PointF> points)
    {
        if (points.size() <= 1)
            return points;

        std::sort(
            points.begin(),
            points.end(),
            [](const PointF& a, const PointF& b)
            {
                if (a.x != b.x)
                    return a.x < b.x;
                return a.y < b.y;
            });

        points.erase(
            std::unique(
                points.begin(),
                points.end(),
                [](const PointF& a, const PointF& b)
                {
                    return a.x == b.x && a.y == b.y;
                }),
            points.end());

        if (points.size() <= 1)
            return points;

        std::vector<PointF> hull(
            points.size() * 2);

        size_t k = 0;

        for (const auto& p : points)
        {
            while (k >= 2 &&
                   pointCross(
                       hull[k - 2],
                       hull[k - 1],
                       p) <= 0.0f)
            {
                --k;
            }

            hull[k++] = p;
        }

        for (size_t i = points.size() - 1,
                    t = k + 1;
             i > 0;
             --i)
        {
            const auto& p =
                points[i - 1];

            while (k >= t &&
                   pointCross(
                       hull[k - 2],
                       hull[k - 1],
                       p) <= 0.0f)
            {
                --k;
            }

            hull[k++] = p;
        }

        if (k > 1)
            --k;

        hull.resize(k);
        return hull;
    }

    static float normalizeLongAxisAngle(
        float angle)
    {
        constexpr float pi =
            3.14159265358979323846f;

        angle =
            std::fmod(
                angle,
                pi);

        if (angle < 0.0f)
            angle += pi;

        return angle;
    }

    static MinAreaRectResult minimumAreaRect(
        const std::vector<PointF>& boundaryPoints)
    {
        MinAreaRectResult result;

        std::vector<PointF> hull =
            convexHull(
                boundaryPoints);

        if (hull.empty())
            return result;

        if (hull.size() == 1)
        {
            result.valid = true;
            result.centerX = hull[0].x;
            result.centerY = hull[0].y;
            result.width = 1.0f;
            result.height = 1.0f;
            result.angle = 0.0f;

            ObbCandidate box{
                result.centerX,
                result.centerY,
                result.width,
                result.height,
                result.angle,
                1.0f,
                0
            };

            result.corners =
                obbToCorners(box);

            return result;
        }

        float bestArea =
            std::numeric_limits<float>::max();

        float bestAngle = 0.0f;
        float bestMinU = 0.0f;
        float bestMaxU = 0.0f;
        float bestMinV = 0.0f;
        float bestMaxV = 0.0f;

        const size_t count =
            hull.size();

        for (size_t i = 0;
             i < count;
             ++i)
        {
            const PointF& a =
                hull[i];

            const PointF& b =
                hull[(i + 1) % count];

            const float dx =
                b.x - a.x;

            const float dy =
                b.y - a.y;

            if (std::abs(dx) < 1e-6f &&
                std::abs(dy) < 1e-6f)
            {
                continue;
            }

            const float angle =
                std::atan2(
                    dy,
                    dx);

            const float c =
                std::cos(angle);

            const float ss =
                std::sin(angle);

            float minU =
                std::numeric_limits<float>::max();

            float maxU =
                -std::numeric_limits<float>::max();

            float minV =
                std::numeric_limits<float>::max();

            float maxV =
                -std::numeric_limits<float>::max();

            for (const auto& p : hull)
            {
                const float u =
                    p.x * c
                    + p.y * ss;

                const float v =
                    -p.x * ss
                    + p.y * c;

                minU =
                    std::min(
                        minU,
                        u);

                maxU =
                    std::max(
                        maxU,
                        u);

                minV =
                    std::min(
                        minV,
                        v);

                maxV =
                    std::max(
                        maxV,
                        v);
            }

            const float width =
                maxU - minU;

            const float height =
                maxV - minV;

            const float area =
                width * height;

            if (area < bestArea)
            {
                bestArea = area;
                bestAngle = angle;
                bestMinU = minU;
                bestMaxU = maxU;
                bestMinV = minV;
                bestMaxV = maxV;
            }
        }

        if (!std::isfinite(bestArea))
            return result;

        float rectWidth =
            std::max(
                1.0f,
                bestMaxU - bestMinU);

        float rectHeight =
            std::max(
                1.0f,
                bestMaxV - bestMinV);

        const float centerU =
            (bestMinU + bestMaxU)
            * 0.5f;

        const float centerV =
            (bestMinV + bestMaxV)
            * 0.5f;

        const float c =
            std::cos(bestAngle);

        const float ss =
            std::sin(bestAngle);

        const float centerX =
            centerU * c
            - centerV * ss;

        const float centerY =
            centerU * ss
            + centerV * c;

        // Industrial convention used by this app:
        // width is always the long side and angle is its direction in [0, pi).
        float angle =
            bestAngle;

        if (rectHeight > rectWidth)
        {
            std::swap(
                rectWidth,
                rectHeight);

            angle +=
                3.14159265358979323846f
                * 0.5f;
        }

        angle =
            normalizeLongAxisAngle(
                angle);

        result.valid = true;
        result.centerX = centerX;
        result.centerY = centerY;
        result.width = rectWidth;
        result.height = rectHeight;
        result.angle = angle;

        ObbCandidate box{
            centerX,
            centerY,
            rectWidth,
            rectHeight,
            angle,
            1.0f,
            0
        };

        result.corners =
            obbToCorners(box);

        return result;
    }


    static float segIou(
        const SegCandidate& a,
        const SegCandidate& b)
    {
        const float xx1 = std::max(a.x1, b.x1);
        const float yy1 = std::max(a.y1, b.y1);
        const float xx2 = std::min(a.x2, b.x2);
        const float yy2 = std::min(a.y2, b.y2);

        const float w = std::max(0.0f, xx2 - xx1);
        const float h = std::max(0.0f, yy2 - yy1);
        const float inter = w * h;

        const float areaA =
            std::max(0.0f, a.x2 - a.x1)
            * std::max(0.0f, a.y2 - a.y1);

        const float areaB =
            std::max(0.0f, b.x2 - b.x1)
            * std::max(0.0f, b.y2 - b.y1);

        const float denom =
            areaA + areaB - inter;

        return denom > 0.0f
            ? inter / denom
            : 0.0f;
    }

    static std::vector<SegCandidate> classAwareSegNms(
        std::vector<SegCandidate> candidates,
        float threshold)
    {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const SegCandidate& a, const SegCandidate& b)
            {
                return a.score > b.score;
            });

        std::vector<SegCandidate> kept;
        kept.reserve(candidates.size());

        for (const auto& candidate : candidates)
        {
            bool suppress = false;

            for (const auto& previous : kept)
            {
                if (candidate.classId != previous.classId)
                    continue;

                if (segIou(candidate, previous) > threshold)
                {
                    suppress = true;
                    break;
                }
            }

            if (!suppress)
                kept.push_back(candidate);
        }

        return kept;
    }

    static float bilinearSampleMask(
        const std::vector<float>& mask,
        int width,
        int height,
        float x,
        float y)
    {
        if (width <= 0 ||
            height <= 0 ||
            mask.empty())
        {
            return 0.0f;
        }

        x =
            clampf(
                x,
                0.0f,
                static_cast<float>(
                    width - 1));

        y =
            clampf(
                y,
                0.0f,
                static_cast<float>(
                    height - 1));

        const int x0 =
            static_cast<int>(
                std::floor(x));

        const int y0 =
            static_cast<int>(
                std::floor(y));

        const int x1 =
            std::min(
                x0 + 1,
                width - 1);

        const int y1 =
            std::min(
                y0 + 1,
                height - 1);

        const float fx =
            x - x0;

        const float fy =
            y - y0;

        const float v00 =
            mask[
                static_cast<size_t>(y0)
                * width
                + x0];

        const float v10 =
            mask[
                static_cast<size_t>(y0)
                * width
                + x1];

        const float v01 =
            mask[
                static_cast<size_t>(y1)
                * width
                + x0];

        const float v11 =
            mask[
                static_cast<size_t>(y1)
                * width
                + x1];

        const float top =
            v00
            + (v10 - v00) * fx;

        const float bottom =
            v01
            + (v11 - v01) * fx;

        return top
            + (bottom - top) * fy;
    }

    static float probabilityToLogit(
        float probability)
    {
        const float p =
            clampf(
                probability,
                0.001f,
                0.999f);

        return std::log(
            p / (1.0f - p));
    }

    struct Detector
    {
        TrtUniquePtr<nvinfer1::IRuntime> runtime;
        TrtUniquePtr<nvinfer1::ICudaEngine> engine;
        TrtUniquePtr<nvinfer1::IExecutionContext> context;

        std::string inputName;

        // Primary prediction output. Detect/OBB engines only have this output.
        // YOLO26 Seg adds one 4D prototype output.
        std::string outputName;
        std::string protoName;

        nvinfer1::DataType inputType{};
        nvinfer1::DataType outputType{};
        nvinfer1::DataType protoType{};

        nvinfer1::Dims inputDims{};
        nvinfer1::Dims outputDims{};
        nvinfer1::Dims protoDims{};

        bool hasProto = false;

        int inputW = 0;
        int inputH = 0;

        DeviceBuffer inputDevice;
        DeviceBuffer outputDevice;
        DeviceBuffer protoDevice;

        StreamHolder stream;
        std::mutex mutex;

        std::wstring modelInfo;

        Detector(
            const std::wstring& enginePath,
            int dynamicW,
            int dynamicH)
        {
            initLibNvInferPlugins(
                &gLogger,
                "");

            auto bytes =
                readAllBytes(enginePath);

            runtime.reset(
                nvinfer1::createInferRuntime(
                    gLogger));

            if (!runtime)
            {
                throw std::runtime_error(
                    "createInferRuntime failed.");
            }

            engine.reset(
                runtime->deserializeCudaEngine(
                    bytes.data(),
                    bytes.size()));

            if (!engine)
            {
                throw std::runtime_error(
                    "deserializeCudaEngine failed. Verify TensorRT version, "
                    "GPU compatibility and plugins.");
            }

            context.reset(
                engine->createExecutionContext());

            if (!context)
            {
                throw std::runtime_error(
                    "createExecutionContext failed.");
            }

            int inputCount = 0;
            std::vector<std::string> outputNames;

            for (int i = 0;
                 i < engine->getNbIOTensors();
                 ++i)
            {
                const char* name =
                    engine->getIOTensorName(i);

                if (!name)
                {
                    throw std::runtime_error(
                        "getIOTensorName returned null.");
                }

                auto mode =
                    engine->getTensorIOMode(name);

                if (mode ==
                    nvinfer1::TensorIOMode::kINPUT)
                {
                    ++inputCount;
                    inputName = name;
                }
                else if (mode ==
                         nvinfer1::TensorIOMode::kOUTPUT)
                {
                    outputNames.emplace_back(name);
                }
            }

            if (inputCount != 1)
            {
                throw std::runtime_error(
                    "This deployment requires exactly one image input tensor.");
            }

            if (outputNames.empty() ||
                outputNames.size() > 2)
            {
                throw std::runtime_error(
                    "This deployment supports one prediction output "
                    "(Detect/OBB) or prediction + proto outputs (YOLO26 Seg).");
            }

            inputDims =
                engine->getTensorShape(
                    inputName.c_str());

            inputType =
                engine->getTensorDataType(
                    inputName.c_str());

            if (inputDims.nbDims != 4)
            {
                throw std::runtime_error(
                    "Expected a 4D NCHW input tensor.");
            }

            if (inputDims.d[1] != 3 &&
                inputDims.d[1] != -1)
            {
                throw std::runtime_error(
                    "Expected 3 input channels.");
            }

            const bool dynamic =
                inputDims.d[0] < 0 ||
                inputDims.d[2] < 0 ||
                inputDims.d[3] < 0;

            inputH =
                inputDims.d[2] > 0
                    ? inputDims.d[2]
                    : dynamicH;

            inputW =
                inputDims.d[3] > 0
                    ? inputDims.d[3]
                    : dynamicW;

            if (inputH <= 0)
                inputH = 640;

            if (inputW <= 0)
                inputW = 640;

            if (dynamic)
            {
                nvinfer1::Dims4 dims{
                    1,
                    3,
                    inputH,
                    inputW
                };

                if (!context->setInputShape(
                        inputName.c_str(),
                        dims))
                {
                    throw std::runtime_error(
                        "setInputShape failed for dynamic engine.");
                }
            }

            const auto resolvedInput =
                context->getTensorShape(
                    inputName.c_str());

            if (resolvedInput.nbDims != 4)
            {
                throw std::runtime_error(
                    "Resolved input tensor is not NCHW.");
            }

            if (resolvedInput.d[0] != 1)
            {
                throw std::runtime_error(
                    "This deployment supports batch=1 only.");
            }

            // Identify outputs by rank instead of relying on exporter tensor names.
            // Detect/OBB:
            //   prediction -> 3D
            // YOLO26 Seg:
            //   prediction -> 3D
            //   proto      -> 4D
            for (const auto& name : outputNames)
            {
                const auto dims =
                    context->getTensorShape(
                        name.c_str());

                if (dims.nbDims == 3)
                {
                    if (!outputName.empty())
                    {
                        throw std::runtime_error(
                            "More than one 3D prediction output was found.");
                    }

                    outputName = name;
                    outputDims = dims;
                    outputType =
                        engine->getTensorDataType(
                            name.c_str());
                }
                else if (dims.nbDims == 4)
                {
                    if (!protoName.empty())
                    {
                        throw std::runtime_error(
                            "More than one 4D prototype output was found.");
                    }

                    protoName = name;
                    protoDims = dims;
                    protoType =
                        engine->getTensorDataType(
                            name.c_str());
                }
                else
                {
                    std::ostringstream oss;
                    oss
                        << "Unsupported output tensor rank "
                        << dims.nbDims
                        << " for tensor "
                        << name
                        << ". Expected a 3D prediction output "
                        << "and optional 4D segmentation proto output.";

                    throw std::runtime_error(
                        oss.str());
                }
            }

            if (outputName.empty())
            {
                throw std::runtime_error(
                    "No 3D prediction output tensor was found.");
            }

            hasProto =
                !protoName.empty();

            if (outputNames.size() == 2 &&
                !hasProto)
            {
                throw std::runtime_error(
                    "Two outputs were found, but no 4D segmentation proto "
                    "tensor could be identified.");
            }

            if (inputType !=
                    nvinfer1::DataType::kFLOAT &&
                inputType !=
                    nvinfer1::DataType::kHALF)
            {
                throw std::runtime_error(
                    "Only FP32/FP16 input tensors are supported.");
            }

            if (outputType !=
                    nvinfer1::DataType::kFLOAT &&
                outputType !=
                    nvinfer1::DataType::kHALF)
            {
                throw std::runtime_error(
                    "Only FP32/FP16 prediction output tensors are supported.");
            }

            if (hasProto &&
                protoType !=
                    nvinfer1::DataType::kFLOAT &&
                protoType !=
                    nvinfer1::DataType::kHALF)
            {
                throw std::runtime_error(
                    "Only FP32/FP16 segmentation proto tensors are supported.");
            }

            const int64_t inputElements =
                volume(resolvedInput);

            const int64_t outputElements =
                volume(outputDims);

            inputDevice.bytes =
                static_cast<size_t>(
                    inputElements)
                * dataTypeSize(inputType);

            outputDevice.bytes =
                static_cast<size_t>(
                    outputElements)
                * dataTypeSize(outputType);

            checkCuda(
                cudaMalloc(
                    &inputDevice.ptr,
                    inputDevice.bytes),
                "cudaMalloc(input)");

            checkCuda(
                cudaMalloc(
                    &outputDevice.ptr,
                    outputDevice.bytes),
                "cudaMalloc(prediction)");

            if (hasProto)
            {
                const int64_t protoElements =
                    volume(protoDims);

                protoDevice.bytes =
                    static_cast<size_t>(
                        protoElements)
                    * dataTypeSize(protoType);

                checkCuda(
                    cudaMalloc(
                        &protoDevice.ptr,
                        protoDevice.bytes),
                    "cudaMalloc(proto)");
            }

            checkCuda(
                cudaStreamCreate(
                    &stream.stream),
                "cudaStreamCreate");

            if (!context->setTensorAddress(
                    inputName.c_str(),
                    inputDevice.ptr))
            {
                throw std::runtime_error(
                    "setTensorAddress(input) failed.");
            }

            if (!context->setTensorAddress(
                    outputName.c_str(),
                    outputDevice.ptr))
            {
                throw std::runtime_error(
                    "setTensorAddress(prediction) failed.");
            }

            if (hasProto &&
                !context->setTensorAddress(
                    protoName.c_str(),
                    protoDevice.ptr))
            {
                throw std::runtime_error(
                    "setTensorAddress(proto) failed.");
            }

            std::ostringstream info;

            info
                << "Input: "
                << inputName
                << " "
                << dimsToString(resolvedInput)
                << " "
                << dataTypeName(inputType)
                << "\nPrediction: "
                << outputName
                << " "
                << dimsToString(outputDims)
                << " "
                << dataTypeName(outputType);

            if (hasProto)
            {
                info
                    << "\nProto: "
                    << protoName
                    << " "
                    << dimsToString(protoDims)
                    << " "
                    << dataTypeName(protoType);
            }

            info
                << "\nPreprocess: LetterBox + RGB + /255 + NCHW"
                << "\nPostprocess: Detect / OBB / YOLO26-Seg selected by task";

            modelInfo =
                widen(info.str());
        }

        std::vector<float> copyTensorToFloat(
            const DeviceBuffer& device,
            const nvinfer1::Dims& dims,
            nvinfer1::DataType type,
            const char* label)
        {
            const int64_t elements =
                volume(dims);

            std::vector<float> out(
                static_cast<size_t>(
                    elements));

            if (type ==
                nvinfer1::DataType::kFLOAT)
            {
                checkCuda(
                    cudaMemcpyAsync(
                        out.data(),
                        device.ptr,
                        device.bytes,
                        cudaMemcpyDeviceToHost,
                        stream.stream),
                    label);

                checkCuda(
                    cudaStreamSynchronize(
                        stream.stream),
                    "cudaStreamSynchronize");
            }
            else if (type ==
                     nvinfer1::DataType::kHALF)
            {
                std::vector<__half> temp(
                    static_cast<size_t>(
                        elements));

                checkCuda(
                    cudaMemcpyAsync(
                        temp.data(),
                        device.ptr,
                        device.bytes,
                        cudaMemcpyDeviceToHost,
                        stream.stream),
                    label);

                checkCuda(
                    cudaStreamSynchronize(
                        stream.stream),
                    "cudaStreamSynchronize");

                for (size_t i = 0;
                     i < temp.size();
                     ++i)
                {
                    out[i] =
                        __half2float(
                            temp[i]);
                }
            }
            else
            {
                throw std::runtime_error(
                    "Unsupported output data type.");
            }

            return out;
        }

        std::vector<float> copyOutputToFloat()
        {
            return copyTensorToFloat(
                outputDevice,
                outputDims,
                outputType,
                "cudaMemcpyAsync(prediction)");
        }

        std::vector<float> copyProtoToFloat()
        {
            if (!hasProto)
            {
                throw std::runtime_error(
                    "This engine does not contain a segmentation proto output.");
            }

            return copyTensorToFloat(
                protoDevice,
                protoDims,
                protoType,
                "cudaMemcpyAsync(proto)");
        }

        std::vector<Candidate> decodeOutput(
            const std::vector<float>& output,
            float confidence,
            const LetterboxInfo& letterbox,
            int originalW,
            int originalH)
        {
            if (outputDims.nbDims != 3)
                throw std::runtime_error(
                    "Expected YOLOv8 raw output with 3 dimensions, e.g. [1,84,8400].");

            if (outputDims.d[0] != 1)
                throw std::runtime_error("Only batch=1 output is supported.");

            const int d1 = outputDims.d[1];
            const int d2 = outputDims.d[2];

            bool channelsFirst = false;
            int channels = 0;
            int candidates = 0;

            // Raw YOLO Detect usually has a small channel dimension and a large candidate dimension.
            if (d1 <= 512 && d2 > d1)
            {
                channelsFirst = true;
                channels = d1;
                candidates = d2;
            }
            else if (d2 <= 512 && d1 > d2)
            {
                channelsFirst = false;
                channels = d2;
                candidates = d1;
            }
            else
            {
                throw std::runtime_error(
                    "Cannot identify YOLO output layout. Expected [1,C,N] or [1,N,C].");
            }

            const int classCount = channels - 4;
            if (classCount <= 0)
                throw std::runtime_error("Output channel count is too small for YOLOv8 Detect.");

            auto valueAt = [&](int candidate, int channel) -> float
            {
                if (channelsFirst)
                    return output[static_cast<size_t>(channel) * candidates + candidate];
                return output[static_cast<size_t>(candidate) * channels + channel];
            };

            std::vector<Candidate> result;
            result.reserve(std::min(candidates, 1000));

            for (int i = 0; i < candidates; ++i)
            {
                int bestClass = 0;
                float bestScore = valueAt(i, 4);

                for (int c = 1; c < classCount; ++c)
                {
                    float s = valueAt(i, 4 + c);
                    if (s > bestScore)
                    {
                        bestScore = s;
                        bestClass = c;
                    }
                }

                if (bestScore < confidence)
                    continue;

                const float cx = valueAt(i, 0);
                const float cy = valueAt(i, 1);
                const float w = valueAt(i, 2);
                const float h = valueAt(i, 3);

                float x1 = (cx - w * 0.5f - letterbox.left) / letterbox.scale;
                float y1 = (cy - h * 0.5f - letterbox.top) / letterbox.scale;
                float x2 = (cx + w * 0.5f - letterbox.left) / letterbox.scale;
                float y2 = (cy + h * 0.5f - letterbox.top) / letterbox.scale;

                x1 = clampf(x1, 0.0f, static_cast<float>(originalW));
                y1 = clampf(y1, 0.0f, static_cast<float>(originalH));
                x2 = clampf(x2, 0.0f, static_cast<float>(originalW));
                y2 = clampf(y2, 0.0f, static_cast<float>(originalH));

                if (x2 <= x1 || y2 <= y1)
                    continue;

                result.push_back(Candidate{ x1, y1, x2, y2, bestScore, bestClass });
            }

            return result;
        }


        std::vector<ObbCandidate> decodeObbOutput(
            const std::vector<float>& output,
            float confidence,
            const LetterboxInfo& letterbox,
            int originalW,
            int originalH,
            int expectedClassCount)
        {
            if (outputDims.nbDims != 3)
            {
                throw std::runtime_error(
                    "Expected YOLO OBB raw output with 3 dimensions, "
                    "e.g. [1,85,8400] for 80 classes.");
            }

            if (outputDims.d[0] != 1)
                throw std::runtime_error("Only batch=1 OBB output is supported.");

            const int d1 = outputDims.d[1];
            const int d2 = outputDims.d[2];

            bool channelsFirst = false;
            int channels = 0;
            int candidates = 0;

            if (d1 <= 512 && d2 > d1)
            {
                channelsFirst = true;
                channels = d1;
                candidates = d2;
            }
            else if (d2 <= 512 && d1 > d2)
            {
                channelsFirst = false;
                channels = d2;
                candidates = d1;
            }
            else
            {
                throw std::runtime_error(
                    "Cannot identify YOLO OBB output layout. "
                    "Expected [1,C,N] or [1,N,C].");
            }

            // Ultralytics non-end-to-end OBB raw output:
            // [x, y, w, h, class_probs..., angle]
            const int classCount = channels - 5;

            if (classCount <= 0)
            {
                throw std::runtime_error(
                    "Output channel count is too small for YOLO OBB. "
                    "Expected channels = 5 + class_count.");
            }

            if (expectedClassCount > 0 &&
                classCount != expectedClassCount)
            {
                std::ostringstream oss;
                oss
                    << "OBB class count mismatch. Engine output implies "
                    << classCount
                    << " classes, but the application label file contains "
                    << expectedClassCount
                    << ". Replace coco.names with the OBB model class names "
                    << "in exactly the training order.";

                throw std::runtime_error(oss.str());
            }

            const int angleChannel =
                4 + classCount;

            auto valueAt =
                [&](int candidate, int channel) -> float
                {
                    if (channelsFirst)
                    {
                        return output[
                            static_cast<size_t>(channel)
                            * candidates
                            + candidate];
                    }

                    return output[
                        static_cast<size_t>(candidate)
                        * channels
                        + channel];
                };

            std::vector<ObbCandidate> result;
            result.reserve(std::min(candidates, 1000));

            for (int i = 0; i < candidates; ++i)
            {
                int bestClass = 0;
                float bestScore =
                    valueAt(i, 4);

                for (int c = 1; c < classCount; ++c)
                {
                    const float score =
                        valueAt(i, 4 + c);

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                if (bestScore < confidence)
                    continue;

                const float rawCx =
                    valueAt(i, 0);

                const float rawCy =
                    valueAt(i, 1);

                const float rawW =
                    valueAt(i, 2);

                const float rawH =
                    valueAt(i, 3);

                const float angle =
                    valueAt(i, angleChannel);

                // Undo LetterBox. Rotation itself is unchanged by the
                // uniform scale + translation preprocessing.
                const float cx =
                    (rawCx - letterbox.left)
                    / letterbox.scale;

                const float cy =
                    (rawCy - letterbox.top)
                    / letterbox.scale;

                const float w =
                    rawW / letterbox.scale;

                const float h =
                    rawH / letterbox.scale;

                if (!std::isfinite(cx) ||
                    !std::isfinite(cy) ||
                    !std::isfinite(w) ||
                    !std::isfinite(h) ||
                    !std::isfinite(angle) ||
                    w <= 0.0f ||
                    h <= 0.0f)
                {
                    continue;
                }

                // Keep boxes whose centers are at least close to the image.
                // Corners are deliberately not individually clamped because
                // doing so would geometrically distort a rotated rectangle.
                if (cx < -w ||
                    cy < -h ||
                    cx > static_cast<float>(originalW) + w ||
                    cy > static_cast<float>(originalH) + h)
                {
                    continue;
                }

                result.push_back(
                    ObbCandidate{
                        cx,
                        cy,
                        w,
                        h,
                        angle,
                        bestScore,
                        bestClass
                    });
            }

            return result;
        }


        std::vector<SegCandidate> decodeSegCandidates(
            const std::vector<float>& prediction,
            float confidence,
            float nmsThreshold,
            int expectedClassCount)
        {
            if (!hasProto)
            {
                throw std::runtime_error(
                    "Segmentation requires a second 4D proto output tensor.");
            }

            if (protoDims.nbDims != 4 ||
                protoDims.d[0] != 1)
            {
                throw std::runtime_error(
                    "Expected proto output [1,nm,H,W].");
            }

            const int maskChannels =
                protoDims.d[1];

            if (maskChannels <= 0)
            {
                throw std::runtime_error(
                    "Invalid segmentation mask coefficient count.");
            }

            if (outputDims.nbDims != 3 ||
                outputDims.d[0] != 1)
            {
                throw std::runtime_error(
                    "Expected segmentation prediction output with rank 3.");
            }

            const int d1 = outputDims.d[1];
            const int d2 = outputDims.d[2];

            const int rawChannels =
                expectedClassCount > 0
                    ? 4 + expectedClassCount + maskChannels
                    : -1;

            const int endToEndChannels =
                6 + maskChannels;

            enum class Layout
            {
                RawChannelsFirst,
                RawChannelsLast,
                EndToEndChannelsFirst,
                EndToEndChannelsLast
            };

            Layout layout{};
            int featureChannels = 0;
            int candidateCount = 0;

            if (rawChannels > 0 &&
                d1 == rawChannels)
            {
                layout = Layout::RawChannelsFirst;
                featureChannels = d1;
                candidateCount = d2;
            }
            else if (rawChannels > 0 &&
                     d2 == rawChannels)
            {
                layout = Layout::RawChannelsLast;
                featureChannels = d2;
                candidateCount = d1;
            }
            else if (d1 == endToEndChannels)
            {
                layout = Layout::EndToEndChannelsFirst;
                featureChannels = d1;
                candidateCount = d2;
            }
            else if (d2 == endToEndChannels)
            {
                layout = Layout::EndToEndChannelsLast;
                featureChannels = d2;
                candidateCount = d1;
            }
            else
            {
                std::ostringstream oss;
                oss
                    << "Segmentation prediction shape is unsupported. Prediction="
                    << dimsToString(outputDims)
                    << ", proto="
                    << dimsToString(protoDims)
                    << ", expected classes="
                    << expectedClassCount
                    << ". Recommended YOLO26 export: end2end=False, nms=False.";

                throw std::runtime_error(
                    oss.str());
            }

            auto valueAt =
                [&](int candidate,
                    int channel) -> float
                {
                    const bool channelsFirst =
                        layout == Layout::RawChannelsFirst ||
                        layout == Layout::EndToEndChannelsFirst;

                    if (channelsFirst)
                    {
                        return prediction[
                            static_cast<size_t>(channel)
                            * candidateCount
                            + candidate];
                    }

                    return prediction[
                        static_cast<size_t>(candidate)
                        * featureChannels
                        + channel];
                };

            const bool raw =
                layout == Layout::RawChannelsFirst ||
                layout == Layout::RawChannelsLast;

            std::vector<SegCandidate> result;
            result.reserve(
                std::min(
                    candidateCount,
                    1000));

            if (raw)
            {
                if (expectedClassCount <= 0)
                {
                    throw std::runtime_error(
                        "Raw segmentation output requires a valid class-name count. "
                        "Replace coco.names with the model's class names.");
                }

                const int coefficientStart =
                    4 + expectedClassCount;

                for (int i = 0;
                     i < candidateCount;
                     ++i)
                {
                    int bestClass = 0;
                    float bestScore =
                        valueAt(i, 4);

                    for (int c = 1;
                         c < expectedClassCount;
                         ++c)
                    {
                        const float score =
                            valueAt(
                                i,
                                4 + c);

                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestClass = c;
                        }
                    }

                    if (!std::isfinite(bestScore) ||
                        bestScore < confidence)
                    {
                        continue;
                    }

                    const float cx =
                        valueAt(i, 0);

                    const float cy =
                        valueAt(i, 1);

                    const float w =
                        valueAt(i, 2);

                    const float h =
                        valueAt(i, 3);

                    if (!std::isfinite(cx) ||
                        !std::isfinite(cy) ||
                        !std::isfinite(w) ||
                        !std::isfinite(h) ||
                        w <= 0.0f ||
                        h <= 0.0f)
                    {
                        continue;
                    }

                    SegCandidate candidate;
                    candidate.x1 = cx - w * 0.5f;
                    candidate.y1 = cy - h * 0.5f;
                    candidate.x2 = cx + w * 0.5f;
                    candidate.y2 = cy + h * 0.5f;
                    candidate.score = bestScore;
                    candidate.classId = bestClass;

                    candidate.coefficients.resize(
                        static_cast<size_t>(
                            maskChannels));

                    for (int k = 0;
                         k < maskChannels;
                         ++k)
                    {
                        candidate.coefficients[
                            static_cast<size_t>(k)] =
                            valueAt(
                                i,
                                coefficientStart + k);
                    }

                    result.push_back(
                        std::move(candidate));
                }

                return classAwareSegNms(
                    std::move(result),
                    nmsThreshold);
            }

            // YOLO26 end-to-end path:
            // [x1,y1,x2,y2,confidence,class_id,mask_coefficients...]
            for (int i = 0;
                 i < candidateCount;
                 ++i)
            {
                const float score =
                    valueAt(i, 4);

                if (!std::isfinite(score) ||
                    score < confidence)
                {
                    continue;
                }

                const float classValue =
                    valueAt(i, 5);

                if (!std::isfinite(classValue))
                    continue;

                const int classId =
                    static_cast<int>(
                        std::lround(
                            classValue));

                if (expectedClassCount > 0 &&
                    (classId < 0 ||
                     classId >= expectedClassCount))
                {
                    std::ostringstream oss;
                    oss
                        << "Segmentation class id "
                        << classId
                        << " is outside the label file range [0,"
                        << (expectedClassCount - 1)
                        << "]. Replace coco.names with the model's "
                        << "actual class names in training order.";

                    throw std::runtime_error(
                        oss.str());
                }

                SegCandidate candidate;
                candidate.x1 = valueAt(i, 0);
                candidate.y1 = valueAt(i, 1);
                candidate.x2 = valueAt(i, 2);
                candidate.y2 = valueAt(i, 3);
                candidate.score = score;
                candidate.classId = classId;

                if (!std::isfinite(candidate.x1) ||
                    !std::isfinite(candidate.y1) ||
                    !std::isfinite(candidate.x2) ||
                    !std::isfinite(candidate.y2))
                {
                    continue;
                }

                candidate.coefficients.resize(
                    static_cast<size_t>(
                        maskChannels));

                for (int k = 0;
                     k < maskChannels;
                     ++k)
                {
                    candidate.coefficients[
                        static_cast<size_t>(k)] =
                        valueAt(
                            i,
                            6 + k);
                }

                result.push_back(
                    std::move(candidate));
            }

            std::sort(
                result.begin(),
                result.end(),
                [](const SegCandidate& a,
                   const SegCandidate& b)
                {
                    return a.score > b.score;
                });

            return result;
        }

        std::vector<YoloSegDetection> buildSegmentationResults(
            const std::vector<SegCandidate>& candidates,
            const std::vector<float>& proto,
            const LetterboxInfo& letterbox,
            int originalW,
            int originalH,
            float maskThreshold,
            uint16_t* instanceMask,
            int maskStride,
            int resultCapacity)
        {
            if (!hasProto)
            {
                throw std::runtime_error(
                    "Segmentation proto output is missing.");
            }

            if (!instanceMask ||
                maskStride < originalW)
            {
                throw std::runtime_error(
                    "Instance mask buffer/stride is invalid.");
            }

            const int maskChannels =
                protoDims.d[1];

            const int protoH =
                protoDims.d[2];

            const int protoW =
                protoDims.d[3];

            if (maskChannels <= 0 ||
                protoH <= 0 ||
                protoW <= 0)
            {
                throw std::runtime_error(
                    "Invalid segmentation proto dimensions.");
            }

            const size_t protoPlane =
                static_cast<size_t>(
                    protoH)
                * protoW;

            const size_t expectedProto =
                static_cast<size_t>(
                    maskChannels)
                * protoPlane;

            if (proto.size() !=
                expectedProto)
            {
                throw std::runtime_error(
                    "Segmentation proto element count mismatch.");
            }

            for (int y = 0;
                 y < originalH;
                 ++y)
            {
                std::fill_n(
                    instanceMask
                    + static_cast<size_t>(y)
                    * maskStride,
                    originalW,
                    static_cast<uint16_t>(0));
            }

            const float logitThreshold =
                probabilityToLogit(
                    maskThreshold);

            const int maxResults =
                std::max(
                    0,
                    std::min(
                        resultCapacity,
                        static_cast<int>(
                            candidates.size())));

            std::vector<YoloSegDetection> results;
            results.reserve(
                static_cast<size_t>(
                    maxResults));

            for (int candidateIndex = 0;
                 candidateIndex < static_cast<int>(candidates.size()) &&
                 static_cast<int>(results.size()) < maxResults;
                 ++candidateIndex)
            {
                const SegCandidate& candidate =
                    candidates[
                        static_cast<size_t>(
                            candidateIndex)];

                if (candidate.coefficients.size() !=
                    static_cast<size_t>(
                        maskChannels))
                {
                    continue;
                }

                std::vector<float> maskLogits(
                    protoPlane,
                    0.0f);

                // Official Ultralytics mask reconstruction is the matrix
                // product: mask_coefficients @ prototype_masks.
                for (int k = 0;
                     k < maskChannels;
                     ++k)
                {
                    const float coefficient =
                        candidate.coefficients[
                            static_cast<size_t>(k)];

                    const float* protoChannel =
                        proto.data()
                        + static_cast<size_t>(k)
                        * protoPlane;

                    for (size_t p = 0;
                         p < protoPlane;
                         ++p)
                    {
                        maskLogits[p] +=
                            coefficient
                            * protoChannel[p];
                    }
                }

                float boxX1 =
                    (candidate.x1 -
                     letterbox.left)
                    / letterbox.scale;

                float boxY1 =
                    (candidate.y1 -
                     letterbox.top)
                    / letterbox.scale;

                float boxX2 =
                    (candidate.x2 -
                     letterbox.left)
                    / letterbox.scale;

                float boxY2 =
                    (candidate.y2 -
                     letterbox.top)
                    / letterbox.scale;

                if (boxX2 < boxX1)
                    std::swap(boxX1, boxX2);

                if (boxY2 < boxY1)
                    std::swap(boxY1, boxY2);

                boxX1 =
                    clampf(
                        boxX1,
                        0.0f,
                        static_cast<float>(
                            originalW));

                boxY1 =
                    clampf(
                        boxY1,
                        0.0f,
                        static_cast<float>(
                            originalH));

                boxX2 =
                    clampf(
                        boxX2,
                        0.0f,
                        static_cast<float>(
                            originalW));

                boxY2 =
                    clampf(
                        boxY2,
                        0.0f,
                        static_cast<float>(
                            originalH));

                const int startX =
                    std::max(
                        0,
                        static_cast<int>(
                            std::floor(boxX1)));

                const int startY =
                    std::max(
                        0,
                        static_cast<int>(
                            std::floor(boxY1)));

                const int endX =
                    std::min(
                        originalW,
                        static_cast<int>(
                            std::ceil(boxX2)));

                const int endY =
                    std::min(
                        originalH,
                        static_cast<int>(
                            std::ceil(boxY2)));

                if (endX <= startX ||
                    endY <= startY)
                {
                    continue;
                }

                std::vector<int> positiveOffsets;
                positiveOffsets.reserve(
                    static_cast<size_t>(
                        (endX - startX)
                        * (endY - startY)
                        / 4));

                std::vector<PointF> boundaryPoints;
                boundaryPoints.reserve(
                    static_cast<size_t>(
                        (endY - startY)
                        * 2));

                int maskMinX =
                    originalW;

                int maskMinY =
                    originalH;

                int maskMaxX =
                    -1;

                int maskMaxY =
                    -1;

                for (int y = startY;
                     y < endY;
                     ++y)
                {
                    int firstX = -1;
                    int lastX = -1;

                    for (int x = startX;
                         x < endX;
                         ++x)
                    {
                        // Map the original-image pixel center through the
                        // exact LetterBox transform, then through a bilinear
                        // resize from proto resolution to network resolution.
                        const float networkX =
                            (static_cast<float>(x) + 0.5f)
                            * letterbox.scale
                            + letterbox.left;

                        const float networkY =
                            (static_cast<float>(y) + 0.5f)
                            * letterbox.scale
                            + letterbox.top;

                        const float protoX =
                            networkX
                            * static_cast<float>(protoW)
                            / static_cast<float>(inputW)
                            - 0.5f;

                        const float protoY =
                            networkY
                            * static_cast<float>(protoH)
                            / static_cast<float>(inputH)
                            - 0.5f;

                        const float logit =
                            bilinearSampleMask(
                                maskLogits,
                                protoW,
                                protoH,
                                protoX,
                                protoY);

                        if (logit <= logitThreshold)
                            continue;

                        positiveOffsets.push_back(
                            y * maskStride + x);

                        maskMinX =
                            std::min(
                                maskMinX,
                                x);

                        maskMinY =
                            std::min(
                                maskMinY,
                                y);

                        maskMaxX =
                            std::max(
                                maskMaxX,
                                x);

                        maskMaxY =
                            std::max(
                                maskMaxY,
                                y);

                        if (firstX < 0)
                            firstX = x;

                        lastX = x;
                    }

                    if (firstX >= 0)
                    {
                        // Represent the occupied pixel cells by their outer
                        // corners rather than only pixel centers. This avoids
                        // a systematic ~1 px under-estimation of thin masks.
                        const float left =
                            static_cast<float>(
                                firstX);

                        const float right =
                            static_cast<float>(
                                lastX + 1);

                        const float top =
                            static_cast<float>(
                                y);

                        const float bottom =
                            static_cast<float>(
                                y + 1);

                        boundaryPoints.push_back(
                            PointF{ left, top });

                        boundaryPoints.push_back(
                            PointF{ left, bottom });

                        boundaryPoints.push_back(
                            PointF{ right, top });

                        boundaryPoints.push_back(
                            PointF{ right, bottom });
                    }
                }

                if (positiveOffsets.size() < 3 ||
                    boundaryPoints.empty() ||
                    maskMaxX < maskMinX ||
                    maskMaxY < maskMinY)
                {
                    continue;
                }

                MinAreaRectResult minRect =
                    minimumAreaRect(
                        boundaryPoints);

                if (!minRect.valid)
                    continue;

                const int maskId =
                    static_cast<int>(
                        results.size())
                    + 1;

                // Higher confidence detections are processed first.
                // Keep the first instance ID in overlap pixels for a stable
                // visualization while area/minRect remain per-instance.
                for (int offset :
                     positiveOffsets)
                {
                    uint16_t& pixel =
                        instanceMask[
                            static_cast<size_t>(
                                offset)];

                    if (pixel == 0)
                    {
                        pixel =
                            static_cast<uint16_t>(
                                maskId);
                    }
                }

                const ObbCorners& corners =
                    minRect.corners;

                YoloSegDetection detection{
                    static_cast<float>(maskMinX),
                    static_cast<float>(maskMinY),
                    static_cast<float>(maskMaxX + 1),
                    static_cast<float>(maskMaxY + 1),
                    candidate.score,
                    candidate.classId,
                    static_cast<float>(
                        positiveOffsets.size()),
                    minRect.centerX,
                    minRect.centerY,
                    minRect.width,
                    minRect.height,
                    minRect.angle,
                    corners.p1x,
                    corners.p1y,
                    corners.p2x,
                    corners.p2y,
                    corners.p3x,
                    corners.p3y,
                    corners.p4x,
                    corners.p4y,
                    maskId
                };

                results.push_back(
                    detection);
            }

            return results;
        }

        std::vector<YoloSegDetection> detectSeg(
            const uint8_t* bgra,
            int width,
            int height,
            int stride,
            float confidence,
            float nmsThreshold,
            float maskThreshold,
            int expectedClassCount,
            uint16_t* instanceMask,
            int maskStride,
            int resultCapacity,
            float& inferenceMs)
        {
            std::lock_guard<std::mutex> lock(
                mutex);

            if (!hasProto)
            {
                throw std::runtime_error(
                    "This Engine has no segmentation proto output. "
                    "Use a YOLO26-seg ONNX/Engine with prediction + proto outputs.");
            }

            std::vector<float> inputFloat;

            LetterboxInfo letterbox =
                preprocessBgra(
                    bgra,
                    width,
                    height,
                    stride,
                    inputW,
                    inputH,
                    inputFloat);

            if (inputType ==
                nvinfer1::DataType::kFLOAT)
            {
                if (inputFloat.size()
                    * sizeof(float)
                    != inputDevice.bytes)
                {
                    throw std::runtime_error(
                        "Input tensor byte size mismatch.");
                }

                checkCuda(
                    cudaMemcpyAsync(
                        inputDevice.ptr,
                        inputFloat.data(),
                        inputDevice.bytes,
                        cudaMemcpyHostToDevice,
                        stream.stream),
                    "cudaMemcpyAsync(input FP32)");
            }
            else
            {
                std::vector<__half> inputHalf(
                    inputFloat.size());

                for (size_t i = 0;
                     i < inputFloat.size();
                     ++i)
                {
                    inputHalf[i] =
                        __float2half(
                            inputFloat[i]);
                }

                if (inputHalf.size()
                    * sizeof(__half)
                    != inputDevice.bytes)
                {
                    throw std::runtime_error(
                        "FP16 input tensor byte size mismatch.");
                }

                checkCuda(
                    cudaMemcpyAsync(
                        inputDevice.ptr,
                        inputHalf.data(),
                        inputDevice.bytes,
                        cudaMemcpyHostToDevice,
                        stream.stream),
                    "cudaMemcpyAsync(input FP16)");
            }

            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;

            checkCuda(
                cudaEventCreate(&start),
                "cudaEventCreate(start)");

            checkCuda(
                cudaEventCreate(&stop),
                "cudaEventCreate(stop)");

            try
            {
                checkCuda(
                    cudaEventRecord(
                        start,
                        stream.stream),
                    "cudaEventRecord(start)");

                if (!context->enqueueV3(
                        stream.stream))
                {
                    throw std::runtime_error(
                        "TensorRT enqueueV3 failed.");
                }

                checkCuda(
                    cudaEventRecord(
                        stop,
                        stream.stream),
                    "cudaEventRecord(stop)");

                checkCuda(
                    cudaEventSynchronize(
                        stop),
                    "cudaEventSynchronize(stop)");

                checkCuda(
                    cudaEventElapsedTime(
                        &inferenceMs,
                        start,
                        stop),
                    "cudaEventElapsedTime");

                std::vector<float> prediction =
                    copyOutputToFloat();

                std::vector<float> proto =
                    copyProtoToFloat();

                auto candidates =
                    decodeSegCandidates(
                        prediction,
                        confidence,
                        nmsThreshold,
                        expectedClassCount);

                auto results =
                    buildSegmentationResults(
                        candidates,
                        proto,
                        letterbox,
                        width,
                        height,
                        maskThreshold,
                        instanceMask,
                        maskStride,
                        resultCapacity);

                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                return results;
            }
            catch (...)
            {
                if (start)
                    cudaEventDestroy(start);

                if (stop)
                    cudaEventDestroy(stop);

                throw;
            }
        }

        std::vector<ObbCandidate> detectObb(
            const uint8_t* bgra,
            int width,
            int height,
            int stride,
            float confidence,
            float nmsThreshold,
            int expectedClassCount,
            float& inferenceMs)
        {
            std::lock_guard<std::mutex> lock(mutex);

            std::vector<float> inputFloat;

            LetterboxInfo letterbox =
                preprocessBgra(
                    bgra,
                    width,
                    height,
                    stride,
                    inputW,
                    inputH,
                    inputFloat);

            if (inputType == nvinfer1::DataType::kFLOAT)
            {
                if (inputFloat.size() * sizeof(float) != inputDevice.bytes)
                {
                    throw std::runtime_error(
                        "Input tensor byte size mismatch.");
                }

                checkCuda(
                    cudaMemcpyAsync(
                        inputDevice.ptr,
                        inputFloat.data(),
                        inputDevice.bytes,
                        cudaMemcpyHostToDevice,
                        stream.stream),
                    "cudaMemcpyAsync(input FP32)");
            }
            else
            {
                std::vector<__half> inputHalf(
                    inputFloat.size());

                for (size_t i = 0; i < inputFloat.size(); ++i)
                {
                    inputHalf[i] =
                        __float2half(inputFloat[i]);
                }

                if (inputHalf.size() * sizeof(__half) != inputDevice.bytes)
                {
                    throw std::runtime_error(
                        "FP16 input tensor byte size mismatch.");
                }

                checkCuda(
                    cudaMemcpyAsync(
                        inputDevice.ptr,
                        inputHalf.data(),
                        inputDevice.bytes,
                        cudaMemcpyHostToDevice,
                        stream.stream),
                    "cudaMemcpyAsync(input FP16)");
            }

            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;

            checkCuda(
                cudaEventCreate(&start),
                "cudaEventCreate(start)");

            checkCuda(
                cudaEventCreate(&stop),
                "cudaEventCreate(stop)");

            try
            {
                checkCuda(
                    cudaEventRecord(
                        start,
                        stream.stream),
                    "cudaEventRecord(start)");

                if (!context->enqueueV3(
                        stream.stream))
                {
                    throw std::runtime_error(
                        "TensorRT enqueueV3 failed.");
                }

                checkCuda(
                    cudaEventRecord(
                        stop,
                        stream.stream),
                    "cudaEventRecord(stop)");

                checkCuda(
                    cudaEventSynchronize(stop),
                    "cudaEventSynchronize(stop)");

                checkCuda(
                    cudaEventElapsedTime(
                        &inferenceMs,
                        start,
                        stop),
                    "cudaEventElapsedTime");

                std::vector<float> output =
                    copyOutputToFloat();

                auto decoded =
                    decodeObbOutput(
                        output,
                        confidence,
                        letterbox,
                        width,
                        height,
                        expectedClassCount);

                auto finalBoxes =
                    classAwareRotatedNms(
                        std::move(decoded),
                        nmsThreshold);

                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                return finalBoxes;
            }
            catch (...)
            {
                if (start)
                    cudaEventDestroy(start);

                if (stop)
                    cudaEventDestroy(stop);

                throw;
            }
        }

        std::vector<Candidate> detect(
            const uint8_t* bgra,
            int width,
            int height,
            int stride,
            float confidence,
            float nmsThreshold,
            float& inferenceMs)
        {
            std::lock_guard<std::mutex> lock(mutex);

            std::vector<float> inputFloat;
            LetterboxInfo letterbox =
                preprocessBgra(bgra, width, height, stride, inputW, inputH, inputFloat);

            if (inputType == nvinfer1::DataType::kFLOAT)
            {
                if (inputFloat.size() * sizeof(float) != inputDevice.bytes)
                    throw std::runtime_error("Input tensor byte size mismatch.");

                checkCuda(cudaMemcpyAsync(
                    inputDevice.ptr,
                    inputFloat.data(),
                    inputDevice.bytes,
                    cudaMemcpyHostToDevice,
                    stream.stream), "cudaMemcpyAsync(input FP32)");
            }
            else
            {
                std::vector<__half> inputHalf(inputFloat.size());
                for (size_t i = 0; i < inputFloat.size(); ++i)
                    inputHalf[i] = __float2half(inputFloat[i]);

                if (inputHalf.size() * sizeof(__half) != inputDevice.bytes)
                    throw std::runtime_error("FP16 input tensor byte size mismatch.");

                checkCuda(cudaMemcpyAsync(
                    inputDevice.ptr,
                    inputHalf.data(),
                    inputDevice.bytes,
                    cudaMemcpyHostToDevice,
                    stream.stream), "cudaMemcpyAsync(input FP16)");
            }

            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
            checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

            try
            {
                checkCuda(cudaEventRecord(start, stream.stream), "cudaEventRecord(start)");

                if (!context->enqueueV3(stream.stream))
                    throw std::runtime_error("TensorRT enqueueV3 failed.");

                checkCuda(cudaEventRecord(stop, stream.stream), "cudaEventRecord(stop)");
                checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

                checkCuda(cudaEventElapsedTime(&inferenceMs, start, stop), "cudaEventElapsedTime");

                std::vector<float> output = copyOutputToFloat();
                auto decoded = decodeOutput(output, confidence, letterbox, width, height);
                auto finalBoxes = classAwareNms(std::move(decoded), nmsThreshold);

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return finalBoxes;
            }
            catch (...)
            {
                if (start) cudaEventDestroy(start);
                if (stop) cudaEventDestroy(stop);
                throw;
            }
        }
    };
}

void* __cdecl YoloCreate(
    const wchar_t* enginePath,
    int32_t dynamicInputWidth,
    int32_t dynamicInputHeight,
    wchar_t* errorBuffer,
    int32_t errorCapacity)
{
    try
    {
        if (!enginePath || !*enginePath)
            throw std::runtime_error("Engine path is empty.");

        auto* detector = new Detector(
            enginePath,
            dynamicInputWidth > 0 ? dynamicInputWidth : 640,
            dynamicInputHeight > 0 ? dynamicInputHeight : 640);

        setError(L"", errorBuffer, errorCapacity);
        return detector;
    }
    catch (const std::exception& ex)
    {
        setError(widen(ex.what()), errorBuffer, errorCapacity);
        return nullptr;
    }
    catch (...)
    {
        setError(L"Unknown error while creating detector.", errorBuffer, errorCapacity);
        return nullptr;
    }
}


int32_t __cdecl YoloGetTaskHint(
    void* handle,
    int32_t expectedClassCount)
{
    if (!handle)
        return -1;

    auto* detector =
        static_cast<Detector*>(handle);

    if (detector->hasProto)
    {
        // A 3D prediction output + 4D prototype output is the
        // YOLO instance-segmentation signature supported in Phase 6.
        return 2;
    }

    if (expectedClassCount <= 0)
        return -1;

    const auto& dims =
        detector->outputDims;

    if (dims.nbDims != 3 ||
        dims.d[0] != 1)
    {
        return -1;
    }

    const int d1 =
        dims.d[1];

    const int d2 =
        dims.d[2];

    int channels = 0;

    if (d1 <= 512 &&
        d2 > d1)
    {
        channels = d1;
    }
    else if (d2 <= 512 &&
             d1 > d2)
    {
        channels = d2;
    }
    else
    {
        return -1;
    }

    if (channels ==
        4 + expectedClassCount)
    {
        return 0;
    }

    if (channels ==
        5 + expectedClassCount)
    {
        return 1;
    }

    return -1;
}

int32_t __cdecl YoloGetModelInfo(
    void* handle,
    wchar_t* infoBuffer,
    int32_t infoCapacity)
{
    if (!handle)
        return -1;

    auto* detector = static_cast<Detector*>(handle);
    if (infoBuffer && infoCapacity > 0)
        wcsncpy_s(infoBuffer, static_cast<size_t>(infoCapacity), detector->modelInfo.c_str(), _TRUNCATE);

    return static_cast<int32_t>(detector->modelInfo.size());
}

int32_t __cdecl YoloDetectBgra(
    void* handle,
    const uint8_t* bgra,
    int32_t width,
    int32_t height,
    int32_t stride,
    float confidenceThreshold,
    float nmsThreshold,
    YoloDetection* results,
    int32_t resultCapacity,
    float* inferenceMilliseconds,
    wchar_t* errorBuffer,
    int32_t errorCapacity)
{
    try
    {
        if (!handle)
            throw std::runtime_error("Detector handle is null.");
        if (!bgra)
            throw std::runtime_error("Image data is null.");
        if (!results || resultCapacity <= 0)
            throw std::runtime_error("Detection result buffer is invalid.");

        confidenceThreshold = clampf(confidenceThreshold, 0.0f, 1.0f);
        nmsThreshold = clampf(nmsThreshold, 0.0f, 1.0f);

        auto* detector = static_cast<Detector*>(handle);
        float ms = 0.0f;
        auto detections = detector->detect(
            bgra, width, height, stride,
            confidenceThreshold, nmsThreshold, ms);

        if (inferenceMilliseconds)
            *inferenceMilliseconds = ms;

        int32_t count = static_cast<int32_t>(
            std::min<size_t>(detections.size(), static_cast<size_t>(resultCapacity)));

        for (int32_t i = 0; i < count; ++i)
        {
            const auto& d = detections[static_cast<size_t>(i)];
            results[i] = YoloDetection{
                d.x1, d.y1, d.x2, d.y2, d.score, d.classId
            };
        }

        setError(L"", errorBuffer, errorCapacity);
        return count;
    }
    catch (const std::exception& ex)
    {
        setError(widen(ex.what()), errorBuffer, errorCapacity);
        return -1;
    }
    catch (...)
    {
        setError(L"Unknown inference error.", errorBuffer, errorCapacity);
        return -1;
    }
}


int32_t __cdecl YoloDetectObbBgra(
    void* handle,
    const uint8_t* bgra,
    int32_t width,
    int32_t height,
    int32_t stride,
    float confidenceThreshold,
    float nmsThreshold,
    int32_t expectedClassCount,
    YoloObbDetection* results,
    int32_t resultCapacity,
    float* inferenceMilliseconds,
    wchar_t* errorBuffer,
    int32_t errorCapacity)
{
    try
    {
        if (!handle)
            throw std::runtime_error("Detector handle is null.");

        if (!bgra)
            throw std::runtime_error("Image data is null.");

        if (!results || resultCapacity <= 0)
            throw std::runtime_error("OBB result buffer is invalid.");

        confidenceThreshold =
            clampf(
                confidenceThreshold,
                0.0f,
                1.0f);

        nmsThreshold =
            clampf(
                nmsThreshold,
                0.0f,
                1.0f);

        auto* detector =
            static_cast<Detector*>(handle);

        float ms = 0.0f;

        auto detections =
            detector->detectObb(
                bgra,
                width,
                height,
                stride,
                confidenceThreshold,
                nmsThreshold,
                expectedClassCount,
                ms);

        if (inferenceMilliseconds)
            *inferenceMilliseconds = ms;

        const int32_t count =
            static_cast<int32_t>(
                std::min<size_t>(
                    detections.size(),
                    static_cast<size_t>(
                        resultCapacity)));

        for (int32_t i = 0; i < count; ++i)
        {
            const auto& d =
                detections[
                    static_cast<size_t>(i)];

            const ObbCorners corners =
                obbToCorners(d);

            results[i] =
                YoloObbDetection{
                    d.centerX,
                    d.centerY,
                    d.width,
                    d.height,
                    d.angle,
                    d.score,
                    d.classId,
                    corners.p1x,
                    corners.p1y,
                    corners.p2x,
                    corners.p2y,
                    corners.p3x,
                    corners.p3y,
                    corners.p4x,
                    corners.p4y
                };
        }

        setError(
            L"",
            errorBuffer,
            errorCapacity);

        return count;
    }
    catch (const std::exception& ex)
    {
        setError(
            widen(ex.what()),
            errorBuffer,
            errorCapacity);

        return -1;
    }
    catch (...)
    {
        setError(
            L"Unknown OBB inference error.",
            errorBuffer,
            errorCapacity);

        return -1;
    }
}


int32_t __cdecl YoloDetectSegBgra(
    void* handle,
    const uint8_t* bgra,
    int32_t width,
    int32_t height,
    int32_t stride,
    float confidenceThreshold,
    float nmsThreshold,
    float maskThreshold,
    int32_t expectedClassCount,
    YoloSegDetection* results,
    int32_t resultCapacity,
    uint16_t* instanceMask,
    int32_t maskStride,
    float* inferenceMilliseconds,
    wchar_t* errorBuffer,
    int32_t errorCapacity)
{
    try
    {
        if (!handle)
        {
            throw std::runtime_error(
                "Detector handle is null.");
        }

        if (!bgra)
        {
            throw std::runtime_error(
                "Image data is null.");
        }

        if (!results ||
            resultCapacity <= 0)
        {
            throw std::runtime_error(
                "Segmentation result buffer is invalid.");
        }

        if (!instanceMask ||
            maskStride < width)
        {
            throw std::runtime_error(
                "Segmentation instance mask buffer/stride is invalid.");
        }

        confidenceThreshold =
            clampf(
                confidenceThreshold,
                0.0f,
                1.0f);

        nmsThreshold =
            clampf(
                nmsThreshold,
                0.0f,
                1.0f);

        maskThreshold =
            clampf(
                maskThreshold,
                0.001f,
                0.999f);

        auto* detector =
            static_cast<Detector*>(
                handle);

        float ms = 0.0f;

        auto detections =
            detector->detectSeg(
                bgra,
                width,
                height,
                stride,
                confidenceThreshold,
                nmsThreshold,
                maskThreshold,
                expectedClassCount,
                instanceMask,
                maskStride,
                resultCapacity,
                ms);

        if (inferenceMilliseconds)
        {
            *inferenceMilliseconds =
                ms;
        }

        const int32_t count =
            static_cast<int32_t>(
                std::min<size_t>(
                    detections.size(),
                    static_cast<size_t>(
                        resultCapacity)));

        for (int32_t i = 0;
             i < count;
             ++i)
        {
            results[i] =
                detections[
                    static_cast<size_t>(i)];
        }

        setError(
            L"",
            errorBuffer,
            errorCapacity);

        return count;
    }
    catch (const std::exception& ex)
    {
        setError(
            widen(ex.what()),
            errorBuffer,
            errorCapacity);

        return -1;
    }
    catch (...)
    {
        setError(
            L"Unknown segmentation inference error.",
            errorBuffer,
            errorCapacity);

        return -1;
    }
}

void __cdecl YoloDestroy(void* handle)
{
    delete static_cast<Detector*>(handle);
}
