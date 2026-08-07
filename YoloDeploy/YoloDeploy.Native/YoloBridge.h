#pragma once
#include <cstdint>

#ifdef YOLODEPLOYNATIVE_EXPORTS
#define YOLO_API extern "C" __declspec(dllexport)
#else
#define YOLO_API extern "C" __declspec(dllimport)
#endif

struct YoloDetection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int32_t classId;
};

YOLO_API void* __cdecl YoloCreate(
    const wchar_t* enginePath,
    int32_t dynamicInputWidth,
    int32_t dynamicInputHeight,
    wchar_t* errorBuffer,
    int32_t errorCapacity);

YOLO_API int32_t __cdecl YoloGetModelInfo(
    void* handle,
    wchar_t* infoBuffer,
    int32_t infoCapacity);

YOLO_API int32_t __cdecl YoloDetectBgra(
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
    int32_t errorCapacity);

YOLO_API void __cdecl YoloDestroy(void* handle);

// Phase 1: ONNX -> TensorRT serialized engine.
//
// Return value:
//   0  success
//  -1  failure (see errorBuffer)
//
// Current intended scope:
// - standard YOLO Detect-style ONNX
// - one input tensor
// - one output tensor
// - batch 1
// - fixed or dynamic NCHW input
// - FP32 or TensorRT FP16 builder mode
YOLO_API int32_t __cdecl YoloBuildEngineFromOnnx(
    const wchar_t* onnxPath,
    const wchar_t* enginePath,
    int32_t inputWidth,
    int32_t inputHeight,
    int32_t enableFp16,
    int32_t workspaceMiB,
    wchar_t* logBuffer,
    int32_t logCapacity,
    wchar_t* errorBuffer,
    int32_t errorCapacity);
