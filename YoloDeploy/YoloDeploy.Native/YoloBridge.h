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

struct YoloObbDetection
{
    // Rotated box in original-image coordinates.
    float centerX;
    float centerY;
    float width;
    float height;

    // Rotation in radians. For Ultralytics OBB this is the model's
    // native xywhr angle convention.
    float angleRadians;

    float score;
    int32_t classId;

    // Four corners in original-image coordinates, ordered around the box.
    float p1x;
    float p1y;
    float p2x;
    float p2y;
    float p3x;
    float p3y;
    float p4x;
    float p4y;
};

YOLO_API void* __cdecl YoloCreate(
    const wchar_t* enginePath,
    int32_t dynamicInputWidth,
    int32_t dynamicInputHeight,
    wchar_t* errorBuffer,
    int32_t errorCapacity);


// Return task hint using output channel count and expected class count:
//   0 = standard Detect
//   1 = OBB
//  -1 = unknown / cannot infer
YOLO_API int32_t __cdecl YoloGetTaskHint(
    void* handle,
    int32_t expectedClassCount);

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


// Ultralytics YOLO OBB raw-output inference.
//
// Expected raw output layout:
//   [1, 5 + nc, N] or [1, N, 5 + nc]
// with channels:
//   x, y, w, h, class_probs..., angle
//
// expectedClassCount:
//   > 0 : validate the model output against the application's label count.
//   <=0 : skip the validation.
YOLO_API int32_t __cdecl YoloDetectObbBgra(
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
    int32_t errorCapacity);

YOLO_API void __cdecl YoloDestroy(void* handle);

// Phase 1: ONNX -> TensorRT serialized engine.
//
// Return value:
//   0  success
//  -1  failure (see errorBuffer)
//
// Current intended scope:
// - standard Ultralytics YOLO Detect or OBB raw-output ONNX
// - one input tensor
// - one output tensor
// - batch 1
// - fixed or dynamic NCHW input
// - FP32 or TensorRT FP16 builder mode

// Phase 2: query active CUDA GPU / CUDA / TensorRT version information.
// Return value: 0 success, -1 failure.
YOLO_API int32_t __cdecl YoloGetGpuInfoJson(
    wchar_t* jsonBuffer,
    int32_t jsonCapacity,
    wchar_t* errorBuffer,
    int32_t errorCapacity);

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
