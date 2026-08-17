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


struct YoloSegDetection
{
    // Axis-aligned bounding box derived from the final binary instance mask
    // in original-image coordinates.
    float x1;
    float y1;
    float x2;
    float y2;

    float score;
    int32_t classId;

    // Binary mask area in original-image pixels.
    float maskAreaPixels;

    // Minimum-area rotated rectangle derived from the mask contour.
    // Angle is normalized to the long-axis orientation in [0, pi).
    float centerX;
    float centerY;
    float rotatedWidth;
    float rotatedHeight;
    float angleRadians;

    // Four corners of the minimum-area rotated rectangle.
    float p1x;
    float p1y;
    float p2x;
    float p2y;
    float p3x;
    float p3y;
    float p4x;
    float p4y;

    // 1..65535, matching the per-pixel UInt16 instanceMask output.
    int32_t maskId;
};

YOLO_API void* __cdecl YoloCreate(
    const wchar_t* enginePath,
    int32_t dynamicInputWidth,
    int32_t dynamicInputHeight,
    wchar_t* errorBuffer,
    int32_t errorCapacity);


// Return task hint using engine output structure and expected class count:
//   0 = standard raw Detect
//   1 = standard raw OBB
//   2 = YOLO26 end-to-end instance segmentation (prediction + proto)
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


// YOLO26 end-to-end instance segmentation.
//
// Expected engine outputs:
//   prediction: [1, max_det, 6 + nm] or [1, 6 + nm, max_det]
//               [x1, y1, x2, y2, confidence, class_id, mask_coefficients...]
//   proto:      [1, nm, mask_h, mask_w]
//
// No NMS is applied for the YOLO26 end-to-end prediction head.
// The native code reconstructs the instance mask, derives an axis-aligned
// mask bounding box, computes a convex hull + minimum-area rectangle, and
// writes an original-resolution instance ID map:
//
//   instanceMask[y * maskStride + x] = 0       background
//   instanceMask[y * maskStride + x] = 1..65535  instance id
//
// maskThreshold is a probability in [0,1]; 0.5 matches sigmoid(logit) > 0.5.
YOLO_API int32_t __cdecl YoloDetectSegBgra(
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
    int32_t errorCapacity);

YOLO_API void __cdecl YoloDestroy(void* handle);

// Phase 1: ONNX -> TensorRT serialized engine.
//
// Return value:
//   0  success
//  -1  failure (see errorBuffer)
//
// Current intended scope:
// - standard Ultralytics YOLO Detect / OBB raw-output ONNX (one output)
// - YOLO26 instance segmentation ONNX (prediction + proto outputs; raw export recommended)
// - one input tensor
// - one or two output tensors
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
