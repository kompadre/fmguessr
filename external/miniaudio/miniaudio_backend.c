#define ma_atomic_global_lock ma_atomic_global_lock_trkr
#define MA_API static
#define MA_NO_DECODER
#define MA_NO_ENCODER
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "miniaudio_backend.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    ma_device device;
    ma_rb rb_left;
    ma_rb rb_right;
    uint32_t sampleRate;
    int is_initialized;
} AudioEngine;

static AudioEngine g_audio = {0};

static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    (void)pInput;
    (void)pDevice;
    float* pOutputF32 = (float*)pOutput;

    size_t bytesAvailL = ma_rb_available_read(&g_audio.rb_left);
    size_t bytesAvailR = ma_rb_available_read(&g_audio.rb_right);

    uint32_t framesAvailL = (uint32_t)(bytesAvailL / sizeof(float));
    uint32_t framesAvailR = (uint32_t)(bytesAvailR / sizeof(float));
    uint32_t framesToRead = (framesAvailL < framesAvailR) ? framesAvailL : framesAvailR;

    if (framesToRead > frameCount) {
        framesToRead = frameCount;
    }

    uint32_t framesReadTotal = 0;
    while (framesReadTotal < framesToRead) {
        size_t reqBytesL = (size_t)(framesToRead - framesReadTotal) * sizeof(float);
        size_t reqBytesR = (size_t)(framesToRead - framesReadTotal) * sizeof(float);
        void* pChunkL = NULL;
        void* pChunkR = NULL;

        ma_rb_acquire_read(&g_audio.rb_left, &reqBytesL, &pChunkL);
        ma_rb_acquire_read(&g_audio.rb_right, &reqBytesR, &pChunkR);

        size_t chunkBytes = (reqBytesL < reqBytesR) ? reqBytesL : reqBytesR;
        uint32_t chunkFrames = (uint32_t)(chunkBytes / sizeof(float));

        if (chunkFrames == 0) {
            ma_rb_commit_read(&g_audio.rb_left, 0);
            ma_rb_commit_read(&g_audio.rb_right, 0);
            break;
        }

        const float* fL = (const float*)pChunkL;
        const float* fR = (const float*)pChunkR;

        for (uint32_t i = 0; i < chunkFrames; ++i) {
            uint32_t outIdx = (framesReadTotal + i) * 2;
            pOutputF32[outIdx + 0] = fL[i];
            pOutputF32[outIdx + 1] = fR[i];
        }

        ma_rb_commit_read(&g_audio.rb_left, chunkFrames * sizeof(float));
        ma_rb_commit_read(&g_audio.rb_right, chunkFrames * sizeof(float));

        framesReadTotal += chunkFrames;
    }

    if (framesReadTotal < frameCount) {
        memset(pOutputF32 + (framesReadTotal * 2), 0, (frameCount - framesReadTotal) * 2 * sizeof(float));
    }
}

int audio_init(uint32_t sampleRate, uint32_t bufferSizeFrames) {
    if (g_audio.is_initialized) return 0;

    size_t ringBufferSizeBytes = (size_t)bufferSizeFrames * sizeof(float);

    if (ma_rb_init(ringBufferSizeBytes, NULL, NULL, &g_audio.rb_left) != MA_SUCCESS) {
        return -1;
    }
    if (ma_rb_init(ringBufferSizeBytes, NULL, NULL, &g_audio.rb_right) != MA_SUCCESS) {
        ma_rb_uninit(&g_audio.rb_left);
        return -2;
    }

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format   = ma_format_f32;
    config.playback.channels = 2; // Stereo output
    config.sampleRate        = sampleRate;
    config.dataCallback      = data_callback;
    config.pUserData         = NULL;

    if (ma_device_init(NULL, &config, &g_audio.device) != MA_SUCCESS) {
        ma_rb_uninit(&g_audio.rb_left);
        ma_rb_uninit(&g_audio.rb_right);
        return -3;
    }

    if (ma_device_start(&g_audio.device) != MA_SUCCESS) {
        ma_device_uninit(&g_audio.device);
        ma_rb_uninit(&g_audio.rb_left);
        ma_rb_uninit(&g_audio.rb_right);
        return -4;
    }

    g_audio.sampleRate = sampleRate;
    g_audio.is_initialized = 1;
    return 0;
}

int audio_write_channels(const float* pLeft, const float* pRight, uint32_t frameCount) {
    if (!g_audio.is_initialized) return -1;

    size_t bytesAvailL = ma_rb_available_write(&g_audio.rb_left);
    size_t bytesAvailR = ma_rb_available_write(&g_audio.rb_right);

    uint32_t framesAvailL = (uint32_t)(bytesAvailL / sizeof(float));
    uint32_t framesAvailR = (uint32_t)(bytesAvailR / sizeof(float));
    uint32_t maxFrames = (framesAvailL < framesAvailR) ? framesAvailL : framesAvailR;

    if (frameCount > maxFrames) {
        frameCount = maxFrames;
    }

    uint32_t framesWrittenTotal = 0;
    while (framesWrittenTotal < frameCount) {
        size_t reqBytesL = (size_t)(frameCount - framesWrittenTotal) * sizeof(float);
        size_t reqBytesR = (size_t)(frameCount - framesWrittenTotal) * sizeof(float);
        void* pChunkL = NULL;
        void* pChunkR = NULL;

        ma_rb_acquire_write(&g_audio.rb_left, &reqBytesL, &pChunkL);
        ma_rb_acquire_write(&g_audio.rb_right, &reqBytesR, &pChunkR);

        size_t chunkBytes = (reqBytesL < reqBytesR) ? reqBytesL : reqBytesR;
        uint32_t chunkFrames = (uint32_t)(chunkBytes / sizeof(float));

        if (chunkFrames == 0) {
            ma_rb_commit_write(&g_audio.rb_left, 0);
            ma_rb_commit_write(&g_audio.rb_right, 0);
            break;
        }

        memcpy(pChunkL, pLeft + framesWrittenTotal, chunkFrames * sizeof(float));
        memcpy(pChunkR, pRight + framesWrittenTotal, chunkFrames * sizeof(float));

        ma_rb_commit_write(&g_audio.rb_left, chunkFrames * sizeof(float));
        ma_rb_commit_write(&g_audio.rb_right, chunkFrames * sizeof(float));

        framesWrittenTotal += chunkFrames;
    }

    return (int)framesWrittenTotal;
}

uint32_t audio_available_write_space(void) {
    if (!g_audio.is_initialized) return 0;
    size_t bytesL = ma_rb_available_write(&g_audio.rb_left);
    size_t bytesR = ma_rb_available_write(&g_audio.rb_right);
    uint32_t framesL = (uint32_t)(bytesL / sizeof(float));
    uint32_t framesR = (uint32_t)(bytesR / sizeof(float));
    return (framesL < framesR) ? framesL : framesR;
}

void audio_close(void) {
    if (!g_audio.is_initialized) return;
    ma_device_uninit(&g_audio.device);
    ma_rb_uninit(&g_audio.rb_left);
    ma_rb_uninit(&g_audio.rb_right);
    g_audio.is_initialized = 0;
}
