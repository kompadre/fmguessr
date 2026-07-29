#ifndef MINIAUDIO_BACKEND_H
#define MINIAUDIO_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int audio_init(uint32_t sampleRate, uint32_t bufferSizeFrames);
void audio_close(void);
int audio_write_channels(const float* pLeft, const float* pRight, uint32_t frameCount);
uint32_t audio_available_write_space(void);

#ifdef __cplusplus
}
#endif

#endif // MINIAUDIO_BACKEND_H
