#ifndef MSFA_WRAPPER_H_
#define MSFA_WRAPPER_H_

#include <stdint.h>

typedef void* DX7Instance;
typedef void* DX7RingBuffer;

#ifdef __cplusplus
extern "C" {
#endif

// Global configuration
void dx7_global_init(double sample_rate);

// RingBuffer allocation & control
DX7RingBuffer dx7_ringbuffer_create();
void dx7_ringbuffer_destroy(DX7RingBuffer rb);
void dx7_ringbuffer_write(DX7RingBuffer rb, const uint8_t* bytes, int size);

// SynthUnit allocation & execution
DX7Instance dx7_synth_create(DX7RingBuffer rb);
void dx7_synth_destroy(DX7Instance synth);
void dx7_get_samples(DX7Instance synth, int n_samples, int16_t* buffer);
// Read per-channel pre-filter audio (int32). Call after dx7_get_samples.
void dx7_get_channel_samples(DX7Instance synth, int channel, int n_samples, int32_t* buffer);

// Pure utility parser
void dx7_unpack_patch(const char* bulk, char* patch);

void dx7_set_channel_volume(int32_t channel, int32_t volume);

void dx7_set_bank(DX7Instance synth, const uint8_t* bank_data);
void dx7_set_voice(DX7Instance synth, int slot, const uint8_t* voice_data);
void dx7_get_bank(DX7Instance synth, uint8_t* bank_data);

#ifdef __cplusplus
}
#endif

#endif
