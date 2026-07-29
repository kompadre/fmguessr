#include "wrapper.h"
#include "synth.h"
#include "synth_unit.h"
#include "ringbuffer.h"
#include "patch.h"

extern "C" {

void dx7_global_init(double sample_rate) {
    SynthUnit::Init(sample_rate);
}

DX7RingBuffer dx7_ringbuffer_create() {
    return reinterpret_cast<DX7RingBuffer>(new RingBuffer());
}

void dx7_ringbuffer_destroy(DX7RingBuffer rb) {
    delete reinterpret_cast<RingBuffer*>(rb);
}

void dx7_ringbuffer_write(DX7RingBuffer rb, const uint8_t* bytes, int size) {
    reinterpret_cast<RingBuffer*>(rb)->Write(bytes, size);
}

DX7Instance dx7_synth_create(DX7RingBuffer rb) {
    return reinterpret_cast<DX7Instance>(new SynthUnit(reinterpret_cast<RingBuffer*>(rb)));
}

void dx7_synth_destroy(DX7Instance synth) {
    delete reinterpret_cast<SynthUnit*>(synth);
}

void dx7_get_samples(DX7Instance synth, int n_samples, int16_t* buffer) {
    reinterpret_cast<SynthUnit*>(synth)->GetSamples(n_samples, buffer);
}

void dx7_get_channel_samples(DX7Instance synth, int channel, int n_samples, int32_t* buffer) {
    reinterpret_cast<SynthUnit*>(synth)->GetChannelSamples(channel, n_samples, buffer);
}

void dx7_unpack_patch(const char* bulk, char* patch) {
    UnpackPatch(bulk, patch);
}

void dx7_set_channel_volume(int32_t channel, int32_t volume) {
    SynthUnit::SetChannelVolume(channel, volume);
}

void dx7_set_bank(DX7Instance synth, const uint8_t* bank_data) {
    reinterpret_cast<SynthUnit*>(synth)->SetBank(bank_data);
}

void dx7_set_voice(DX7Instance synth, int slot, const uint8_t* voice_data) {
    reinterpret_cast<SynthUnit*>(synth)->SetVoice(slot, voice_data);
}

void dx7_get_bank(DX7Instance synth, uint8_t* bank_data) {
    reinterpret_cast<SynthUnit*>(synth)->GetBank(bank_data);
}

}
