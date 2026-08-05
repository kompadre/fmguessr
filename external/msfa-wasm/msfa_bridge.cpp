// MSFA DX7 WASM bridge — clean C exports for AudioWorklet
// Based on MSFA by Google (Apache 2.0), patched for fmguessr

#include "synth_unit.h"
#include "ringbuffer.h"
#include "patch.h"
#include "freqlut.h"
#include "exp2.h"
#include "sin.h"

static RingBuffer rb;
static SynthUnit* synth = nullptr;
static int16_t* render_buf = nullptr;
static int render_buf_size = 0;

extern "C" {

void init_engine(int sample_rate) {
  Freqlut::init(sample_rate);
  Exp2::init();
  Sin::init();
  Lfo::init(sample_rate);
  PitchEnv::init(sample_rate);
  synth = new SynthUnit(&rb);
}

void load_voice(const uint8_t* voice_data) {
  if (!synth) return;
  synth->SetVoice(0, voice_data);
}

void note_on(int note, int velocity) {
  uint8_t msg[3] = { 0x90, (uint8_t)note, (uint8_t)velocity };
  rb.Write(msg, 3);
}

void note_off(int note) {
  uint8_t msg[3] = { 0x80, (uint8_t)note, 0 };
  rb.Write(msg, 3);
}

// Returns pointer to int16 buffer (render_buf_size samples)
int render_audio(int n_samples) {
  if (!synth) return 0;
  if (render_buf_size < n_samples) {
    delete[] render_buf;
    render_buf = new int16_t[n_samples];
    render_buf_size = n_samples;
  }
  synth->GetSamples(n_samples, render_buf);
  return (int)(uintptr_t)render_buf;
}

int main() { return 0; }

}
