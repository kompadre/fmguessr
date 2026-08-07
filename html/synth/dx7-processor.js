// DX7 AudioWorkletProcessor — WASM via processorOptions
// Based on MSFA by Google (Apache 2.0), patched for fmguessr

const wasiStub = {
  fd_close: () => 0,
  fd_fdstat_get: () => 0,
  fd_seek: () => 0,
  fd_write: () => 0,
  poll_oneoff: () => 0,
  proc_exit: () => {},
};

class DX7Processor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufsize = 128;
    this.sr = sampleRate;
    this.ready = false;
    this.hasError = false;

    // Preview playback state (fed from main-thread offline render)
    this.previewBuffer = null;
    this.previewPos = 0;
    this.previewActive = false;

    this.port.onmessage = this.handleMessage.bind(this);

    const wasmBytes = options.processorOptions?.wasmBytes;
    if (wasmBytes) {
      this._init(wasmBytes);
    } else {
      this.hasError = true;
      console.error('[DX7] No wasmBytes in processorOptions');
    }
  }

  async _init(wasmBytes) {
    try {
      const { instance } = await WebAssembly.instantiate(wasmBytes, {
        wasi_snapshot_preview1: wasiStub,
      });
      this.exp = instance.exports;
      this.heapI16 = new Int16Array(this.exp.memory.buffer);

      this.exp.init_engine(this.sr);

      this.patchPtr = this.exp.get_patch_buffer();
      this.patchView = new Uint8Array(this.exp.memory.buffer, this.patchPtr, 128);

      this.ready = true;
      this.port.postMessage({ type: 'ready' });

      if (this._pending) {
        for (const msg of this._pending) this.handleMessage(msg);
        this._pending = null;
      }
    } catch (err) {
      this.hasError = true;
      console.error('[DX7] WASM init failed:', err);
      this.port.postMessage({ type: 'error', error: err.message });
    }
  }

  refreshHeap() {
    if (this.heapI16.buffer !== this.exp.memory.buffer) {
      this.heapI16 = new Int16Array(this.exp.memory.buffer);
      this.patchView = new Uint8Array(this.exp.memory.buffer, this.patchPtr, 128);
    }
  }

  handleMessage(e) {
    if (!this.ready) {
      if (!this._pending) this._pending = [];
      this._pending.push(e);
      return;
    }
    const msg = e.data;
    switch (msg.type) {
      case 'patch':
        this.loadPatch(msg.data, msg.slot || 0);
        break;
      case 'voice-channel':
        this.loadPatch(msg.data, msg.slot);
        this.exp.send_midi(0xC0 | msg.channel, msg.slot, 0);
        break;
      case 'midi':
      case 'raw-midi':
        this.exp.send_midi(msg.data[0], msg.data[1], msg.data[2]);
        break;
      case 'play-samples':
        // Overwrite any active preview, reset position
        this.previewBuffer = msg.samples;
        this.previewPos = 0;
        this.previewActive = true;
        break;
    }
  }

  loadPatch(data, slot) {
    const voice = new Uint8Array(data);
    if (voice.length < 128) return;
    this.refreshHeap();
    this.patchView.set(voice.subarray(0, 128));
    if (slot > 0 && this.exp.load_voice_channel_from_buffer) {
      this.exp.load_voice_channel_from_buffer(slot);
    } else {
      this.exp.load_voice_from_buffer();
    }
  }

  process(inputs, outputs) {
    if (!this.ready || this.hasError) return true;

    // Always render synth (keeps pianoroll notes alive)
    const ptr = this.exp.render_audio(this.bufsize);
    if (!ptr) return true;

    this.refreshHeap();
    const out = outputs[0][0];
    const i16base = ptr >> 1;

    const hasPreview = this.previewActive && this.previewBuffer &&
                       this.previewPos < this.previewBuffer.length;

    for (let i = 0; i < this.bufsize; i++) {
      let sample = this.heapI16[i16base + i] * (1.0 / 32768.0);

      if (hasPreview) {
        // Mix with 0.5 headroom on both sources
        sample = sample * 0.5 + this.previewBuffer[this.previewPos++] * 0.5;
      }

      out[i] = sample;
    }

    // Deactivate preview when buffer exhausted
    if (this.previewActive && this.previewPos >= (this.previewBuffer ? this.previewBuffer.length : 0)) {
      this.previewActive = false;
      this.previewBuffer = null;
    }

    return true;
  }
}

try {
  registerProcessor('dx7-synth', DX7Processor);
} catch (e) {
  // Already registered in this scope (browser caches module execution across contexts)
}
