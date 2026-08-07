// DX7 AudioWorkletProcessor — standalone WASM via msfa_bridge
// Based on MSFA by Google (Apache 2.0), patched for fmguessr

if (!AudioWorkletGlobalScope._dx7Registered) {
  AudioWorkletGlobalScope._dx7Registered = true;

  const wasiStub = {
    fd_close: () => 0,
    fd_fdstat_get: () => 0,
    fd_seek: () => 0,
    fd_write: () => 0,
    poll_oneoff: () => 0,
    proc_exit: () => {},
    clock_time_get: () => 0,
  };

  const envStub = {
    emscripten_notify_memory_growth: () => {},
  };

  class DX7Processor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.bufsize = 128;
      this.sr = sampleRate;
      this.ready = false;

      const binary = AudioWorkletGlobalScope._msfaWasmBinary;
      if (!binary) {
        console.error('[DX7] No WASM binary found');
        return;
      }

      this.port.onmessage = this.handleMessage.bind(this);
      this._init(binary);
    }

    async _init(binary) {
      try {
        const { instance } = await WebAssembly.instantiate(binary, {
          wasi_snapshot_preview1: wasiStub,
          env: envStub,
        });
        this.exp = instance.exports;
        this.heapI16 = new Int16Array(this.exp.memory.buffer);

        this.exp.init_engine(this.sr);

        // Get static patch buffer address and create a view
        this.patchPtr = this.exp.get_patch_buffer();
        this.patchView = new Uint8Array(this.exp.memory.buffer, this.patchPtr, 128);

        this.ready = true;
        this.port.postMessage({ type: 'ready' });

        if (this._pending) {
          for (const msg of this._pending) this.handleMessage(msg);
          this._pending = null;
        }
      } catch (err) {
        console.error('[DX7] WASM init failed:', err);
      }
    }

    refreshHeap() {
      // Recreate views if memory grew
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
          this.exp.send_midi(msg.data[0], msg.data[1], msg.data[2]);
          break;
        case 'raw-midi':
          this.exp.send_midi(msg.data[0], msg.data[1], msg.data[2]);
          break;
      }
    }

    loadPatch(data, slot) {
      const voice = new Uint8Array(data);
      if (voice.length < 128) return;
      this.refreshHeap();
      // Copy directly into static buffer — no malloc/free
      this.patchView.set(voice.subarray(0, 128));
      if (slot > 0 && this.exp.load_voice_channel_from_buffer) {
        this.exp.load_voice_channel_from_buffer(slot);
      } else {
        this.exp.load_voice_from_buffer();
      }
    }

    process(inputs, outputs) {
      if (!this.ready) return true;

      const ptr = this.exp.render_audio(this.bufsize);
      if (!ptr) return true;

      this.refreshHeap();
      const out = outputs[0][0];
      const i16base = ptr >> 1;
      for (let i = 0; i < this.bufsize; i++) {
        out[i] = this.heapI16[i16base + i] * (1.0 / 32768.0);
      }
      return true;
    }
  }

  registerProcessor('dx7-synth', DX7Processor);
}
