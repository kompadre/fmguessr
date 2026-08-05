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
        });
        this.exp = instance.exports;
        this.heapI16 = new Int16Array(this.exp.memory.buffer);

        this.exp.init_engine(this.sr);
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

    handleMessage(e) {
      if (!this.ready) {
        if (!this._pending) this._pending = [];
        this._pending.push(e);
        return;
      }
      const msg = e.data;
      switch (msg.type) {
        case 'patch': this.loadPatch(msg.data); break;
        case 'midi':
          if (msg.data[0] === 0x90) this.exp.note_on(msg.data[1], msg.data[2]);
          else if (msg.data[0] === 0x80) this.exp.note_off(msg.data[1]);
          break;
      }
    }

    loadPatch(data) {
      const voice = new Uint8Array(data);
      if (voice.length < 128) return;
      // Allocate temp buffer in WASM heap, copy voice, load, free
      const ptr = this.exp._Znam(128);
      new Uint8Array(this.exp.memory.buffer).set(voice.subarray(0, 128), ptr);
      this.exp.load_voice(ptr);
      this.exp._ZdaPv(ptr);
    }

    process(inputs, outputs) {
      if (!this.ready) return true;

      // render_audio returns pointer to int16 buffer
      const ptr = this.exp.render_audio(this.bufsize);
      if (!ptr) return true;

      // Update heap view (memory may have grown)
      const heapI16 = new Int16Array(this.exp.memory.buffer);
      const out = outputs[0][0];
      const i16base = ptr >> 1;
      for (let i = 0; i < this.bufsize; i++) {
        out[i] = heapI16[i16base + i] * (1.0 / 32768.0);
      }
      return true;
    }
  }

  registerProcessor('dx7-synth', DX7Processor);
}
