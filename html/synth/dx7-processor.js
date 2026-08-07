// DX7 AudioWorkletProcessor — WASM via processorOptions
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
    constructor(options) {
      super();
      this.bufsize = 128;
      this.sr = sampleRate;
      this.ready = false;
      this.hasError = false;
      this._renderCount = 0;
      this._lastDebugTime = 0;

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
        this.hasError = true;
        console.error('[DX7] WASM init failed:', err);
        this.port.postMessage({ type: 'error', error: err.message });
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
          this._debugMidi(0xC0 | msg.channel, msg.slot, 0, 'voice-channel');
          break;
        case 'midi':
        case 'raw-midi':
          this.exp.send_midi(msg.data[0], msg.data[1], msg.data[2]);
          this._debugMidi(msg.data[0], msg.data[1], msg.data[2], msg.type);
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

    _debugMidi(status, d1, d2, source) {
      const cmd = status & 0xf0;
      const ch = status & 0x0f;
      const NAMES = { 0x80: 'NOTE_OFF', 0x90: 'NOTE_ON', 0xA0: 'AFTERTOUCH', 0xB0: 'CC', 0xC0: 'PC', 0xD0: 'PRESSURE', 0xE0: 'PITCHBEND' };
      const name = NAMES[cmd] || 'UNKNOWN';
      let detail = '';
      if (cmd === 0x90) detail = ` note=${d1} vel=${d2}`;
      else if (cmd === 0x80) detail = ` note=${d1}`;
      else if (cmd === 0xC0) detail = ` program=${d1}`;
      else detail = ` d1=${d1} d2=${d2}`;
      this.port.postMessage({
        type: 'debug',
        tag: 'MIDI',
        msg: `[MIDI] ${name} ch=${ch}${detail} (0x${status.toString(16)} ${d1} ${d2}) [${source}]`,
      });
    }

    process(inputs, outputs) {
      if (!this.ready || this.hasError) return true;

      const ptr = this.exp.render_audio(this.bufsize);
      if (!ptr) return true;

      this.refreshHeap();
      const out = outputs[0][0];
      const i16base = ptr >> 1;

      // Compute RMS and peak for this block
      let sumSq = 0;
      let peak = 0;
      for (let i = 0; i < this.bufsize; i++) {
        const s = this.heapI16[i16base + i];
        const absS = s < 0 ? -s : s;
        if (absS > peak) peak = absS;
        sumSq += s * s;
        out[i] = s * (1.0 / 32768.0);
      }
      const rms = Math.sqrt(sumSq / this.bufsize);

      this._renderCount++;
      // Post render debug every ~1s (every ~345 blocks at 128/44100)
      if (this._renderCount % 345 === 1 || peak > 100) {
        const now = currentTime;
        this.port.postMessage({
          type: 'debug',
          tag: 'RENDER',
          msg: `[RENDER] block=${this._renderCount} rms=${rms.toFixed(1)} peak=${peak} samples=${this.bufsize}`,
        });
      }

      return true;
    }
  }

  registerProcessor('dx7-synth', DX7Processor);
}
