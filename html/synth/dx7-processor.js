// DX7 AudioWorkletProcessor — wraps webdx7 WASM (MSFA engine)
// Based on webdx7 by Jari Kleimola (MIT License) and MSFA by Raph Levien (Apache 2.0)

if (!AudioWorkletGlobalScope._dx7Registered) {
  AudioWorkletGlobalScope._dx7Registered = true;

  class DX7Processor extends AudioWorkletProcessor {
    constructor(options) {
      super();
      const WAM = AudioWorkletGlobalScope.WAM.DX7;
      this.WAM = WAM;
      this.bufsize = 128;
      this.sr = sampleRate;

      const wam_ctor = WAM.cwrap('createModule', 'number', []);
      const wam_init = WAM.cwrap('wam_init', null, ['number','number','number','string']);
      this.wam_onmidi = WAM.cwrap('wam_onmidi', null, ['number','number','number','number']);
      this.wam_onpatch = WAM.cwrap('wam_onpatch', null, ['number','number','number']);
      this.wam_onprocess = WAM.cwrap('wam_onprocess', 'number', ['number','number','number']);

      this.inst = wam_ctor();
      wam_init(this.inst, this.bufsize, this.sr, '');

      const obuf = WAM._malloc(this.bufsize * 4);
      this.audiobus = WAM._malloc(8);
      WAM.setValue(this.audiobus, 0, 'i32');
      WAM.setValue(this.audiobus + 4, obuf, 'i32');
      this.outBufIdx = obuf / 4;

      // Offline mode: patch and note passed via processorOptions
      // (postMessage doesn't work because OfflineAudioContext.startRendering
      //  doesn't yield to the event loop)
      const opts = options.processorOptions;
      if (opts && opts.patch) {
        this.handlePatch(opts.patch);
        if (opts.note != null) {
          this.handleMidi(0x90, opts.note, opts.velocity || 100);
        }
      }

      this.port.onmessage = this.handleMessage.bind(this);
      this.port.start();
    }

    handleMessage(e) {
      const msg = e.data;
      switch (msg.type) {
        case 'midi': this.handleMidi(msg.data[0], msg.data[1], msg.data[2]); break;
        case 'patch': this.handlePatch(msg.data); break;
      }
    }

    handleMidi(status, data1, data2) {
      this.wam_onmidi(this.inst, status, data1, data2);
    }

    handlePatch(data) {
      const buffer = new Uint8Array(data);
      const WAM = this.WAM;
      const buf = WAM._malloc(buffer.length);
      for (let i = 0; i < buffer.length; i++)
        WAM.setValue(buf + i, buffer[i], 'i8');
      this.wam_onpatch(this.inst, buf, buffer.length);
      WAM._free(buf);
    }

    process(inputs, outputs) {
      const WAM = this.WAM;
      this.wam_onprocess(this.inst, this.audiobus, 0);
      outputs[0][0].set(WAM.HEAPF32.subarray(this.outBufIdx, this.outBufIdx + this.bufsize));
      return true;
    }
  }

  registerProcessor('dx7-synth', DX7Processor);
}
