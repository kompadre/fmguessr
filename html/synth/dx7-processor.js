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
      this.wam_onsysex = WAM.cwrap('wam_onsysex', null, ['number','number','number']);
      this.wam_onprocess = WAM.cwrap('wam_onprocess', 'number', ['number','number','number']);

      this.inst = wam_ctor();
      wam_init(this.inst, this.bufsize, this.sr, '');

      const obuf = WAM._malloc(this.bufsize * 4);
      this.audiobus = WAM._malloc(8);
      WAM.setValue(this.audiobus, 0, 'i32');
      WAM.setValue(this.audiobus + 4, obuf, 'i32');
      this.outBufIdx = obuf / 4;

      const opts = options.processorOptions;
      if (opts && opts.patch) {
        this.loadPatch(opts.patch);
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
        case 'patch': this.loadPatch(msg.data); break;
      }
    }

    // Load patch as DX7 sysex bank dump (slot 0), same as demo's dx7.setPatch()
    loadPatch(data) {
      const voice = new Uint8Array(data);
      if (voice.length < 128) return;

      // Build 4104-byte DX7 sysex: 6 header + 4096 data + 1 checksum + 1 EOX
      const sysex = new Uint8Array(4104);
      sysex[0] = 0xF0; // Sysex start
      sysex[1] = 0x43; // Yamaha ID
      sysex[2] = 0x00; // Sub-status + channel
      sysex[3] = 0x09; // Format: 32 voices
      sysex[4] = 0x20; // Data length MSB (4096)
      sysex[5] = 0x00; // Data length LSB
      // Copy voice into slot 0 (first 128 bytes of 4096-byte data block)
      sysex.set(voice, 6);
      // Remaining 3968 bytes stay zero (empty slots)
      // Checksum: XOR of 4096 data bytes, complemented & 0x7F
      let checksum = 0;
      for (let i = 6; i < 6 + 4096; i++) checksum ^= sysex[i];
      sysex[6 + 4096] = (~checksum) & 0x7F;
      sysex[6 + 4096 + 1] = 0xF7; // EOX

      const WAM = this.WAM;
      const buf = WAM._malloc(sysex.length);
      for (let i = 0; i < sysex.length; i++)
        WAM.setValue(buf + i, sysex[i], 'i8');
      this.wam_onsysex(this.inst, buf, sysex.length);
      WAM._free(buf);

      // Select slot 0 via Program Change
      this.handleMidi(0xC0, 0, 0);
    }

    handleMidi(status, data1, data2) {
      this.wam_onmidi(this.inst, status, data1, data2);
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
