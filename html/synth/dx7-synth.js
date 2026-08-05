// DX7 Synth — client-side DX7 rendering via webdx7 WASM
// Based on webdx7 by Jari Kleimola (MIT License) and MSFA by Raph Levien (Apache 2.0)

let dx7Node = null;
let dx7Ready = false;
let dx7Loading = null;
let wasmBinary = null;

export function isDx7Ready() {
  return dx7Ready;
}

export async function initDx7Synth(audioCtx) {
  if (dx7Ready) return true;
  if (dx7Loading) return dx7Loading;

  dx7Loading = (async () => {
    try {
      const base = new URL('./wasm/', import.meta.url).pathname;
      if (!wasmBinary) {
        const resp = await fetch(base + 'dx7.wasm.js');
        const text = await resp.text();
        const match = text.match(/new Uint8Array\((\[[\d,\s]+\])\)/);
        if (!match) throw new Error('Failed to parse dx7.wasm.js');
        wasmBinary = new Uint8Array(JSON.parse(match[1]));
      }

      await audioCtx.audioWorklet.addModule(base + 'dx7.wasm.js');
      await audioCtx.audioWorklet.addModule(base + 'dx7.js');
      await audioCtx.audioWorklet.addModule(new URL('./dx7-processor.js', import.meta.url));

      dx7Node = new AudioWorkletNode(audioCtx, 'dx7-synth', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: {
          mod: AudioWorkletGlobalScope.WAM.DX7,
          wasmBinary: wasmBinary,
        },
      });
      dx7Node.connect(audioCtx.destination);
      dx7Ready = true;
      return true;
    } catch (err) {
      console.warn('DX7 synth init failed:', err);
      dx7Ready = false;
      dx7Node = null;
      return false;
    }
  })();

  return dx7Loading;
}

function base64ToUint8Array(b64) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}

export function playNoteRealtime(voiceData, midi, velocity = 100) {
  if (!dx7Ready || !dx7Node) return null;
  const patch = typeof voiceData === 'string' ? base64ToUint8Array(voiceData) : voiceData;
  dx7Node.port.postMessage({ type: 'patch', data: patch.buffer });
  dx7Node.port.postMessage({ type: 'midi', data: [0x90, midi, velocity] });
  return () => {
    dx7Node.port.postMessage({ type: 'midi', data: [0x80, midi, 0] });
  };
}

export async function renderOffline(voiceData, midi, durationMs) {
  const sr = 44100;
  const totalSec = (durationMs / 1000) + 0.25;
  const totalSamples = Math.ceil(totalSec * sr);

  const offlineCtx = new OfflineAudioContext(1, totalSamples, sr);
  const base = new URL('./wasm/', import.meta.url).pathname;
  await offlineCtx.audioWorklet.addModule(base + 'dx7.wasm.js');
  await offlineCtx.audioWorklet.addModule(base + 'dx7.js');
  await offlineCtx.audioWorklet.addModule(new URL('./dx7-processor.js', import.meta.url));

  const node = new AudioWorkletNode(offlineCtx, 'dx7-synth', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      mod: AudioWorkletGlobalScope.WAM.DX7,
      wasmBinary: wasmBinary,
    },
  });
  node.connect(offlineCtx.destination);

  const patch = typeof voiceData === 'string' ? base64ToUint8Array(voiceData) : voiceData;
  node.port.postMessage({ type: 'patch', data: patch.buffer });
  node.port.postMessage({ type: 'midi', data: [0x90, midi, 100] });

  const audioBuffer = await offlineCtx.startRendering();
  return audioBufferToWavBlob(audioBuffer);
}

function audioBufferToWavBlob(audioBuffer) {
  const samples = audioBuffer.getChannelData(0);
  const sr = audioBuffer.sampleRate;
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
