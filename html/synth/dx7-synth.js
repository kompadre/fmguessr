// DX7 Synth — client-side DX7 rendering via webdx7 WASM
// Based on webdx7 by Jari Kleimola (MIT License) and MSFA by Raph Levien (Apache 2.0)

let dx7Node = null;
let dx7Ready = false;
let dx7Loading = null;

export function isDx7Ready() {
  return dx7Ready;
}

async function loadModules(audioCtx) {
  const base = new URL('./wasm/', import.meta.url).pathname;
  await audioCtx.audioWorklet.addModule(base + 'dx7.wasm.js');
  await audioCtx.audioWorklet.addModule(base + 'dx7.js');
  await audioCtx.audioWorklet.addModule(new URL('./dx7-processor.js', import.meta.url));
}

export async function initDx7Synth(audioCtx) {
  if (dx7Ready) return true;
  if (dx7Loading) return dx7Loading;

  dx7Loading = (async () => {
    try {
      await loadModules(audioCtx);

      dx7Node = new AudioWorkletNode(audioCtx, 'dx7-synth', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
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
  // Note On after patch loads (sysex must be processed first by the ring buffer)
  dx7Node.port.postMessage({ type: 'midi', data: [0x90, midi, velocity] });
  return () => {
    dx7Node.port.postMessage({ type: 'midi', data: [0x80, midi, 0] });
  };
}

export async function renderOffline(voiceData, midi, durationMs) {
  const sr = 44100;
  const renderDur = Math.max(durationMs, 500);
  const captureDur = (renderDur / 1000) + 0.3;
  const captureSamples = Math.ceil(captureDur * sr);

  const tmpCtx = new AudioContext({ sampleRate: sr });
  if (tmpCtx.state === 'suspended') await tmpCtx.resume();

  try {
    await loadModules(tmpCtx);

    const node = new AudioWorkletNode(tmpCtx, 'dx7-synth', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });

    const patch = typeof voiceData === 'string' ? base64ToUint8Array(voiceData) : voiceData;
    node.port.postMessage({ type: 'patch', data: patch.buffer });
    node.port.postMessage({ type: 'midi', data: [0x90, midi, 100] });

    // Capture via ScriptProcessorNode
    const collected = new Float32Array(captureSamples);
    let offset = 0;
    const scriptNode = tmpCtx.createScriptProcessor(1024, 1, 1);
    scriptNode.onaudioprocess = (e) => {
      if (offset >= captureSamples) return;
      const input = e.inputBuffer.getChannelData(0);
      const end = Math.min(offset + input.length, captureSamples);
      collected.set(input.subarray(0, end - offset), offset);
      offset = end;
    };
    node.connect(scriptNode);
    scriptNode.connect(tmpCtx.destination);

    await new Promise((resolve) => {
      setTimeout(() => {
        node.port.postMessage({ type: 'midi', data: [0x80, midi, 0] });
        setTimeout(() => {
          scriptNode.disconnect();
          node.disconnect();
          resolve();
        }, 300);
      }, renderDur);
    });

    tmpCtx.close();
    return samplesToWavBlob(collected.subarray(0, offset), sr);
  } catch (err) {
    tmpCtx.close();
    throw err;
  }
}

function samplesToWavBlob(samples, sr) {
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
