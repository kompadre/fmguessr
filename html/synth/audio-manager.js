// Audio Manager — singleton audio engine.
// AudioContext + AudioWorklet initialized once, live for page lifetime.
// No repeated addModule(), no blob URLs, no re-initialization.

const MAX_CHANNELS = 8;

let ctx = null;
let dx7Node = null;
let initialized = false;
let initPromise = null;

// Multi-channel state: channel → { voiceData (Uint8Array), b64Key (string) }
const channelState = new Array(MAX_CHANNELS).fill(null);

let onStatusChange = null;

function setState(state) {
    if (onStatusChange) onStatusChange(state);
}

export function setOnStatusChange(fn) {
    onStatusChange = fn;
    fn(getState());
}

export function isReady() {
    return initialized && ctx && ctx.state === 'running';
}

export function getState() {
    if (!ctx) return 'closed';
    if (ctx.state === 'suspended') return 'suspended';
    if (initialized) return 'ready';
    if (initPromise) return 'loading';
    return 'closed';
}

// --- Singleton initialization ---

async function initAudio() {
    if (initialized) return ctx;
    if (initPromise) return initPromise;

    initPromise = (async () => {
        ctx = new (window.AudioContext || window.webkitAudioContext)();
        setState('loading');

        // Load modules ONCE — static file paths, browser caches them
        const base = new URL('./wasm/', import.meta.url).pathname;
        await ctx.audioWorklet.addModule(base + 'msfa.wasm.js');
        await ctx.audioWorklet.addModule(new URL('./dx7-processor.js', import.meta.url));

        // Create ONE persistent worklet node
        dx7Node = new AudioWorkletNode(ctx, 'dx7-synth', {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [1],
        });
        dx7Node.connect(ctx.destination);

        // Wait for WASM ready
        await new Promise((resolve) => {
            dx7Node.port.onmessage = (e) => {
                if (e.data.type === 'ready') resolve();
            };
            setTimeout(resolve, 2000);
        });

        initialized = true;
        setState('ready');
        return ctx;
    })();

    return initPromise;
}

export async function acquire() {
    if (initialized && ctx && ctx.state === 'running') {
        return ctx;
    }
    if (ctx && ctx.state === 'suspended') {
        await ctx.resume();
        return ctx;
    }
    return initAudio();
}

// --- Voice loading ---

function base64ToUint8Array(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
}

/**
 * Load a voice into a bank slot and assign to a channel.
 * @returns {number} channel number (same as slot), or -1 if failed
 */
export function loadVoice(voiceData, slot) {
    if (!isReady() || !dx7Node) return -1;
    if (slot < 0 || slot >= MAX_CHANNELS) return -1;

    const patch = typeof voiceData === 'string' ? base64ToUint8Array(voiceData) : voiceData;
    const b64Key = typeof voiceData === 'string' ? voiceData : null;

    // Skip if already loaded on this slot
    if (channelState[slot] && channelState[slot].b64Key === b64Key) {
        return slot;
    }

    dx7Node.port.postMessage({
        type: 'voice-channel',
        slot: slot,
        channel: slot,
        data: patch.buffer,
    });

    channelState[slot] = { voiceData: patch, b64Key };
    return slot;
}

/**
 * Load all VoiceBank entries into channels.
 */
export function loadAllVoices(entries) {
    if (!isReady() || !dx7Node) return;
    for (let i = 0; i < Math.min(entries.length, MAX_CHANNELS); i++) {
        if (entries[i] && entries[i].voiceData) {
            loadVoice(entries[i].voiceData, i);
        }
    }
}

/**
 * Hot-swap a voice on a channel.
 */
export function updateVoice(voiceData, channel) {
    loadVoice(voiceData, channel);
}

// --- Channel-based playback ---

/**
 * Play a note on a specific MIDI channel (no patch load).
 * @returns {function|null} stop function
 */
export function playNoteOnChannel(channel, midi, velocity = 100) {
    if (!isReady() || !dx7Node) return null;
    if (channel < 0 || channel >= MAX_CHANNELS) return null;

    dx7Node.port.postMessage({ type: 'midi', data: [0x90 | channel, midi, velocity] });
    return () => {
        if (dx7Node) dx7Node.port.postMessage({ type: 'midi', data: [0x80 | channel, midi, 0] });
    };
}

/**
 * Legacy: play note with voiceData (loads patch on channel 0).
 */
export function playNoteRealtime(voiceData, midi, velocity = 100) {
    if (!isReady() || !dx7Node) return null;
    const patch = typeof voiceData === 'string' ? base64ToUint8Array(voiceData) : voiceData;
    dx7Node.port.postMessage({ type: 'patch', slot: 0, data: patch.buffer });
    dx7Node.port.postMessage({ type: 'midi', data: [0x90, midi, velocity] });
    return () => {
        if (dx7Node) dx7Node.port.postMessage({ type: 'midi', data: [0x80, midi, 0] });
    };
}

// --- Offline rendering ---

export async function renderOffline(voiceData, midi, durationMs) {
    const sr = 44100;
    const renderDur = Math.max(durationMs, 500);
    const totalDur = (renderDur / 1000) + 0.5;
    const totalSamples = Math.ceil(totalDur * sr);

    const offCtx = new OfflineAudioContext(1, totalSamples, sr);

    try {
        const base = new URL('./wasm/', import.meta.url).pathname;
        await offCtx.audioWorklet.addModule(base + 'msfa.wasm.js');
        await offCtx.audioWorklet.addModule(new URL('./dx7-processor.js', import.meta.url));

        const node = new AudioWorkletNode(offCtx, 'dx7-synth', {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [1],
        });

        await new Promise((resolve) => {
            node.port.onmessage = (e) => {
                if (e.data.type === 'ready') resolve();
            };
            setTimeout(resolve, 2000);
        });

        const patch = typeof voiceData === 'string' ? base64ToUint8Array(voiceData) : voiceData;
        node.port.postMessage({ type: 'patch', slot: 0, data: patch.buffer });
        node.port.postMessage({ type: 'midi', data: [0x90, midi, 100] });

        node.connect(offCtx.destination);

        const rendered = await offCtx.startRendering();
        const samples = rendered.getChannelData(0);

        const outSamples = Math.min(Math.ceil((renderDur / 1000) * sr), samples.length);
        const result = new Float32Array(outSamples);
        result.set(samples.subarray(0, outSamples));

        // Fade-out over last 50ms
        const fadeLen = Math.min(Math.ceil(0.05 * sr), outSamples);
        for (let i = 0; i < fadeLen; i++) {
            result[outSamples - fadeLen + i] *= (1 - i / fadeLen);
        }

        return samplesToWavBlob(result, sr);
    } catch (err) {
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

// --- Shutdown hooks (disabled for testing) ---

export function setPianorollStop(fn) {
    // No-op
}

export async function shutdownAll() {
    // No-op
}

export function extendIdle() {
    // No-op
}
