// Patch card for uploaded .syx banks.
// Shows patch name + grayed spectrogram with play button.
// On play: sends patch to server, plays audio, reveals spectrogram.

export function createPatchCard(patch, { baseUrl, onFav, onAssign, onEdit, isFav }) {
    const card = document.createElement('div');
    card.className = 'patch-card';

    // Header: name + star + algo/fb
    const header = document.createElement('div');
    header.className = 'patch-card-header';

    const nameEl = document.createElement('span');
    nameEl.className = 'patch-card-name';
    nameEl.textContent = patch.name;
    nameEl.title = patch.name;
    header.appendChild(nameEl);

    const star = document.createElement('span');
    star.className = 'patch-star' + (isFav ? ' fav-active' : '');
    star.textContent = isFav ? '★' : '☆';
    star.title = 'Favorite';
    if (onFav) {
        star.addEventListener('click', (e) => {
            e.stopPropagation();
            onFav(patch);
        });
    }
    header.appendChild(star);

    const algoEl = document.createElement('span');
    algoEl.className = 'patch-col';
    algoEl.textContent = `A${patch.algorithm + 1}`;
    header.appendChild(algoEl);

    const fbEl = document.createElement('span');
    fbEl.className = 'patch-col';
    fbEl.textContent = `FB${patch.feedback}`;
    header.appendChild(fbEl);

    card.appendChild(header);

    // Spectrogram area (grayed out initially)
    const specWrap = document.createElement('div');
    specWrap.className = 'patch-spec';
    const canvas = document.createElement('canvas');
    canvas.width = 220;
    canvas.height = 56;
    canvas.className = 'patch-canvas';
    // Draw gray placeholder
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, 220, 56);
    ctx.fillStyle = '#333';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Not loaded', 110, 32);
    specWrap.appendChild(canvas);

    const playBtn = document.createElement('button');
    playBtn.className = 'patch-play';
    playBtn.textContent = '▶';
    playBtn.title = 'Load & play';
    specWrap.appendChild(playBtn);

    card.appendChild(specWrap);

    // Assign button
    const assignBtn = document.createElement('button');
    assignBtn.className = 'patch-assign';
    assignBtn.textContent = '♫';
    assignBtn.title = 'Assign to pianoroll';
    assignBtn.style.opacity = '0';
    card.appendChild(assignBtn);

    // Edit button
    const editBtn = document.createElement('button');
    editBtn.className = 'patch-edit-btn';
    editBtn.textContent = '🔧';
    editBtn.title = 'Edit patch';
    editBtn.style.opacity = '0';
    if (onEdit) {
        editBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            onEdit(patch);
        });
    }
    card.appendChild(editBtn);

    // Lazy-load state
    let loaded = false;
    let wavUrl = null;

    playBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (loaded && wavUrl) {
            new Audio(baseUrl + wavUrl).play();
            return;
        }

        playBtn.textContent = '⏳'; // hourglass
        playBtn.disabled = true;

        try {
            // Send patch to server for rendering
            const resp = await fetch(baseUrl + '/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    voice_data: btoa(String.fromCharCode.apply(null, patch.raw)),
                    note: 60,
                })
            });
            const data = await resp.json();

            if (data.error) {
                console.error('Play error:', data.error);
                playBtn.textContent = '▶';
                playBtn.disabled = false;
                return;
            }

            wavUrl = data.wav_url;
            loaded = true;

            // Play audio
            if (wavUrl) {
                new Audio(baseUrl + wavUrl).play();
            }

            // Fetch and draw spectrogram
            if (wavUrl) {
                const wavResp = await fetch(baseUrl + wavUrl);
                const wavBuf = await wavResp.arrayBuffer();
                const samples = decodeWavSamples(wavBuf);
                if (samples && samples.length > 0) {
                    drawMiniWaveform(canvas, samples);
                    // Remove grayed state
                    specWrap.classList.add('loaded');
                }
            }

            playBtn.textContent = '▶';
            playBtn.disabled = false;
        } catch (err) {
            console.warn('Play failed:', err);
            playBtn.textContent = '▶';
            playBtn.disabled = false;
        }
    });

    // Assign to pianoroll
    assignBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (onAssign) onAssign(patch);
    });

    // Hover to show action buttons
    card.addEventListener('mouseenter', () => {
        assignBtn.style.opacity = '1';
        editBtn.style.opacity = '1';
    });
    card.addEventListener('mouseleave', () => {
        assignBtn.style.opacity = '0';
        editBtn.style.opacity = '0';
    });

    return card;
}

// Minimal WAV decoder (same as in patch.js)
function decodeWavSamples(arrayBuffer) {
    try {
        const view = new DataView(arrayBuffer);
        if (view.getUint32(0, false) !== 0x52494646) return null;
        let offset = 12;
        while (offset < view.byteLength - 8) {
            const chunkId = view.getUint32(offset, false);
            const chunkSize = view.getUint32(offset + 4, true);
            if (chunkId === 0x666d7420) {
                const channels = view.getUint16(offset + 10, true);
                const bitsPerSample = view.getUint16(offset + 22, true);
                const dataChunkOffset = offset + 8 + chunkSize;
                const dataId = view.getUint32(dataChunkOffset, false);
                const dataSize = view.getUint32(dataChunkOffset + 4, true);
                if (dataId === 0x64617461) {
                    const dataStart = dataChunkOffset + 8;
                    const numSamples = Math.floor(dataSize / (bitsPerSample / 8));
                    const samples = new Float32Array(numSamples);
                    for (let i = 0; i < numSamples; i++) {
                        samples[i] = view.getInt16(dataStart + i * 2, true) / 32768.0;
                    }
                    if (channels === 2) {
                        const mono = new Float32Array(numSamples / 2);
                        for (let i = 0; i < mono.length; i++) {
                            mono[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2;
                        }
                        return mono;
                    }
                    return samples;
                }
                offset += 8 + chunkSize;
                if (chunkSize % 2 !== 0) offset++;
            } else {
                offset += 8 + chunkSize;
                if (chunkSize % 2 !== 0) offset++;
            }
        }
    } catch (e) {
        console.warn('WAV decode failed:', e);
    }
    return null;
}

function drawMiniWaveform(canvas, samples) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const mid = h / 2;

    ctx.fillStyle = '#0f0f23';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();

    const step = Math.max(1, Math.floor(samples.length / w));
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
        const abs = Math.abs(samples[i]);
        if (abs > peak) peak = abs;
    }
    if (peak < 0.001) peak = 1;

    ctx.beginPath();
    ctx.moveTo(0, mid);
    for (let x = 0; x < w; x++) {
        const idx = Math.floor(x * step);
        if (idx >= samples.length) break;
        const val = samples[idx] / peak;
        ctx.lineTo(x, mid - val * (mid - 2));
    }
    for (let x = w - 1; x >= 0; x--) {
        const idx = Math.floor(x * step);
        if (idx >= samples.length) continue;
        const val = samples[idx] / peak;
        ctx.lineTo(x, mid + val * (mid - 2));
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(76, 175, 80, 0.35)';
    ctx.fill();

    ctx.beginPath();
    for (let x = 0; x < w; x++) {
        const idx = Math.floor(x * step);
        if (idx >= samples.length) break;
        const val = samples[idx] / peak;
        const y = mid - val * (mid - 2);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 1;
    ctx.stroke();
}
