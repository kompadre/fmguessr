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

function createPatchRow(match, { onSelect, onStar, onPlay, onEdit, baseUrl, isSelected, isFav }) {
    const row = document.createElement('div');
    row.className = 'patch-row' + (isSelected ? ' selected' : '');

    // Mini spectrogram + play button
    const specWrap = document.createElement('div');
    specWrap.className = 'patch-spec';
    const canvas = document.createElement('canvas');
    canvas.width = 180;
    canvas.height = 48;
    canvas.className = 'patch-canvas';
    specWrap.appendChild(canvas);

    const playBtn = document.createElement('button');
    playBtn.className = 'patch-play';
    playBtn.textContent = '▶';
    playBtn.title = 'Play';
    specWrap.appendChild(playBtn);

    let waveformDrawn = !!match.wav_url;

    if (onPlay) {
        playBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const result = onPlay(match);
            if (!waveformDrawn && result && typeof result.then === 'function') {
                try {
                    const wavUrl = await result;
                    if (wavUrl && !waveformDrawn) {
                        waveformDrawn = true;
                        const wavResp = await fetch(wavUrl);
                        const wavBuf = await wavResp.arrayBuffer();
                        const samples = decodeWavSamples(wavBuf);
                        if (samples && samples.length > 0) drawMiniWaveform(canvas, samples);
                    }
                } catch (_) {}
            }
        });
    }

    row.appendChild(specWrap);

    // Fetch and render waveform
    const wavUrl = (baseUrl || '') + match.wav_url;
    if (match.wav_url) {
        fetch(wavUrl)
            .then(r => r.arrayBuffer())
            .then(buf => {
                const samples = decodeWavSamples(buf);
                if (samples && samples.length > 0) drawMiniWaveform(canvas, samples);
            })
            .catch(() => {});
    }

    // Name
    const nameEl = document.createElement('span');
    nameEl.className = 'patch-name';
    nameEl.textContent = match.name || ('#' + match.rank);
    nameEl.title = match.name || ('#' + match.rank);
    row.appendChild(nameEl);

    // Algo
    const algoEl = document.createElement('span');
    algoEl.className = 'patch-col patch-algo';
    algoEl.textContent = match.algorithm;
    row.appendChild(algoEl);

    // FB
    const fbEl = document.createElement('span');
    fbEl.className = 'patch-col patch-fb';
    fbEl.textContent = match.feedback;
    row.appendChild(fbEl);

    // Dist
    const distEl = document.createElement('span');
    distEl.className = 'patch-col patch-dist';
    distEl.textContent = match.distance.toFixed(4);
    row.appendChild(distEl);

    // Assign to pianoroll
    const assignBtn = document.createElement('button');
    assignBtn.className = 'patch-assign';
    assignBtn.textContent = '♫';
    assignBtn.title = 'Assign to pianoroll';
    assignBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (onSelect) onSelect(match);
        const dock = document.getElementById('piano-dock');
        if (dock) dock.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
    row.appendChild(assignBtn);

    // Star (with other action buttons)
    const star = document.createElement('button');
    star.className = 'patch-star' + (isFav ? ' fav-active' : '');
    star.textContent = isFav ? '★' : '☆';
    star.title = 'Favorite';
    if (onStar) {
        star.addEventListener('click', (e) => {
            e.stopPropagation();
            const isNowFav = onStar(match);
            star.textContent = isNowFav ? '★' : '☆';
            star.classList.toggle('fav-active', isNowFav);
        });
    }
    row.appendChild(star);

    // Edit button
    if (onEdit) {
        const editBtn = document.createElement('button');
        editBtn.className = 'patch-edit-btn';
        editBtn.textContent = '🔧';
        editBtn.title = 'Edit patch';
        editBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            onEdit(match);
        });
        row.appendChild(editBtn);
    }

    // Click row → select
    row.addEventListener('click', () => {
        if (onSelect) onSelect(match);
    });

    // Store match data for highlighting
    row.dataset.algo = match.algorithm;
    row.dataset.fb = match.feedback;

    return row;
}

function createPatchList(matches, { title, subtitle, onSelect, onStar, onPlay, onEdit, baseUrl, selectedMatch, favIds }) {
    const container = document.createElement('div');
    container.className = 'patch-list';

    if (title) {
        const h = document.createElement('h3');
        h.className = 'section-title';
        h.textContent = title;
        container.appendChild(h);
    }
    if (subtitle) {
        const p = document.createElement('p');
        p.className = 'section-subtitle';
        p.textContent = subtitle;
        container.appendChild(p);
    }

    const header = document.createElement('div');
    header.className = 'patch-list-header';
    header.innerHTML = '<span class="patch-spec"></span><span class="patch-name">Name</span><span class="patch-col">Algo</span><span class="patch-col">FB</span><span class="patch-col">Dist</span>';
    container.appendChild(header);

    const body = document.createElement('div');
    body.className = 'patch-list-body';

    const favSet = new Set(favIds || []);
    (matches || []).forEach(m => {
        const isSelected = selectedMatch && selectedMatch.algorithm === m.algorithm && selectedMatch.feedback === m.feedback;
        body.appendChild(createPatchRow(m, {
            onSelect,
            onStar,
            onPlay,
            onEdit,
            baseUrl,
            isSelected,
            isFav: favSet.has(m.algorithm + '-' + m.feedback),
        }));
    });

    container.appendChild(body);
    return container;
}

function highlightPatchRow(container, match) {
    if (!container) return;
    container.querySelectorAll('.patch-row.selected').forEach(r => r.classList.remove('selected'));
    if (!match) return;
    container.querySelectorAll('.patch-row').forEach(row => {
        if (row.dataset.algo == match.algorithm && row.dataset.fb == match.feedback) {
            row.classList.add('selected');
        }
    });
}

export { createPatchRow, createPatchList, highlightPatchRow, decodeWavSamples, drawMiniWaveform };
