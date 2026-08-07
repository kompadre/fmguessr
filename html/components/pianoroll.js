// Pianoroll: 3 octaves (C3–C6) × 1 measure of 4/4 (16 sixteenth notes)
// Click+drag creates notes, double-click removes, sequencer plays all notes.

const NOTES = [
    { midi: 84, name: 'C6',  white: true },
    { midi: 83, name: 'B5',  white: true },
    { midi: 82, name: 'A#5', white: false },
    { midi: 81, name: 'A5',  white: true },
    { midi: 80, name: 'G#5', white: false },
    { midi: 79, name: 'G5',  white: true },
    { midi: 78, name: 'F#5', white: false },
    { midi: 77, name: 'F5',  white: true },
    { midi: 76, name: 'E5',  white: true },
    { midi: 75, name: 'D#5', white: false },
    { midi: 74, name: 'D5',  white: true },
    { midi: 73, name: 'C#5', white: false },
    { midi: 72, name: 'C5',  white: true },
    { midi: 71, name: 'B4',  white: true },
    { midi: 70, name: 'A#4', white: false },
    { midi: 69, name: 'A4',  white: true },
    { midi: 68, name: 'G#4', white: false },
    { midi: 67, name: 'G4',  white: true },
    { midi: 66, name: 'F#4', white: false },
    { midi: 65, name: 'F4',  white: true },
    { midi: 64, name: 'E4',  white: true },
    { midi: 63, name: 'D#4', white: false },
    { midi: 62, name: 'D4',  white: true },
    { midi: 61, name: 'C#4', white: false },
    { midi: 60, name: 'C4',  white: true },
    { midi: 59, name: 'B3',  white: true },
    { midi: 58, name: 'A#3', white: false },
    { midi: 57, name: 'A3',  white: true },
    { midi: 56, name: 'G#3', white: false },
    { midi: 55, name: 'G3',  white: true },
    { midi: 54, name: 'F#3', white: false },
    { midi: 53, name: 'F3',  white: true },
    { midi: 52, name: 'E3',  white: true },
    { midi: 51, name: 'D#3', white: false },
    { midi: 50, name: 'D3',  white: true },
    { midi: 49, name: 'C#3', white: false },
    { midi: 48, name: 'C3',  white: true },
];

const COLS = 16; // 16 sixteenth notes = 1 measure of 4/4
const ROW_H = 20;
const COL_W = 30;
const LABEL_W = 36;
const CANVAS_W = LABEL_W + COLS * COL_W;
const CANVAS_H = NOTES.length * ROW_H;

const PATCH_COLORS = ['#e94560', '#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#00bcd4'];

export function createPianoroll(opts) {
    const { notes, playNoteFn, onNotesChange } = opts;

    const container = document.createElement('div');
    container.className = 'pianoroll-container';

    // Controls row
    const controls = document.createElement('div');
    controls.className = 'pianoroll-controls';

    const playBtn = document.createElement('button');
    playBtn.className = 'pianoroll-btn pianoroll-play';
    playBtn.textContent = '▶ Play';
    controls.appendChild(playBtn);

    const stopBtn = document.createElement('button');
    stopBtn.className = 'pianoroll-btn pianoroll-stop';
    stopBtn.textContent = '■ Stop';
    stopBtn.disabled = true;
    controls.appendChild(stopBtn);

    const clearBtn = document.createElement('button');
    clearBtn.className = 'pianoroll-btn';
    clearBtn.textContent = 'Clear';
    controls.appendChild(clearBtn);

    const label = document.createElement('span');
    label.className = 'pianoroll-hint';
    label.textContent = 'Click+drag to add notes · Double-click to remove';
    controls.appendChild(label);

    container.appendChild(controls);

    // Scrollable canvas area
    const scroll = document.createElement('div');
    scroll.className = 'pianoroll-scroll';

    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_W;
    canvas.height = CANVAS_H;
    canvas.className = 'pianoroll-canvas';
    scroll.appendChild(canvas);
    container.appendChild(scroll);

    // Legend
    const legend = document.createElement('div');
    legend.className = 'pianoroll-legend';
    container.appendChild(legend);

    const ctx = canvas.getContext('2d');
    let playing = false;
    let playTimer = null;
    let playCol = -1;
    let dragging = false;
    let dragStart = null; // { row, col }

    function colorForPatch(patchId) {
        if (!patchId) return PATCH_COLORS[0];
        let hash = 0;
        for (let i = 0; i < patchId.length; i++) {
            hash = ((hash << 5) - hash + patchId.charCodeAt(i)) | 0;
        }
        return PATCH_COLORS[Math.abs(hash) % PATCH_COLORS.length];
    }

    function draw() {
        ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

        // Row backgrounds
        for (let r = 0; r < NOTES.length; r++) {
            const y = r * ROW_H;
            ctx.fillStyle = NOTES[r].white ? '#1a1a2e' : '#12121f';
            ctx.fillRect(LABEL_W, y, COLS * COL_W, ROW_H);
        }

        // Column lines (beat boundaries thicker)
        for (let c = 0; c <= COLS; c++) {
            const x = LABEL_W + c * COL_W;
            ctx.strokeStyle = c % 4 === 0 ? '#444' : '#2a2a3a';
            ctx.lineWidth = c % 4 === 0 ? 1.5 : 0.5;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, CANVAS_H);
            ctx.stroke();
        }

        // Row lines
        for (let r = 0; r <= NOTES.length; r++) {
            const y = r * ROW_H;
            ctx.strokeStyle = '#2a2a3a';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(LABEL_W, y);
            ctx.lineTo(CANVAS_W, y);
            ctx.stroke();
        }

        // Note labels
        ctx.font = '10px monospace';
        ctx.textBaseline = 'middle';
        for (let r = 0; r < NOTES.length; r++) {
            const y = r * ROW_H + ROW_H / 2;
            ctx.fillStyle = NOTES[r].white ? '#aaa' : '#666';
            ctx.textAlign = 'right';
            ctx.fillText(NOTES[r].name, LABEL_W - 4, y);
        }

        // Notes
        for (const n of notes) {
            const r = NOTES.findIndex(nn => nn.midi === n.midi);
            if (r < 0) continue;
            const x = LABEL_W + n.start * COL_W;
            const y = r * ROW_H;
            const w = n.dur * COL_W - 1;
            ctx.fillStyle = n.color || PATCH_COLORS[0];
            ctx.globalAlpha = 0.85;
            ctx.fillRect(x + 1, y + 1, w, ROW_H - 2);
            ctx.globalAlpha = 1.0;
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(x + 1, y + 1, w, ROW_H - 2);
        }

        // Playback cursor
        if (playCol >= 0) {
            const x = LABEL_W + playCol * COL_W;
            ctx.fillStyle = 'rgba(255,255,255,0.15)';
            ctx.fillRect(x, 0, COL_W, CANVAS_H);
        }

        // Drag preview
        if (dragging && dragStart) {
            const cur = lastMousePos;
            if (cur) {
                const endCol = pixelToCol(cur.x);
                const endRow = pixelToRow(cur.y);
                if (endRow === dragStart.row) {
                    const minC = Math.min(dragStart.col, endCol);
                    const maxC = Math.max(dragStart.col, endCol);
                    const x = LABEL_W + minC * COL_W;
                    const w = (maxC - minC + 1) * COL_W;
                    ctx.fillStyle = 'rgba(255,255,255,0.1)';
                    ctx.fillRect(x, dragStart.row * ROW_H, w, ROW_H);
                }
            }
        }
    }

    function pixelToCol(px) {
        return Math.max(0, Math.min(COLS - 1, Math.floor((px - LABEL_W) / COL_W)));
    }

    function pixelToRow(py) {
        return Math.max(0, Math.min(NOTES.length - 1, Math.floor(py / ROW_H)));
    }

    function noteAt(row, col) {
        const midi = NOTES[row].midi;
        return notes.find(n => n.midi === midi && col >= n.start && col < n.start + n.dur);
    }

    let lastMousePos = null;
    let lastClickTime = 0;
    let lastClickRow = -1;
    let lastClickCol = -1;

    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const row = pixelToRow(y);
        const col = pixelToCol(x);

        if (col < 0) return;

        const now = Date.now();
        const existing = noteAt(row, col);

        // Double-click detection
        if (existing && now - lastClickTime < 350 && row === lastClickRow && Math.abs(col - lastClickCol) <= 1) {
            // Remove note
            const idx = notes.indexOf(existing);
            if (idx >= 0) notes.splice(idx, 1);
            draw();
            renderLegend();
            onNotesChange();
            lastClickTime = 0;
            return;
        }

        lastClickTime = now;
        lastClickRow = row;
        lastClickCol = col;

        if (existing) {
            playNoteFn(existing.midi, existing.dur * 250, existing.voiceData || null);
            return;
        }

        // Start drag to create note
        dragging = true;
        dragStart = { row, col };
        lastMousePos = { x, y };
        draw();
    });

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        lastMousePos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        if (dragging) draw();
    });

    canvas.addEventListener('mouseup', (e) => {
        if (!dragging || !dragStart) {
            dragging = false;
            return;
        }

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const endCol = pixelToCol(x);
        const endRow = pixelToRow(y);

        if (endRow === dragStart.row) {
            const minC = Math.min(dragStart.col, endCol);
            const maxC = Math.max(dragStart.col, endCol);
            const dur = maxC - minC + 1;
            const midi = NOTES[dragStart.row].midi;

            notes.push({
                midi,
                start: minC,
                dur,
                patchId: opts.activePatchId || null,
                voiceData: opts.activeVoiceData || null,
                color: opts.activeColor || colorForPatch(opts.activePatchId),
            });
            draw();
            renderLegend();
            onNotesChange();
        }

        dragging = false;
        dragStart = null;
        lastMousePos = null;
        draw();
    });

    canvas.addEventListener('mouseleave', () => {
        if (dragging) {
            dragging = false;
            dragStart = null;
            lastMousePos = null;
            draw();
        }
    });

    // Sequencer
    function startPlayback() {
        if (playing) stopPlayback();
        if (notes.length === 0) return;

        playing = true;
        playCol = 0;
        playBtn.disabled = true;
        stopBtn.disabled = false;

        const msPerSixteenth = 250; // 60 BPM → 60000/60/4 = 250ms

        function tick() {
            if (!playing) return;

            for (const n of notes) {
                if (n.start === playCol) {
                    playNoteFn(n.midi, n.dur * 250, n.voiceData || null);
                }
            }

            draw();

            const current = playCol;
            playCol = (current + 1) % COLS;

            playTimer = setTimeout(tick, msPerSixteenth);
        }

        draw();
        playTimer = setTimeout(tick, msPerSixteenth);
    }

    function stopPlayback() {
        playing = false;
        playCol = -1;
        if (playTimer) clearTimeout(playTimer);
        playTimer = null;
        playBtn.disabled = false;
        stopBtn.disabled = true;
        draw();
    }

    playBtn.addEventListener('click', startPlayback);
    stopBtn.addEventListener('click', stopPlayback);
    clearBtn.addEventListener('click', () => {
        notes.length = 0;
        draw();
        renderLegend();
        onNotesChange();
    });

    function renderLegend() {
        legend.innerHTML = '';
        const patchIds = [...new Set(notes.map(n => n.patchId).filter(Boolean))];
        if (patchIds.length === 0) {
            legend.style.display = 'none';
            return;
        }
        legend.style.display = '';
        for (const pid of patchIds) {
            const item = document.createElement('span');
            item.className = 'pianoroll-legend-item';
            const dot = document.createElement('span');
            dot.className = 'pianoroll-legend-dot';
            dot.style.background = colorForPatch(pid);
            item.appendChild(dot);
            const name = document.createElement('span');
            name.textContent = pid;
            item.appendChild(name);
            legend.appendChild(item);
        }
    }

    draw();

    return {
        element: container,
        draw,
        stopPlayback,
        addNote(midi, start, dur, patchId) {
            notes.push({ midi, start, dur, patchId, color: opts.activeColor || colorForPatch(patchId) });
            draw();
            renderLegend();
        },
        setActivePatch(patchId, voiceData, color) {
            opts.activePatchId = patchId;
            opts.activeVoiceData = voiceData;
            opts.activeColor = color || null;
        },
        colorForPatch,
    };
}
