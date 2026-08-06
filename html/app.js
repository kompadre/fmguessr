import { createPatchRow, createPatchList, highlightPatchRow, decodeWavSamples, drawMiniWaveform, makeDropTarget } from './components/patch.js';
import { createPianoDock } from './components/piano-dock.js';
import { createSpectrogram } from './components/spectrogram.js';
import { parseSyxFile, voiceRawToBase64 } from './components/syx-parser.js';
import { initDx7Synth, playNoteRealtime, renderOffline, isDx7Ready, batchRenderWaveforms } from './synth/dx7-synth.js';
import { initNightSky, addContainer } from './components/night-sky.js';

const fileInput = document.getElementById('wav-input');
const log = document.getElementById('log');
const tabsEl = document.getElementById('tabs');
const tabContent = document.getElementById('tab-content');
const importedPanel = document.getElementById('imported-panel');

function showImportedPanel() {
    importedPanel.classList.add('visible');
}

// Imported panel: minimize toggle
document.getElementById('imported-header').addEventListener('click', (e) => {
    if (e.target.closest('.config-arrow') || e.currentTarget === e.target) {
        importedPanel.classList.toggle('minimized');
    }
});

// Imported panel: drag by header
(() => {
    const header = document.getElementById('imported-header');
    let dragging = false, startX, startY, origLeft, origTop;
    header.addEventListener('mousedown', (e) => {
        if (e.target.closest('.config-arrow')) return;
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
        const rect = importedPanel.getBoundingClientRect();
        origLeft = rect.left;
        origTop = rect.top;
        importedPanel.style.position = 'fixed';
        importedPanel.style.left = origLeft + 'px';
        importedPanel.style.top = origTop + 'px';
        importedPanel.style.zIndex = 1000;
        e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        importedPanel.style.left = (origLeft + e.clientX - startX) + 'px';
        importedPanel.style.top = (origTop + e.clientY - startY) + 'px';
    });
    document.addEventListener('mouseup', () => { dragging = false; });
})();
const baseUrl = window.location.origin.indexOf("localhost") > -1 ? "http://localhost:8080" : window.location.origin;

let pollInterval = null;
let jobs = [];
let activeJobIndex = -1;
let pollQueue = [];
let activeSyxVoiceData = null; // voice_data of currently selected .syx patch for piano
let activePianorollPatchId = null; // patch name/id for pianoroll note coloring
let activeKeyboardNote = 60; // last-clicked keyboard note (used for detached patch playback)
let pianorollNotes = []; // [{midi, start, dur, patchId, color}]
let pianorollInstance = null; // current pianoroll component instance

const probeFields = ['attack_ms', 'decay_ms', 'sustain_level', 'release_ms', 'gate_ms', 'brightness', 'harmonicity'];

// --- Session Persistence (localStorage + IndexedDB) ---
const STORAGE_KEY = 'fmguessr_sessions';
const ACTIVE_KEY = 'fmguessr_active';
const SYX_KEY = 'fmguessr_syx_tabs';
const FAVORITES_KEY = 'fmguessr_favorites';
const PATCH_NAMES_KEY = 'fmguessr_patch_names';
const MAX_SESSIONS = 50;
const DB_NAME = 'fmguessr';
const DB_VERSION = 1;
const STORE_NAME = 'uploads';

const dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => e.target.result.createObjectStore(STORE_NAME);
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
});

async function dbPut(key, value) {
    const db = await dbPromise;
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).put(value, key);
}

async function dbGet(key) {
    const db = await dbPromise;
    const tx = db.transaction(STORE_NAME, 'readonly');
    return new Promise((resolve, reject) => {
        const req = tx.objectStore(STORE_NAME).get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function dbDelete(key) {
    const db = await dbPromise;
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).delete(key);
}

async function dbClear() {
    const db = await dbPromise;
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).clear();
}

function saveSessions() {
    const data = jobs.map(j => ({
        id: j.id,
        fileName: j.fileName,
        statusUrl: j.statusUrl,
        probeData: j.probeData,
        resultData: j.resultData,
        selectedMatch: j.selectedMatch
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    localStorage.setItem(ACTIVE_KEY, String(activeJobIndex));
    updateClearBtn();
}

function updateClearBtn() {
    const btn = document.getElementById('clear-btn');
    if (btn) btn.style.display = jobs.length > 0 ? 'inline-block' : 'none';
}

async function restoreSessions() {
    let saved;
    try {
        saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch { saved = []; }
    const activeIdx = parseInt(localStorage.getItem(ACTIVE_KEY) || '-1');

    const inProgressJobs = [];

    for (const s of saved) {
        const arrayBuffer = await dbGet(s.id);
        if (!arrayBuffer) continue;

        const blob = new Blob([arrayBuffer]);
        const file = new File([blob], s.fileName, { type: 'audio/wav' });
        const objectUrl = URL.createObjectURL(blob);

        const job = {
            id: s.id,
            file, objectUrl,
            fileName: s.fileName,
            probeData: s.probeData,
            resultData: s.resultData,
            statusUrl: s.statusUrl || null,
            selectedMatch: s.selectedMatch || null,
            tabEl: null,
            pollCount: 0
        };
        jobs.push(job);
        const idx = jobs.length - 1;
        addTab(idx);
        if (job.resultData) {
            upgradeTab(idx);
        } else if (job.statusUrl) {
            // Job was in-progress when page was unloaded — resume polling
            inProgressJobs.push(idx);
        } else if (job.probeData) {
            // Probe completed but match never started — remove spinner
            upgradeTab(idx);
        } else {
            // Probe was in-flight — re-trigger it
            reprobeJob(idx);
        }
    }

    if (activeIdx >= 0 && activeIdx < jobs.length) {
        showJob(activeIdx);
    }
    updateClearBtn();

    // Resume polling for in-progress jobs
    if (inProgressJobs.length > 0) {
        for (const idx of inProgressJobs) {
            pollQueue.push(idx);
        }
        appendLog(`[restore] Resuming ${inProgressJobs.length} in-progress job(s)`);
        startPolling();
    }
}

async function deleteSession(jobIndex) {
    const job = jobs[jobIndex];
    if (!job) return;

    if (!confirm(`Remove "${job.fileName}" from history?`)) return;

    await dbDelete(job.id);
    if (job.objectUrl) URL.revokeObjectURL(job.objectUrl);
    jobs.splice(jobIndex, 1);

    // Remove only this job's content div and tab header
    const contentEl = tabContent.querySelector('#tab-wav-' + jobIndex);
    if (contentEl) contentEl.remove();
    if (job.tabEl) job.tabEl.remove();

    // Re-index remaining WAV tabs (IDs shift after splice)
    tabContent.querySelectorAll('[id^="tab-wav-"]').forEach(el => el.remove());
    tabsEl.querySelectorAll('[data-index]').forEach(el => el.remove());
    jobs.forEach((j, i) => {
        j.tabEl = null;
        addTab(i);
        if (j.resultData) upgradeTab(i);
    });

    if (activeJobIndex === jobIndex) {
        activeJobIndex = jobs.length > 0 ? 0 : -1;
        if (activeJobIndex >= 0) {
            showJob(0);
        } else {
            // No WAV tabs left — show first .syx tab if any
            const firstSyx = tabsEl.querySelector('[data-tab-id]');
            if (firstSyx) {
                switchTab(firstSyx.dataset.tabId);
            } else {
                tabContent.classList.remove('visible');
                document.getElementById('piano-dock').innerHTML = '';
            }
        }
    } else if (activeJobIndex > jobIndex) {
        activeJobIndex--;
    }
    saveSessions();
}

async function clearAllSessions() {
    jobs = [];
    activeJobIndex = -1;
    stopPolling();
    pollQueue = [];
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(ACTIVE_KEY);
    localStorage.removeItem(SYX_KEY);
    favorites = [];
    localStorage.removeItem(FAVORITES_KEY);
    await dbClear();
    tabsEl.innerHTML = '';
    tabsEl.classList.remove('visible');
    tabContent.innerHTML = '';
    tabContent.classList.remove('visible');
    importedPanel.classList.remove('visible');
    document.getElementById('piano-dock').innerHTML = '';
    if (pianorollInstance) pianorollInstance.stopPlayback();
    pianorollInstance = null;
    pianorollNotes.length = 0;
    renderFavorites();
    updateClearBtn();
}

function rebuildPianoDock(midiNote, job) {
    const pianoDock = document.getElementById('piano-dock');
    if (pianorollInstance) pianorollInstance.stopPlayback();
    pianoDock.innerHTML = '';

    const playFn = (midi, durationMs, voiceData) => playNote(midi, job || null, voiceData || activeSyxVoiceData, durationMs);

    const dock = createPianoDock({
        matchedMidi: activeKeyboardNote,
        playNoteFn: playFn,
        pianorollNotes,
        activePatchId: activePianorollPatchId,
        activeVoiceData: activeSyxVoiceData,
        onNoteClick: (midi) => { activeKeyboardNote = midi; },
    });
    pianoDock.appendChild(dock);
    pianorollInstance = dock._pianoroll || null;
}

// --- .syx Tab Persistence ---

let syxTabs = []; // { tabId, name, patches }

function deleteSyxTab(tabId) {
    if (!confirm('Remove this .syx tab?')) return;

    // Remove tab header
    const tabEl = tabsEl.querySelector(`[data-tab-id="${tabId}"]`);
    if (tabEl) tabEl.remove();

    // Remove tab content
    const contentEl = tabContent.querySelector('#tab-' + tabId);
    if (contentEl) contentEl.remove();

    // If this was the active tab, show something else
    if (activeJobIndex === null) {
        const firstTab = tabsEl.querySelector('.tab');
        if (firstTab) {
            if (firstTab.dataset.tabId) switchTab(firstTab.dataset.tabId);
            else if (firstTab.dataset.index !== undefined) showJob(parseInt(firstTab.dataset.index));
        } else {
            tabContent.classList.remove('visible');
            document.getElementById('piano-dock').innerHTML = '';
        }
    }

    saveSyxTabs();
}

function saveSyxTabs() {
    const data = [];
    tabContent.querySelectorAll('.tab-content').forEach(tc => {
        if (tc.id && tc.id.startsWith('tab-syx_')) {
            const tabId = tc.id.replace('tab-', '');
            const heading = tc.querySelector('h2');
            const name = heading ? heading.textContent : tabId;
            const tab = tabsEl.querySelector(`[data-tab-id="${tabId}"]`);
            if (tab) {
                // Collect patch data from the row elements
                const patches = [];
                tc.querySelectorAll('.patch-row').forEach(row => {
                    const algo = parseInt(row.dataset.algo) || 0;
                    const fb = parseInt(row.dataset.fb) || 0;
                    const nameEl = row.querySelector('.patch-name');
                    const patchName = nameEl ? nameEl.textContent : '';
                    // Retrieve stored voice_data from the row
                    const voiceData = row.dataset.voiceData || null;
                    if (voiceData) {
                        patches.push({ id: row.dataset.patchId || null, name: patchName, algorithm: algo, feedback: fb, voice_data: voiceData });
                    }
                });
                data.push({ tabId, name, patches });
            }
        }
    });
    localStorage.setItem(SYX_KEY, JSON.stringify(data));
}

function restoreSyxTabs() {
    let saved;
    try {
        saved = JSON.parse(localStorage.getItem(SYX_KEY) || '[]');
    } catch { saved = []; }

    const patchNames = loadPatchNames();

    for (const s of saved) {
        if (s.patches && s.patches.length > 0) {
            // Fully restore: rebuild the tab from saved patch data
            const patches = s.patches.map((p, i) => ({
                slot: i,
                id: p.id || patchId({ voice_data: p.voice_data }),
                name: (p.id && patchNames[p.id]) || p.name,
                algorithm: p.algorithm,
                feedback: p.feedback,
                raw: null,
                voice_data: p.voice_data,
            }));
            createSyxTabFromData(s.tabId, s.name, patches);
        } else {
            // Stub tab — no patch data saved
            const tab = document.createElement('div');
            tab.className = 'tab';
            tab.dataset.tabId = s.tabId;
            tab.textContent = s.name;
            tab.addEventListener('click', () => switchTab(s.tabId));
            tabsEl.appendChild(tab);

            const content = document.createElement('div');
            content.className = 'tab-content';
            content.id = 'tab-' + s.tabId;
            const header = document.createElement('div');
            header.className = 'upload-tab-header';
            const heading = document.createElement('h2');
            heading.textContent = s.name;
            header.appendChild(heading);
            content.appendChild(header);

            const msg = document.createElement('p');
            msg.style.cssText = 'color:#888;text-align:center;padding:1rem;';
            msg.textContent = 'Re-import the .syx file to restore patches.';
            content.appendChild(msg);

            tabContent.appendChild(content);
        }
    }
}

// --- Patch Identity ---

// DJB2 hash of a string — stable, fast, good distribution
function hashString(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) + hash) + str.charCodeAt(i);
        hash = hash & hash; // Convert to 32-bit int
    }
    return (hash >>> 0).toString(36);
}

// Compute a stable ID for a patch from its raw bytes or voice_data
function patchId(patch) {
    if (patch.raw) {
        return hashString(String.fromCharCode.apply(null, patch.raw));
    }
    if (patch.voice_data) {
        return hashString(patch.voice_data);
    }
    // Fallback: algo+fb+name — less stable but still useful
    return hashString(`${patch.algorithm}_${patch.feedback}_${patch.name || ''}`);
}

// --- Patch Rename Persistence ---

function loadPatchNames() {
    try {
        return JSON.parse(localStorage.getItem(PATCH_NAMES_KEY) || '{}');
    } catch { return {}; }
}

function savePatchNames(names) {
    localStorage.setItem(PATCH_NAMES_KEY, JSON.stringify(names));
}

// Add inline rename handlers to a patch row's .patch-name element.
// `obj` is the data object whose `.name` property to update on rename.
// `saveFn` is called after a successful rename to persist.
function addRenameHandlers(row, obj, saveFn) {
    const nameEl = row.querySelector('.patch-name');
    if (!nameEl) return;
    const originalName = obj.name;
    nameEl.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        nameEl.contentEditable = 'true';
        nameEl.focus();
        const range = document.createRange();
        range.selectNodeContents(nameEl);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
    });
    nameEl.addEventListener('blur', () => {
        nameEl.contentEditable = 'false';
        const newName = nameEl.textContent.trim();
        if (newName && newName !== obj.name) {
            obj.name = newName;
            saveFn();
        }
    });
    nameEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            nameEl.blur();
        }
        if (e.key === 'Escape') {
            nameEl.textContent = obj.name;
            nameEl.contentEditable = 'false';
        }
    });
}

// --- Favorites Management ---
let favorites = [];

function loadFavorites() {
    try {
        favorites = JSON.parse(localStorage.getItem(FAVORITES_KEY) || '[]');
    } catch { favorites = []; }
    renderFavorites();
}

function saveFavoritesToStorage() {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites));
}

async function addFavorite(job, match) {
    if (!job.resultData || !job.resultData.syx_url) {
        appendLog('Cannot favorite: no .syx bank available');
        return;
    }

    // Deduplicate by patch data hash
    const pid = patchId({ voice_data: match.voice_data, algorithm: match.algorithm, feedback: match.feedback });
    const existing = favorites.find(f => f.patch_hash === pid);
    if (existing) {
        appendLog(`Already favorited: ${existing.name}`);
        return;
    }

    const fav = {
        id: 'fav_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
        patch_hash: pid,
        name: job.fileName.replace(/\.[^.]+$/, '') + ' #' + match.rank,
        syx_url: job.resultData.syx_url,
        slot: match.rank - 1,
        algorithm: match.algorithm,
        feedback: match.feedback,
        distance: match.distance,
        wav_url: match.wav_url,
        voice_data: match.voice_data || null,
        source_job: job.fileName,
        created: Date.now()
    };

    favorites.push(fav);
    saveFavoritesToStorage();
    renderFavorites();
    appendLog(`Favorited: ${fav.name} (algo ${match.algorithm} fb ${match.feedback})`);
}

function removeFavorite(favId) {
    favorites = favorites.filter(f => f.id !== favId);
    saveFavoritesToStorage();
    renderFavorites();
    // Clear from edit slots if present
    if (editSlot?.id === favId) editSlot = null;
    if (editSlotB?.id === favId) editSlotB = null;
    currentMutation = null;
    renderEditPanel();
}

// --- Edit Panel (Morph / Params / Random) ---
let editSlot = null;       // The patch currently loaded for editing
let editMode = 'morph';    // 'morph' | 'params' | 'random'
let editSlotB = null;      // Second slot for morph mode only
let currentMutation = null; // Last mutation/tweak result (unsaved)

function openEdit(fav) {
    // In morph mode, if slot A is already filled, fill slot B instead of replacing
    if (editMode === 'morph' && editSlot && fav) {
        // Don't replace slot A with the same patch
        if (editSlot.id !== fav.id) {
            editSlotB = fav;
        }
    } else {
        editSlot = fav;
        editSlotB = null;
    }
    currentMutation = null;
    const panel = document.getElementById('edit-panel');
    panel.classList.add('visible');
    renderEditPanel();
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setEditMode(mode) {
    editMode = mode;
    currentMutation = null;
    document.querySelectorAll('.edit-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    renderEditPanel();
}

function renderEditPanel() {
    const slotArea = document.getElementById('edit-slot-area');
    const sliderRow = document.getElementById('edit-slider-row');
    const paramsArea = document.getElementById('edit-params-area');
    const goBtn = document.getElementById('edit-go');
    const status = document.getElementById('edit-status');

    if (!editSlot && !editSlotB) {
        slotArea.innerHTML = '';
        sliderRow.style.display = 'none';
        paramsArea.style.display = 'none';
        goBtn.disabled = true;
        goBtn.textContent = 'Morph';
        status.textContent = 'Click 🔧 on any patch to edit it';
        document.getElementById('edit-panel').classList.remove('visible');
        return;
    }

    if (editMode === 'morph') {
        renderMorphMode(slotArea, sliderRow, paramsArea, goBtn, status);
    } else if (editMode === 'params') {
        renderParamsMode(slotArea, sliderRow, paramsArea, goBtn, status);
    } else if (editMode === 'random') {
        renderRandomMode(slotArea, sliderRow, paramsArea, goBtn, status);
    }
}

function renderEditSlot(container, fav, slotIdx, showRemove) {
    container.innerHTML = '';
    if (!fav) {
        container.innerHTML = '<span class="edit-slot-empty">Click 🔧 on any patch</span>';
        container.classList.remove('filled');
        return;
    }
    container.classList.add('filled');

    const nameEl = document.createElement('div');
    nameEl.className = 'edit-slot-name';
    nameEl.textContent = fav.name;
    container.appendChild(nameEl);

    const infoEl = document.createElement('div');
    infoEl.className = 'edit-slot-info';
    infoEl.textContent = `A${fav.algorithm} FB${fav.feedback}`;
    container.appendChild(infoEl);

    if (showRemove) {
        const removeBtn = document.createElement('button');
        removeBtn.className = 'edit-slot-remove';
        removeBtn.textContent = '×';
        removeBtn.title = 'Remove';
        removeBtn.addEventListener('click', () => {
            if (slotIdx === 0) { editSlot = null; } else { editSlotB = null; }
            renderEditPanel();
        });
        container.appendChild(removeBtn);
    }
}

// --- Morph Mode ---
function renderMorphMode(slotArea, sliderRow, paramsArea, goBtn, status) {
    sliderRow.style.display = '';
    paramsArea.style.display = 'none';

    // Configure slider for morph
    const slider = document.getElementById('edit-slider');
    const sliderVal = document.getElementById('edit-slider-val');
    const sliderLabel = document.getElementById('edit-slider-label');
    slider.min = '-1'; slider.max = '1'; slider.step = '0.01';
    if (!slider.dataset.morphInit) { slider.value = '0.5'; slider.dataset.morphInit = '1'; }
    sliderVal.textContent = parseFloat(slider.value).toFixed(2);
    sliderLabel.innerHTML = 'Distance: <span id="edit-slider-val">' + parseFloat(slider.value).toFixed(2) + '</span>';

    // Dual slot layout
    slotArea.innerHTML = '';
    const slotsDiv = document.createElement('div');
    slotsDiv.className = 'edit-slots';

    const slotA = document.createElement('div');
    slotA.className = 'edit-slot';
    slotA.id = 'edit-slot-a';
    const labelA = document.createElement('div');
    labelA.className = 'edit-slot-label';
    labelA.textContent = 'A';
    const contentA = document.createElement('div');
    contentA.className = 'edit-slot-content';
    slotA.appendChild(labelA);
    slotA.appendChild(contentA);
    renderEditSlot(contentA, editSlot, 0, true);

    const swapBtn = document.createElement('button');
    swapBtn.className = 'edit-swap-btn';
    swapBtn.textContent = '⇄';
    swapBtn.title = 'Swap A ↔ B';
    swapBtn.addEventListener('click', () => {
        if (!editSlot || !editSlotB) return;
        [editSlot, editSlotB] = [editSlotB, editSlot];
        renderEditPanel();
    });

    const slotB = document.createElement('div');
    slotB.className = 'edit-slot';
    slotB.id = 'edit-slot-b';
    const labelB = document.createElement('div');
    labelB.className = 'edit-slot-label';
    labelB.textContent = 'B';
    const contentB = document.createElement('div');
    contentB.className = 'edit-slot-content';
    slotB.appendChild(labelB);
    slotB.appendChild(contentB);
    renderEditSlot(contentB, editSlotB, 1, true);

    slotsDiv.appendChild(slotA);
    slotsDiv.appendChild(swapBtn);
    slotsDiv.appendChild(slotB);
    slotArea.appendChild(slotsDiv);

    goBtn.disabled = !(editSlot && editSlotB);
    goBtn.textContent = 'Morph';

    if (editSlot && editSlotB) {
        const r = parseFloat(slider.value);
        status.textContent = `Ready: ${editSlot.name} ${ratioLabel(r)} ${editSlotB.name}`;
    } else if (editSlot) {
        status.textContent = `${editSlot.name} — add a target (B)`;
    } else {
        status.textContent = '';
    }
}

// --- Params Mode ---
function renderParamsMode(slotArea, sliderRow, paramsArea, goBtn, status) {
    sliderRow.style.display = 'none';
    paramsArea.style.display = '';

    // Single slot
    slotArea.innerHTML = '';
    const slotsDiv = document.createElement('div');
    slotsDiv.className = 'edit-slots';
    const slotA = document.createElement('div');
    slotA.className = 'edit-slot';
    const labelA = document.createElement('div');
    labelA.className = 'edit-slot-label';
    labelA.textContent = 'A';
    const contentA = document.createElement('div');
    contentA.className = 'edit-slot-content';
    slotA.appendChild(labelA);
    slotA.appendChild(contentA);
    renderEditSlot(contentA, editSlot, 0, true);
    slotsDiv.appendChild(slotA);
    slotArea.appendChild(slotsDiv);

    // Build param sliders from editSlot (or currentMutation)
    if (!currentMutation) {
        // Initialize mutation from slot
        currentMutation = JSON.parse(JSON.stringify(editSlot));
        // If the slot has voice_data, decode it to get actual param values
        if (editSlot.voice_data && !editSlot.ops) {
            try {
                const bytes = atob(editSlot.voice_data);
                const arr = new Uint8Array(bytes.length);
                for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
                currentMutation.ops = [];
                for (let i = 0; i < 6; i++) {
                    const base = i * 17;
                    currentMutation.ops.push({
                        outputLevel: arr[base + 14],
                        freqCoarse: (arr[base + 15] >> 1) & 0x1F,
                    });
                }
                currentMutation.feedback = arr[111] & 0x07;
                currentMutation.algorithm = arr[110];
            } catch {
                currentMutation.ops = Array.from({length: 6}, () => ({
                    outputLevel: 0, freqCoarse: 1
                }));
                currentMutation.feedback = editSlot.feedback || 0;
            }
        }
        if (!currentMutation.ops) {
            currentMutation.ops = Array.from({length: 6}, () => ({
                outputLevel: 0, freqCoarse: 1
            }));
        }
    }

    paramsArea.innerHTML = '';
    const grid = document.createElement('div');
    grid.className = 'edit-params-grid';

    // Per-operator sliders: OutputLevel + FreqCoarse
    for (let i = 0; i < 6; i++) {
        const op = currentMutation.ops[i] || { outputLevel: 0, freqCoarse: 1 };

        // OutputLevel
        const rowOL = document.createElement('div');
        rowOL.className = 'edit-param-row';
        const lblOL = document.createElement('span');
        lblOL.className = 'edit-param-label';
        lblOL.innerHTML = `Op${6-i} <small>Level</small>`;
        rowOL.appendChild(lblOL);
        const slOL = document.createElement('input');
        slOL.type = 'range'; slOL.min = '0'; slOL.max = '99'; slOL.value = String(op.outputLevel);
        slOL.className = 'edit-param-slider';
        const valOL = document.createElement('span');
        valOL.className = 'edit-param-val';
        valOL.textContent = op.outputLevel;
        slOL.addEventListener('input', () => {
            currentMutation.ops[i].outputLevel = parseInt(slOL.value);
            valOL.textContent = slOL.value;
        });
        rowOL.appendChild(slOL);
        rowOL.appendChild(valOL);
        grid.appendChild(rowOL);

        // FreqCoarse
        const rowFC = document.createElement('div');
        rowFC.className = 'edit-param-row';
        const lblFC = document.createElement('span');
        lblFC.className = 'edit-param-label';
        lblFC.innerHTML = ` <small>Freq</small>`;
        rowFC.appendChild(lblFC);
        const slFC = document.createElement('input');
        slFC.type = 'range'; slFC.min = '0'; slFC.max = '31'; slFC.value = String(op.freqCoarse);
        slFC.className = 'edit-param-slider';
        const valFC = document.createElement('span');
        valFC.className = 'edit-param-val';
        valFC.textContent = op.freqCoarse;
        slFC.addEventListener('input', () => {
            currentMutation.ops[i].freqCoarse = parseInt(slFC.value);
            valFC.textContent = slFC.value;
        });
        rowFC.appendChild(slFC);
        rowFC.appendChild(valFC);
        grid.appendChild(rowFC);

        if (i < 5) {
            const sep = document.createElement('hr');
            sep.className = 'edit-param-sep';
            grid.appendChild(sep);
        }
    }

    // Feedback
    const fbRow = document.createElement('div');
    fbRow.className = 'edit-feedback-row';
    const fbLbl = document.createElement('span');
    fbLbl.className = 'edit-feedback-label';
    fbLbl.textContent = 'Feedback';
    fbRow.appendChild(fbLbl);
    const fbSl = document.createElement('input');
    fbSl.type = 'range'; fbSl.min = '0'; fbSl.max = '7';
    fbSl.value = String(currentMutation.feedback || 0);
    fbSl.className = 'edit-param-slider';
    const fbVal = document.createElement('span');
    fbVal.className = 'edit-param-val';
    fbVal.textContent = currentMutation.feedback || 0;
    fbSl.addEventListener('input', () => {
        currentMutation.feedback = parseInt(fbSl.value);
        fbVal.textContent = fbSl.value;
    });
    fbRow.appendChild(fbSl);
    fbRow.appendChild(fbVal);
    grid.appendChild(fbRow);

    paramsArea.appendChild(grid);

    // Play preview button
    const playBtn = document.createElement('button');
    playBtn.textContent = '▶ Play';
    playBtn.className = 'edit-play-btn';
    playBtn.type = 'button';
    playBtn.addEventListener('click', previewParamsEdit);
    paramsArea.appendChild(playBtn);

    goBtn.disabled = false;
    goBtn.textContent = 'Save as new';
    status.textContent = `Editing ${editSlot.name} — tweak sliders, then save`;
}

// --- Random Mode ---
function renderRandomMode(slotArea, sliderRow, paramsArea, goBtn, status) {
    sliderRow.style.display = '';
    paramsArea.style.display = 'none';

    // Configure slider for mutation intensity
    const slider = document.getElementById('edit-slider');
    const sliderVal = document.getElementById('edit-slider-val');
    const sliderLabel = document.getElementById('edit-slider-label');
    slider.min = '0'; slider.max = '100'; slider.step = '1';
    if (!slider.dataset.randomInit) { slider.value = '30'; slider.dataset.randomInit = '1'; }
    sliderVal.textContent = slider.value;
    sliderLabel.innerHTML = 'Mutation: <span id="edit-slider-val">' + slider.value + '</span>%';

    // Single slot
    slotArea.innerHTML = '';
    const slotsDiv = document.createElement('div');
    slotsDiv.className = 'edit-slots';
    const slotA = document.createElement('div');
    slotA.className = 'edit-slot';
    const labelA = document.createElement('div');
    labelA.className = 'edit-slot-label';
    labelA.textContent = 'A';
    const contentA = document.createElement('div');
    contentA.className = 'edit-slot-content';
    slotA.appendChild(labelA);
    slotA.appendChild(contentA);
    renderEditSlot(contentA, editSlot, 0, true);
    slotsDiv.appendChild(slotA);
    slotArea.appendChild(slotsDiv);

    goBtn.disabled = false;
    goBtn.textContent = currentMutation ? 'Save as new' : 'Mutate';
    status.textContent = currentMutation
        ? `Mutation ready — save or mutate again`
        : `${editSlot.name} — set intensity, then mutate`;
}

function ratioLabel(r) {
    if (r < 0) return `← ${Math.abs(r).toFixed(2)}`;
    if (r === 0) return '↔ 0';
    return `→ ${r.toFixed(2)}`;
}

function morphName(nameA, nameB) {
    let a = '?';
    for (const ch of nameA) {
        if (/[A-Za-z0-9]/.test(ch)) { a = ch.toUpperCase(); break; }
    }
    let b = '?';
    for (const ch of nameB) {
        if (/[A-Za-z0-9]/.test(ch)) { b = ch.toUpperCase(); break; }
    }
    return a + '_' + b;
}

async function runEditAction() {
    if (editMode === 'morph') {
        await runMorphEdit();
    } else if (editMode === 'params') {
        await saveParamsEdit();
    } else if (editMode === 'random') {
        if (currentMutation) {
            await saveMutation();
        } else {
            await runRandomMutate();
        }
    }
}

async function runMorphEdit() {
    if (!editSlot || !editSlotB) return;

    const goBtn = document.getElementById('edit-go');
    const status = document.getElementById('edit-status');
    const ratio = parseFloat(document.getElementById('edit-slider').value);

    goBtn.disabled = true;
    status.textContent = 'Fetching patches...';

    try {
        const fetchVoice = async (fav) => {
            if (fav.voice_data) return { voice_data: fav.voice_data };
            const resp = await fetch(baseUrl + '/api/voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ syx_url: fav.syx_url, slot: fav.slot })
            });
            return resp.json();
        };

        const [srcData, tgtData] = await Promise.all([fetchVoice(editSlot), fetchVoice(editSlotB)]);
        if (srcData.error || tgtData.error) {
            status.textContent = 'Error: ' + (srcData.error || tgtData.error);
            goBtn.disabled = false;
            return;
        }

        status.textContent = 'Morphing...';
        const resp = await fetch(baseUrl + '/api/morph', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_data: srcData.voice_data, target_data: tgtData.voice_data, note: 60, ratio })
        });
        const data = await resp.json();
        if (data.error) {
            status.textContent = 'Error: ' + data.error;
            goBtn.disabled = false;
            return;
        }

        status.textContent = `Done — distance: ${data.source_dist.toFixed(4)} → ${data.morph_dist.toFixed(4)} (${ratioLabel(data.ratio)})`;
        if (data.wav_url) new Audio(baseUrl + data.wav_url).play();

        const morphFav = {
            id: 'fav_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
            name: morphName(editSlot.name, editSlotB.name),
            syx_url: data.syx_url, slot: 0,
            algorithm: editSlot.algorithm, feedback: editSlot.feedback,
            distance: data.morph_dist, wav_url: data.wav_url,
            source_job: 'morph', created: Date.now()
        };
        favorites.push(morphFav);
        saveFavoritesToStorage();
        renderFavorites();
        appendLog(`Morph: ${editSlot.name} ${ratioLabel(ratio)} ${editSlotB.name}`);

        editSlot = morphFav;
        editSlotB = null;
        renderEditPanel();
    } catch (err) {
        status.textContent = 'Error: ' + err.message;
        goBtn.disabled = false;
    }
}

async function buildModifiedVoiceData() {
    if (!currentMutation) return null;
    let voiceData;
    if (editSlot.voice_data) {
        voiceData = editSlot.voice_data;
    } else if (editSlot.syx_url) {
        const resp = await fetch(baseUrl + '/api/voice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ syx_url: editSlot.syx_url, slot: editSlot.slot })
        });
        const d = await resp.json();
        voiceData = d.voice_data;
    }
    if (!voiceData || !currentMutation.ops) return voiceData;

    const bytes = atob(voiceData);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);

    for (let i = 0; i < 6; i++) {
        const op = currentMutation.ops[i];
        if (!op) continue;
        const base = i * 17;
        arr[base + 14] = op.outputLevel;
        arr[base + 15] = ((op.freqCoarse & 0x1F) << 1) | (arr[base + 15] & 1);
    }
    arr[111] = (arr[111] & 0xF8) | (currentMutation.feedback & 0x07);

    let newB64 = '';
    for (let i = 0; i < arr.length; i++) newB64 += String.fromCharCode(arr[i]);
    return btoa(newB64);
}

async function previewParamsEdit() {
    if (!currentMutation) return;
    const status = document.getElementById('edit-status');
    try {
        status.textContent = 'Rendering preview...';
        const voiceData = await buildModifiedVoiceData();
        if (!voiceData) { status.textContent = 'Error: no voice data'; return; }
        const resp = await fetch(baseUrl + '/api/play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice_data: voiceData, note: 60 })
        });
        const d = await resp.json();
        if (d.wav_url) {
            new Audio(baseUrl + d.wav_url).play();
            status.textContent = 'Playing preview...';
        } else {
            status.textContent = 'Error: no WAV returned';
        }
    } catch (err) {
        status.textContent = 'Error: ' + err.message;
    }
}

async function saveParamsEdit() {
    if (!currentMutation) return;
    const status = document.getElementById('edit-status');

    try {
        const voiceData = await buildModifiedVoiceData();
        if (!voiceData) { status.textContent = 'Error: no voice data'; return; }

        // Create favorite with modified params
        const newFav = {
            id: 'fav_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
            name: editSlot.name + ' tweaked',
            syx_url: null, voice_data: voiceData,
            slot: 0,
            algorithm: editSlot.algorithm,
            feedback: currentMutation.feedback || editSlot.feedback,
            distance: 0, wav_url: null,
            source_job: 'params', created: Date.now(),
            ops: JSON.parse(JSON.stringify(currentMutation.ops)),
        };

        // Render the new patch for preview
        status.textContent = 'Rendering...';
        const resp = await fetch(baseUrl + '/api/play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice_data: voiceData, note: 60 })
        });
        const playData = await resp.json();
        if (playData.wav_url) {
            newFav.wav_url = playData.wav_url;
            new Audio(baseUrl + playData.wav_url).play();
        }

        favorites.push(newFav);
        saveFavoritesToStorage();
        renderFavorites();
        status.textContent = `Saved: ${newFav.name}`;
        appendLog(`Params saved: ${newFav.name}`);
    } catch (err) {
        status.textContent = 'Error: ' + err.message;
    }
}

async function runRandomMutate() {
    if (!editSlot) return;
    const status = document.getElementById('edit-status');
    const intensity = parseInt(document.getElementById('edit-slider').value) / 100;

    status.textContent = 'Mutating...';

    try {
        // Get original voice_data
        let voiceData;
        if (editSlot.voice_data) {
            voiceData = editSlot.voice_data;
        } else if (editSlot.syx_url) {
            const resp = await fetch(baseUrl + '/api/voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ syx_url: editSlot.syx_url, slot: editSlot.slot })
            });
            const d = await resp.json();
            voiceData = d.voice_data;
        }

        if (!voiceData) {
            status.textContent = 'Error: no voice data';
            return;
        }

        // Decode, mutate, re-encode
        const bytes = atob(voiceData);
        const arr = new Uint8Array(bytes.length);
        for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);

        // Gaussian-ish mutation
        const gaussRand = () => {
            let u = 0, v = 0;
            while (u === 0) u = Math.random();
            while (v === 0) v = Math.random();
            return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        };

        const mutatedOps = [];
        for (let i = 0; i < 6; i++) {
            const base = i * 17;
            const origOL = arr[base + 14];
            const origFC = (arr[base + 15] >> 1) & 0x1F;

            // Mutate OutputLevel: σ = intensity × 15
            let newOL = Math.round(origOL + gaussRand() * intensity * 15);
            newOL = Math.max(0, Math.min(99, newOL));

            // Mutate FreqCoarse: σ = intensity × 2
            let newFC = Math.round(origFC + gaussRand() * intensity * 2);
            newFC = Math.max(0, Math.min(31, newFC));

            arr[base + 14] = newOL;
            arr[base + 15] = ((newFC & 0x1F) << 1) | (arr[base + 15] & 1);

            mutatedOps.push({ outputLevel: newOL, freqCoarse: newFC });
        }

        // Mutate Feedback
        const origFB = arr[111] & 0x07;
        let newFB = Math.round(origFB + gaussRand() * intensity * 1.5);
        newFB = Math.max(0, Math.min(7, newFB));
        arr[111] = (arr[111] & 0xF8) | newFB;

        // Re-encode
        let newB64 = '';
        for (let i = 0; i < arr.length; i++) newB64 += String.fromCharCode(arr[i]);
        const mutatedB64 = btoa(newB64);

        // Store as current mutation
        currentMutation = {
            ...editSlot,
            voice_data: mutatedB64,
            feedback: newFB,
            ops: mutatedOps,
            name: editSlot.name + ' *',
        };

        // Render for preview
        const resp = await fetch(baseUrl + '/api/play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice_data: mutatedB64, note: 60 })
        });
        const playData = await resp.json();
        if (playData.wav_url) {
            currentMutation.wav_url = playData.wav_url;
            new Audio(baseUrl + playData.wav_url).play();
        }

        const goBtn = document.getElementById('edit-go');
        goBtn.textContent = 'Save as new';
        status.textContent = `Mutation ready — save or mutate again (intensity: ${Math.round(intensity*100)}%)`;
        appendLog(`Random mutation: intensity ${Math.round(intensity*100)}%`);
    } catch (err) {
        status.textContent = 'Error: ' + err.message;
    }
}

async function saveMutation() {
    if (!currentMutation) return;
    const status = document.getElementById('edit-status');

    try {
        // Render if not already rendered
        if (!currentMutation.wav_url && currentMutation.voice_data) {
            const resp = await fetch(baseUrl + '/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ voice_data: currentMutation.voice_data, note: 60 })
            });
            const playData = await resp.json();
            if (playData.wav_url) currentMutation.wav_url = playData.wav_url;
        }

        const newFav = {
            id: 'fav_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
            name: currentMutation.name || (editSlot.name + ' mutated'),
            syx_url: null, voice_data: currentMutation.voice_data,
            slot: 0,
            algorithm: currentMutation.algorithm || editSlot.algorithm,
            feedback: currentMutation.feedback,
            distance: 0, wav_url: currentMutation.wav_url || null,
            source_job: 'random', created: Date.now(),
            ops: currentMutation.ops,
        };

        favorites.push(newFav);
        saveFavoritesToStorage();
        renderFavorites();
        status.textContent = `Saved: ${newFav.name}`;
        appendLog(`Mutation saved: ${newFav.name}`);

        // Reset for next mutation from original
        currentMutation = null;
        const goBtn = document.getElementById('edit-go');
        goBtn.textContent = 'Mutate';
    } catch (err) {
        status.textContent = 'Error: ' + err.message;
    }
}

function renderFavorites() {
    const panel = document.getElementById('favorites-panel');
    const body = document.getElementById('fav-body');
    const countEl = document.getElementById('fav-count');

    countEl.textContent = favorites.length;

    if (favorites.length === 0) {
        panel.classList.remove('visible');
        return;
    }

    panel.classList.add('visible');
    body.innerHTML = '';

    // Export button
    const exportRow = document.createElement('div');
    exportRow.style.cssText = 'display:flex;justify-content:flex-end;margin-bottom:0.4rem;';
    const exportBtn = document.createElement('button');
    exportBtn.className = 'fav-export-btn';
    exportBtn.textContent = 'Export .syx';
    exportBtn.title = 'Download all favorites as a .syx bank';
    exportBtn.addEventListener('click', exportFavorites);
    exportRow.appendChild(exportBtn);
    body.appendChild(exportRow);

    favorites.forEach(fav => {
        const row = createPatchRow({
            rank: '',
            name: fav.name,
            algorithm: fav.algorithm,
            feedback: fav.feedback,
            distance: fav.distance || 0,
            wav_url: fav.wav_url,
        }, {
            onSelect: null,
            onStar: () => removeFavorite(fav.id),
            onPlay: async () => {
                try {
                    if (fav.voice_data) {
                        await ensureDx7Synth();
                        if (isDx7Ready()) {
                            const stop = playNoteRealtime(fav.voice_data, 60);
                            if (stop) setTimeout(stop, 3000);
                            const blob = await renderOffline(fav.voice_data, 60, 3000);
                            return URL.createObjectURL(blob);
                        }
                        const resp = await fetch(baseUrl + '/api/play', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ voice_data: fav.voice_data, note: 60 })
                        });
                        const data = await resp.json();
                        if (data.wav_url) {
                            new Audio(baseUrl + data.wav_url).play();
                            return baseUrl + data.wav_url;
                        }
                    } else if (fav.syx_url) {
                        const resp = await fetch(baseUrl + '/api/play', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ syx_url: fav.syx_url, slot: fav.slot, note: 60 })
                        });
                        const data = await resp.json();
                        if (data.wav_url) {
                            new Audio(baseUrl + data.wav_url).play();
                            return baseUrl + data.wav_url;
                        }
                    }
                } catch (e) {
                    console.warn('Play fav failed:', e);
                }
                return null;
            },
            baseUrl,
            isFav: true,
        });

        row.classList.add('fav-row');

        // Inline rename on double-click
        addRenameHandlers(row, fav, saveFavoritesToStorage);

        // Add edit + delete buttons
        const actions = document.createElement('span');
        actions.className = 'fav-actions';

        const editBtn = document.createElement('button');
        editBtn.className = 'patch-edit-btn';
        editBtn.textContent = '🔧';
        editBtn.title = 'Edit patch';
        editBtn.style.opacity = '1';
        editBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openEdit(fav);
        });
        actions.appendChild(editBtn);

        const delBtn = document.createElement('button');
        delBtn.className = 'fav-del';
        delBtn.textContent = '×';
        delBtn.title = 'Remove';
        delBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeFavorite(fav.id);
        });
        actions.appendChild(delBtn);

        row.appendChild(actions);
        body.appendChild(row);
    });
}

// --- Favorites Export ---

async function exportFavorites() {
    if (favorites.length === 0) return;

    const patches = favorites.map(fav => ({
        syx_url: fav.syx_url || null,
        slot: fav.slot,
        name: fav.name,
        voice_data: fav.voice_data || null,
    }));

    try {
        const resp = await fetch(baseUrl + '/api/export-syx', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patches })
        });
        const data = await resp.json();
        if (data.error) {
            appendLog('Export error: ' + data.error);
            return;
        }
        // Trigger download
        const a = document.createElement('a');
        a.href = baseUrl + data.syx_url;
        a.download = 'favorites.syx';
        a.click();
        appendLog(`Exported ${data.count} patches to favorites.syx`);
    } catch (e) {
        console.warn('Export failed:', e);
        appendLog('Export failed: ' + e.message);
    }
}

// --- .syx Upload & Tab Creation ---

function initSyxUpload() {
    // .syx files are now handled by the main file input
    // No separate button needed
}

async function handleSyxUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
        const arrayBuffer = await file.arrayBuffer();
        const patches = parseSyxFile(arrayBuffer);
        if (!patches || patches.length === 0) {
            appendLog(`Failed to parse .syx file: ${file.name}`);
            return;
        }

        const name = file.name.replace(/\.[^.]+$/, '');
        appendLog(`Parsed .syx: ${name} (${patches.length} voices)`);

        // Create a new tab for this .syx
        createSyxTab(name, patches);
    } catch (err) {
        appendLog(`Error reading .syx file: ${err.message}`);
    }
}

function createSyxTab(name, patches) {
    const tabId = 'syx_' + Date.now();
    createSyxTabInner(tabId, name, patches);
    saveSyxTabs();
    switchTab(tabId);
}

// Create SYX tab from saved data (used by restoreSyxTabs)
function createSyxTabFromData(tabId, name, patches) {
    createSyxTabInner(tabId, name, patches);
    // Move tab to end of tabs bar
    const tab = tabsEl.querySelector(`[data-tab-id="${tabId}"]`);
    if (tab) tabsEl.appendChild(tab);
    tabsEl.classList.add('visible');
    showImportedPanel();
    switchTab(tabId);
}

function createSyxTabInner(tabId, name, patches) {
    const tab = document.createElement('div');
    tab.className = 'tab';
    tab.dataset.tabId = tabId;

    const label = document.createElement('span');
    label.className = 'tab-label';
    label.textContent = name;

    const closeBtn = document.createElement('button');
    closeBtn.className = 'tab-close';
    closeBtn.textContent = '×';
    closeBtn.title = 'Remove';
    closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteSyxTab(tabId);
    });

    tab.appendChild(label);
    tab.appendChild(closeBtn);
    tab.addEventListener('click', () => switchTab(tabId));
    tabsEl.appendChild(tab);

    const content = document.createElement('div');
    content.className = 'tab-content';
    content.id = 'tab-' + tabId;

    const header = document.createElement('div');
    header.className = 'upload-tab-header';
    const heading = document.createElement('h2');
    heading.textContent = name;
    header.appendChild(heading);
    content.appendChild(header);

    const list = document.createElement('div');
    list.className = 'patch-list';

    const listHeader = document.createElement('div');
    listHeader.className = 'patch-list-header';
    listHeader.innerHTML = '<span class="patch-spec"></span><span class="patch-name">Name</span><span class="patch-col">Algo</span><span class="patch-col">FB</span><span class="patch-col">Dist</span>';
    list.appendChild(listHeader);

    const body = document.createElement('div');
    body.className = 'patch-list-body';
    makeDropTarget(body);

    const patchNames = loadPatchNames();

    patches.forEach(patch => {
        // Ensure voice_data exists from raw bytes
        if (!patch.voice_data && patch.raw) {
            patch.voice_data = voiceRawToBase64(patch.raw);
        }
        // Ensure stable ID
        if (!patch.id) {
            patch.id = patchId(patch);
        }
        // Apply persisted rename
        const savedName = patchNames[patch.id];
        const displayName = savedName || patch.name;
        const row = createPatchRow({
            rank: '',
            name: displayName,
            algorithm: patch.algorithm,
            feedback: patch.feedback,
            distance: 0,
            wav_url: null,
            voice_data: patch.voice_data || null,
        }, {
            onSelect: () => {
                // Assign to piano — set active voice_data and pianoroll patch
                if (patch.voice_data) {
                    activeSyxVoiceData = patch.voice_data;
                    activePianorollPatchId = displayName;
                    rebuildPianoDock();
                    // Add to voicebank
                    const dock = document.getElementById('piano-dock');
                    if (dock._voiceBank) dock._voiceBank._addEntry(displayName, patch.voice_data, displayName);
                    document.getElementById('piano-dock').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            },
            onStar: () => {
                // Deduplicate by patch data hash
                const pid = patch.id || patchId(patch);
                const existing = favorites.find(f => f.patch_hash === pid);
                if (existing) {
                    appendLog(`Already favorited: ${existing.name}`);
                    return;
                }
                const fav = {
                    id: 'fav_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
                    patch_hash: pid,
                    name: displayName,
                    syx_url: null,
                    voice_data: patch.voice_data || null,
                    slot: patch.slot,
                    algorithm: patch.algorithm,
                    feedback: patch.feedback,
                    distance: 0,
                    wav_url: null,
                    source_job: name + '.syx',
                    created: Date.now(),
                };
                favorites.push(fav);
                saveFavoritesToStorage();
                renderFavorites();
                const starEl = row.querySelector('.patch-star');
                if (starEl) {
                    starEl.classList.add('fav-active');
                    starEl.textContent = '★';
                }
            },
            onPlay: async () => {
                try {
                    if (!patch.voice_data) return null;
                    await ensureDx7Synth();
                    if (isDx7Ready()) {
                        const stop = playNoteRealtime(patch.voice_data, 60);
                        if (stop) setTimeout(stop, 3000);
                        const blob = await renderOffline(patch.voice_data, 60, 3000);
                        return URL.createObjectURL(blob);
                    }
                    const resp = await fetch(baseUrl + '/api/play', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ voice_data: patch.voice_data, note: 60 })
                    });
                    const data = await resp.json();
                    if (data.wav_url) {
                        new Audio(baseUrl + data.wav_url).play();
                        return baseUrl + data.wav_url;
                    }
                } catch (e) {
                    console.warn('Play patch failed:', e);
                }
                return null;
            },
            baseUrl,
            isFav: false,
        });

        // Store voice_data and patchId on row for persistence
        if (patch.voice_data) {
            row.dataset.voiceData = patch.voice_data;
        }
        row.dataset.patchId = patch.id;

        // Add rename handler
        addRenameHandlers(row, patch, () => {
            patchNames[patch.id] = patch.name;
            savePatchNames(patchNames);
            saveSyxTabs();
        });

        body.appendChild(row);
    });

    list.appendChild(body);
    content.appendChild(list);
    tabContent.appendChild(content);

    // Batch-render waveforms for patches with voice_data
    const patchEls = [];
    body.querySelectorAll('.patch-row').forEach(row => {
        const vd = row.dataset.voiceData;
        const cvs = row.querySelector('.patch-canvas');
        if (vd && cvs) patchEls.push({ voiceData: vd, canvas: cvs });
    });
    if (patchEls.length > 0) {
        batchRenderWaveforms(patchEls, (canvas, blob) => {
            blob.arrayBuffer().then(buf => {
                const samples = decodeWavSamples(buf);
                if (samples && samples.length > 0) drawMiniWaveform(canvas, samples);
            });
        });
    }
}

fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Route .syx files to the dedicated handler
    if (file.name.toLowerCase().endsWith('.syx')) {
        appendLog(`Routing .syx file: ${file.name} (${file.size} bytes)`);
        handleSyxUpload({ target: { files: [file] } });
        fileInput.value = '';
        return;
    }

    const objectUrl = URL.createObjectURL(file);
    const jobId = 'job_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
    const jobIndex = jobs.length;
    const job = {
        id: jobId,
        file, objectUrl,
        fileName: file.name,
        probeData: null, resultData: null,
        statusUrl: null, tabEl: null,
        selectedMatch: null, pollCount: 0
    };
    jobs.push(job);

    try {
        const arrayBuffer = await file.arrayBuffer();
        await dbPut(jobId, arrayBuffer);
    } catch (e) {
        console.warn('Failed to store WAV in IndexedDB:', e);
    }

    addTab(jobIndex);
    showJob(jobIndex);
    saveSessions();

    appendLog(`Probing: ${file.name}`);
    const wrapper = document.getElementById('progress-wrapper');
    if (wrapper) wrapper.style.display = 'block';
    const fill = document.getElementById('progress-fill');
    if (fill) fill.style.width = '10%';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(baseUrl + '/api/probe', {
            method: 'POST',
            body: formData
        });

        const text = await response.text();

        if (!response.ok) {
            appendLog(`[${response.status}] ${text}`);
            return;
        }

        let data;
        try {
            data = JSON.parse(text);
        } catch {
            appendLog(`[${response.status}] ${text}`);
            return;
        }

        appendLog(`[${response.status}] ${JSON.stringify(data, null, 2)}`);
        job.probeData = data;
        saveSessions();
        upgradeTab(jobIndex);
        if (activeJobIndex === jobIndex) renderProbe(jobIndex);
    } catch (err) {
        appendLog(`Error: ${err.message}`);
    }
});

async function reprobeJob(idx) {
    const job = jobs[idx];
    if (!job || !job.file) return;
    appendLog(`Re-probing: ${job.fileName}`);
    try {
        const formData = new FormData();
        formData.append('file', job.file);
        const response = await fetch(baseUrl + '/api/probe', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            appendLog(`[reprobe ${response.status}] ${await response.text()}`);
            upgradeTab(idx);
            return;
        }
        const data = await response.json();
        job.probeData = data;
        saveSessions();
        upgradeTab(idx);
        if (activeJobIndex === idx) renderProbe(idx);
        appendLog(`Re-probe complete: ${job.fileName}`);
    } catch (err) {
        appendLog(`Re-probe error: ${err.message}`);
        upgradeTab(idx);
    }
}

function startPolling() {
    stopPolling();
    pollInterval = setInterval(() => {
        if (pollQueue.length > 0) checkStatus(pollQueue[0]);
    }, 3000);
    checkStatus(pollQueue[0]);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

function advanceQueue() {
    pollQueue.shift();
    if (pollQueue.length > 0) {
        startPolling();
    }
}

async function checkStatus(jobIndex) {
    const job = jobs[jobIndex];
    if (!job || !job.statusUrl) return;
    try {
        const response = await fetch(baseUrl + job.statusUrl);
        const text = await response.text();

        let data;
        try {
            data = JSON.parse(text);
        } catch {
            appendLog(`[status] ${text}`);
            if (tabLabel) tabLabel.textContent = job.fileName + ' (failed)';
            const tabSpinner = job.tabEl ? job.tabEl.querySelector('.spinner') : null;
            if (tabSpinner) tabSpinner.remove();
            advanceQueue();
            return;
        }

        const tabLabel = job.tabEl ? job.tabEl.querySelector('.tab-label') : null;

        if (data.status === 'queued') {
            job.pollCount = (job.pollCount || 0) + 1;
            appendLog(`[${job.fileName}] Queued, waiting on ${data.queue_position}`);
            if (tabLabel) tabLabel.textContent = job.fileName + ' (queued)';
            const wrapper = document.getElementById('progress-wrapper');
            if (wrapper) wrapper.style.display = 'block';
            const fill = document.getElementById('progress-fill');
            if (fill) fill.style.width = '0%';
        } else if (data.status === 'processing') {
            job.pollCount = (job.pollCount || 0) + 1;
            appendLog(`[${job.fileName}] Processing: ${data.progress_percent}% — ${data.phase || ''}`);
            if (tabLabel) tabLabel.textContent = job.fileName + ' [' + job.pollCount + ']';
            const wrapper = document.getElementById('progress-wrapper');
            if (wrapper) wrapper.style.display = 'block';
            const fill = document.getElementById('progress-fill');
            if (fill) fill.style.width = data.progress_percent + '%';
        } else if (data.status === 'completed') {
            appendLog(`[${job.fileName}] Completed`);
            const wrapper = document.getElementById('progress-wrapper');
            if (wrapper) wrapper.style.display = 'none';
            if (tabLabel) tabLabel.textContent = job.fileName;
            job.resultData = data;
            job.pollCount = 0;
            saveSessions();
            upgradeTab(jobIndex);
            if (activeJobIndex === jobIndex) renderResults(jobIndex);
            advanceQueue();
        } else {
            appendLog(`[${job.fileName}] ${JSON.stringify(data, null, 2)}`);
            const wrapper = document.getElementById('progress-wrapper');
            if (wrapper) wrapper.style.display = 'none';
            if (tabLabel) tabLabel.textContent = job.fileName + ' (failed)';
            const tabSpinner = job.tabEl ? job.tabEl.querySelector('.spinner') : null;
            if (tabSpinner) tabSpinner.remove();
            advanceQueue();
        }
    } catch (err) {
        appendLog(`[status error] ${err.message}`);
        if (job.tabEl) {
            const tabLabel = job.tabEl.querySelector('.tab-label');
            if (tabLabel) tabLabel.textContent = job.fileName + ' (failed)';
            const tabSpinner = job.tabEl.querySelector('.spinner');
            if (tabSpinner) tabSpinner.remove();
        }
        advanceQueue();
    }
}

function addTab(index) {
    const job = jobs[index];
    const tab = document.createElement('div');
    tab.className = 'tab';
    tab.dataset.index = index;
    tab.dataset.jobId = job.id;

    const spinner = document.createElement('div');
    spinner.className = 'spinner';

    const label = document.createElement('span');
    label.className = 'tab-label';
    const shortId = job.id.length > 12 ? job.id.slice(-8) : job.id;
    label.textContent = job.fileName + ' [' + shortId + ']';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'tab-close';
    closeBtn.textContent = '×';
    closeBtn.title = 'Remove';
    closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteSession(index);
    });

    tab.appendChild(spinner);
    tab.appendChild(label);
    tab.appendChild(closeBtn);
    tab.addEventListener('click', () => showJob(index));
    tabsEl.appendChild(tab);
    tabsEl.classList.add('visible');
    showImportedPanel();
    job.tabEl = tab;

    // Create per-job content div inside tabContent
    const contentId = 'tab-wav-' + index;
    let content = tabContent.querySelector('#' + contentId);
    if (!content) {
        content = document.createElement('div');
        content.className = 'tab-content';
        content.id = contentId;
        tabContent.appendChild(content);
    }
}

// Get or create the content div for a tab
function ensureTabContent(id) {
    let el = tabContent.querySelector('#' + id);
    if (!el) {
        el = document.createElement('div');
        el.className = 'tab-content';
        el.id = id;
        tabContent.appendChild(el);
    }
    return el;
}

// Hide all tab content children, show only the matching one
function showOnlyTabContent(id) {
    tabContent.querySelectorAll('.tab-content').forEach(tc => {
        tc.classList.toggle('visible', tc.id === id);
    });
    tabContent.classList.add('visible');
    showImportedPanel();
}

function upgradeTab(index) {
    const job = jobs[index];
    const tab = job.tabEl;
    if (!tab) return;

    tab.innerHTML = '';

    const playBtn = document.createElement('button');
    playBtn.className = 'tab-play';
    playBtn.textContent = '▶';
    playBtn.title = 'Play original';
    const audio = new Audio(job.objectUrl);
    playBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (audio.paused) {
            audio.play();
            playBtn.textContent = '⏸';
        } else {
            audio.pause();
            audio.currentTime = 0;
            playBtn.textContent = '▶';
        }
    });
    audio.addEventListener('ended', () => {playBtn.textContent = '▶';});

    const label = document.createElement('span');
    label.className = 'tab-label';
    label.textContent = job.fileName;

    const closeBtn = document.createElement('button');
    closeBtn.className = 'tab-close';
    closeBtn.textContent = '×';
    closeBtn.title = 'Remove';
    closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteSession(index);
    });

    tab.appendChild(playBtn);
    tab.appendChild(label);
    tab.appendChild(closeBtn);
    tab.addEventListener('click', () => showJob(index));
}

function switchTab(tabId) {
    activeJobIndex = null;
    activeSyxVoiceData = null;
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tabId === tabId);
    });
    showOnlyTabContent('tab-' + tabId);

    // Set up piano with first patch's voice_data as default
    const syxContent = tabContent.querySelector('#tab-' + tabId);
    if (syxContent) {
        const firstRow = syxContent.querySelector('.patch-row');
        if (firstRow && firstRow.dataset.voiceData) {
            activeSyxVoiceData = firstRow.dataset.voiceData;
            activePianorollPatchId = firstRow.dataset.patchName || null;
        }
    }

    rebuildPianoDock();

    // Make patch rows set the active voice_data on click
    if (syxContent) {
        syxContent.querySelectorAll('.patch-row').forEach(row => {
            row.addEventListener('click', () => {
                if (row.dataset.voiceData) {
                    activeSyxVoiceData = row.dataset.voiceData;
                }
            });
        });
    }
}

function showJob(index) {
    activeJobIndex = index;
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', Number(t.dataset.index) === index);
    });
    showOnlyTabContent('tab-wav-' + index);
    const job = jobs[index];
    if (job.resultData) {
        renderResults(index);
    } else if (job.probeData) {
        renderProbe(index);
    } else {
        const el = ensureTabContent('tab-wav-' + index);
        el.innerHTML = '<div class="result-info"><span>Probing...</span></div>';
    }
}

function renderProbe(index) {
    if (forkOsc) { forkOsc.stop(); forkOsc = null; forkGain = null; }
    const job = jobs[index];
    const data = job.probeData;
    const locked = !!job.statusUrl;
    const el = ensureTabContent('tab-wav-' + index);
    el.innerHTML = '';
    rebuildPianoDock();

    const header = document.createElement('div');
    header.className = 'config-header' + (locked ? ' collapsed' : '');
    header.innerHTML = `<span>Configuration</span><span class="config-arrow">▼</span>`;
    el.appendChild(header);

    const body = document.createElement('div');
    body.className = 'config-body' + (locked ? ' collapsed' : '');

    const info = document.createElement('div');
    info.className = 'probe-info';
    const badges = [
        {label: 'Freq', value: data.f0_hz ? data.f0_hz.toFixed(1) + ' Hz' : null},
        {label: 'Conf', value: data.f0_confidence ? data.f0_confidence.toFixed(2) : null},
        {label: 'Centroid', value: data.spectral_centroid_hz ? data.spectral_centroid_hz.toFixed(0) + ' Hz' : null},
        data.seen_before ? {label: 'SEEN BEFORE', value: '', cls: 'seen'} : null,
        data.best_match ? {label: 'Best', value: `algo ${data.best_match.algorithm} fb ${data.best_match.feedback} (${data.best_match.distance.toFixed(3)})`, cls: 'match'} : null,
    ].filter(Boolean);

    badges.forEach(b => {
        const span = document.createElement('span');
        span.className = 'badge' + (b.cls ? ' ' + b.cls : '');
        span.innerHTML = b.value ? `<strong>${b.label}:</strong> ${b.value}` : `<strong>${b.label}</strong>`;
        info.appendChild(span);
    });
    body.appendChild(info);

    const form = document.createElement('form');
    form.className = 'probe-form';
    form.onsubmit = () => false;

    const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

    function parseRootNote(rootNote) {
        if (!rootNote) return { note: 0, octave: 4 };
        const s = String(rootNote).trim();
        if (/^\d+$/.test(s)) {
            const midi = parseInt(s, 10);
            return { note: midi % 12, octave: Math.floor(midi / 12) - 1 };
        }
        const m = s.match(/^([A-G]#?)(\d)$/);
        if (m) {
            const idx = NOTE_NAMES.indexOf(m[1]);
            if (idx >= 0) return { note: idx, octave: parseInt(m[2], 10) };
        }
        return { note: 0, octave: 4 };
    }

    const parsed = parseRootNote(data.root_note);

    const noteRow = document.createElement('div');
    noteRow.className = 'note-row';

    const noteField = document.createElement('div');
    noteField.className = 'probe-field';
    noteField.innerHTML = `<label for="pf-note-${index}">Note</label>`;
    const noteSelect = document.createElement('select');
    noteSelect.id = `pf-note-${index}`;
    noteSelect.name = 'note_name';
    NOTE_NAMES.forEach((n, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = n;
        if (i === parsed.note) opt.selected = true;
        noteSelect.appendChild(opt);
    });
    if (locked) noteSelect.disabled = true;
    noteField.appendChild(noteSelect);
    noteRow.appendChild(noteField);

    const octField = document.createElement('div');
    octField.className = 'probe-field';
    octField.innerHTML = `<label for="pf-oct-${index}">Octave</label>`;
    const octSelect = document.createElement('select');
    octSelect.id = `pf-oct-${index}`;
    for (let o = 0; o <= 8; o++) {
        const opt = document.createElement('option');
        opt.value = o;
        opt.textContent = o;
        if (o === parsed.octave) opt.selected = true;
        octSelect.appendChild(opt);
    }
    if (locked) octSelect.disabled = true;
    octField.appendChild(octSelect);
    noteRow.appendChild(octField);

    const hiddenNote = document.createElement('input');
    hiddenNote.type = 'hidden';
    hiddenNote.name = 'note';
    hiddenNote.value = data.root_note || 'C4';
    noteRow.appendChild(hiddenNote);

    const forkBtn = document.createElement('button');
    forkBtn.type = 'button';
    forkBtn.className = 'tuning-fork-btn';
    forkBtn.title = 'Tuning fork — click to play/stop';
    forkBtn.innerHTML = '<b>Y</b>';
    if (locked) forkBtn.disabled = true;
    noteRow.appendChild(forkBtn);

    function syncHiddenNote() {
        hiddenNote.value = NOTE_NAMES[noteSelect.value] + octSelect.value;
    }
    noteSelect.addEventListener('change', () => {
        syncHiddenNote();
        if (forkOsc) forkOsc.frequency.value = noteToFreq(parseInt(noteSelect.value), parseInt(octSelect.value));
    });
    octSelect.addEventListener('change', () => {
        syncHiddenNote();
        if (forkOsc) forkOsc.frequency.value = noteToFreq(parseInt(noteSelect.value), parseInt(octSelect.value));
    });

    forkBtn.addEventListener('click', () => toggleFork(forkBtn, noteSelect, octSelect));

    form.appendChild(noteRow);

    probeFields.forEach(key => {
        const field = document.createElement('div');
        field.className = 'probe-field';
        const numVal = data[key];
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `pf-${key}-${index}`;
        input.name = key;
        input.step = key === 'sustain_level' || key === 'harmonicity' ? '0.001' : '0.01';
        input.min = '0';
        if (key === 'sustain_level' || key === 'brightness' || key === 'harmonicity') input.max = '1';
        input.value = numVal !== undefined ? roundForInput(numVal) : '';
        if (locked) input.disabled = true;
        field.innerHTML = `<label for="pf-${key}-${index}">${key.replace(/_/g, ' ')}</label>`;
        field.appendChild(input);
        form.appendChild(field);
    });

    if (!locked) {
        const actions = document.createElement('div');
        actions.className = 'probe-actions';

        const resetBtn = document.createElement('button');
        resetBtn.type = 'button';
        resetBtn.className = 'btn-reset';
        resetBtn.textContent = 'Reset';
        resetBtn.addEventListener('click', () => renderProbe(index));

        const matchBtn = document.createElement('button');
        matchBtn.type = 'button';
        matchBtn.className = 'btn-match';
        matchBtn.textContent = 'Start Matching';
        matchBtn.addEventListener('click', () => startMatch(index));

        actions.appendChild(resetBtn);
        actions.appendChild(matchBtn);
        form.appendChild(actions);
    }

    body.appendChild(form);
    el.appendChild(body);

    requestAnimationFrame(() => {
        if (!body.classList.contains('collapsed')) {
            body.style.maxHeight = body.scrollHeight + 'px';
        }
    });

    header.addEventListener('click', () => {
        const collapsed = body.classList.toggle('collapsed');
        header.classList.toggle('collapsed', collapsed);
        if (!collapsed) {
            body.style.maxHeight = body.scrollHeight + 'px';
        }
    });

    tabContent.classList.add('visible');
    showImportedPanel();
}

function roundForInput(val) {
    if (Number.isInteger(val)) return val;
    return parseFloat(val.toFixed(6));
}

async function startMatch(index) {
    const job = jobs[index];
    if (!job || !job.probeData) return;

    appendLog(`Matching: ${job.file.name}`);

    const el = ensureTabContent('tab-wav-' + index);

    try {
        const formData = new FormData();
        formData.append('file', job.file);

        const noteInput = el.querySelector(`[name="note"]`);
        if (noteInput && noteInput.value !== '') {
            formData.append('note', noteInput.value);
        }

        probeFields.forEach(key => {
            const input = el.querySelector(`[name="${key}"]`);
            if (input && input.value !== '') {
                formData.append(key, input.value);
            }
        });

        const response = await fetch(baseUrl + '/api/match', {
            method: 'POST',
            body: formData
        });

        const text = await response.text();

        if (!response.ok) {
            appendLog(`[${response.status}] ${text}`);
            return;
        }

        let data;
        try {
            data = JSON.parse(text);
        } catch {
            appendLog(`[${response.status}] ${text}`);
            return;
        }

        appendLog(`[${response.status}] ${JSON.stringify(data, null, 2)}`);

        if (data.status_url) {
            job.statusUrl = data.status_url;
            saveSessions(); // Persist statusUrl so polling resumes on reload
            appendLog(`Status URL: ${data.status_url}`);
            renderProbe(index);
            pollQueue.push(index);
            if (pollQueue.length === 1) startPolling();
        }
    } catch (err) {
        appendLog(`Error: ${err.message}`);
    }
}

let audioCtx;

async function ensureDx7Synth() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    return initDx7Synth(audioCtx);
}

function noteToFreq(note, octave) {
    return 440 * Math.pow(2, (note + (octave - 4) * 12 - 9) / 12);
}

let forkOsc = null;
let forkGain = null;
function toggleFork(btn, noteSel, octSel) {
    if (forkOsc) {
        forkOsc.stop();
        forkOsc = null;
        forkGain = null;
        btn.classList.remove('active');
        return;
    }
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const freq = noteToFreq(parseInt(noteSel.value), parseInt(octSel.value));
    forkOsc = audioCtx.createOscillator();
    forkGain = audioCtx.createGain();
    forkOsc.type = 'sine';
    forkOsc.frequency.value = freq;
    forkGain.gain.setValueAtTime(0.25, audioCtx.currentTime);
    forkOsc.connect(forkGain);
    forkGain.connect(audioCtx.destination);
    forkOsc.start();
    btn.classList.add('active');
}

async function playNote(midi, job, voiceData, durationMs) {
    const dur = Math.round(durationMs || 0);

    // Resolve voice_data from job if not directly provided
    if (!voiceData && job) {
        const selected = job.selectedMatch;
        if (selected && selected.voice_data) voiceData = selected.voice_data;
    }

    if (voiceData) {
        // Try client-side DX7 synth first
        await ensureDx7Synth();
        if (isDx7Ready()) {
            const stop = playNoteRealtime(voiceData, midi);
            if (stop && dur > 0) {
                setTimeout(stop, dur * 0.6);
            }
            return;
        }

        // Fallback: server /api/play
        try {
            const body = { voice_data: voiceData, note: midi };
            if (dur) body.duration_ms = dur;
            const resp = await fetch(baseUrl + '/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await resp.json();
            if (data.wav_url) {
                new Audio(baseUrl + data.wav_url).play();
                return;
            }
        } catch (err) {
            console.warn('API play failed, falling back to sine:', err);
        }
    }

    // WAV job fallback via syx_url (no voice_data available)
    let selected = job ? job.selectedMatch : null;
    if (selected && job.resultData && job.resultData.syx_url) {
        try {
            const body = {
                syx_url: job.resultData.syx_url,
                slot: selected.rank - 1,
                note: midi,
            };
            if (dur) body.duration_ms = dur;
            const resp = await fetch(baseUrl + '/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await resp.json();
            if (data.wav_url) {
                new Audio(baseUrl + data.wav_url).play();
                return;
            }
        } catch (err) {
            console.warn('API play failed, falling back to sine:', err);
        }
    }

    // Final fallback: sine oscillator
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const freq = 440 * Math.pow(2, (midi - 69) / 12);
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.6);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.6);
}

function renderResults(index) {
    const job = jobs[index];
    const data = job.resultData;
    const el = ensureTabContent('tab-wav-' + index);

    if (!el.querySelector('.config-body')) {
        renderProbe(index);
    }

    // Collapse config when showing results — spectrogram + patch list are the focus
    const configBody = el.querySelector('.config-body');
    const configHeader = el.querySelector('.config-header');
    if (configBody && !configBody.classList.contains('collapsed')) {
        configBody.classList.add('collapsed');
        if (configHeader) configHeader.classList.add('collapsed');
    }

    let resultsArea = el.querySelector('.results-area');
    if (!resultsArea) {
        resultsArea = document.createElement('div');
        resultsArea.className = 'results-area';
        el.appendChild(resultsArea);
    }
    resultsArea.innerHTML = '';

    // Render generation — stale async callbacks bail out
    job._renderGen = (job._renderGen || 0) + 1;
    const myGen = job._renderGen;

    // Spectrogram — show target waveform with play button
    if (job.file) {
        const reader = new FileReader();
        reader.onload = () => {
            if (job._renderGen !== myGen) return; // stale render, bail
            const samples = decodeWavSamples(reader.result);
            if (samples && samples.length > 0) {
                const specEl = createSpectrogram(samples, 44100, () => {
                    if (job.objectUrl) new Audio(job.objectUrl).play();
                });
                resultsArea.insertBefore(specEl, resultsArea.firstChild);
            }
        };
        reader.readAsArrayBuffer(job.file);
    }

    const info = document.createElement('div');
    info.className = 'result-info';
    info.innerHTML = `
        <span><strong>${data.root_note}</strong></span>
        <span>MIDI ${data.midi_note}</span>
        <span>${data.f0_hz.toFixed(1)} Hz</span>
        <span>Best: ${data.best_distance.toFixed(4)}</span>
    `;
    resultsArea.appendChild(info);

    job.selectedMatch = job.selectedMatch || null;

    const onStarFn = (m) => {
        const favName = job.fileName.replace(/\.[^.]+$/, '') + ' #' + m.rank;
        const existing = favorites.find(f => f.name === favName);
        if (existing) {
            removeFavorite(existing.id);
            return false;
        } else {
            addFavorite(job, m);
            return true;
        }
    };

    const onSelectFn = (m) => {
        job.selectedMatch = m;
        highlightPatchRow(primaryList, m);
        highlightPatchRow(timbralList, m);
        highlightPatchRow(fullList, m);
        // Add to voicebank
        if (m.voice_data) {
            const dock = document.getElementById('piano-dock');
            if (dock._voiceBank) dock._voiceBank._addEntry(m.name || ('#' + m.rank), m.voice_data, m.name || ('#' + m.rank));
        }
    };

    const detectedNote = job.probeData ? (job.probeData.midi_note || 60) : 60;

    const onPlayFn = async (m) => {
        if (m.voice_data) {
            await ensureDx7Synth();
            if (isDx7Ready()) {
                const stop = playNoteRealtime(m.voice_data, detectedNote);
                if (stop) setTimeout(stop, 3000);
                return;
            }
        }
        if (m.wav_url) {
            new Audio(baseUrl + m.wav_url).play();
        }
    };

    const onEditFn = (m) => {
        openEdit({
            id: 'morph_' + Date.now(),
            name: m.name || (`${job.fileName.replace(/\.[^.]+$/, '')} #${m.rank}`),
            syx_url: job.resultData?.syx_url || null,
            slot: m.rank - 1,
            algorithm: m.algorithm,
            feedback: m.feedback,
            distance: m.distance,
            wav_url: m.wav_url,
            voice_data: m.voice_data || null,
        });
    };

    const favIds = favorites.map(f => f.syx_url + ':' + f.slot);

    const primaryList = createPatchList(data.matches, {
        onSelect: onSelectFn,
        onStar: onStarFn,
        onPlay: onPlayFn,
        onEdit: onEditFn,
        baseUrl,
        selectedMatch: job.selectedMatch,
        favIds,
    });
    resultsArea.appendChild(primaryList);

    // Add rename support to result rows
    const resultPatchNames = loadPatchNames();
    primaryList.querySelectorAll('.patch-row').forEach(row => {
        const algo = parseInt(row.dataset.algo);
        const fb = parseInt(row.dataset.fb);
        const m = data.matches.find(m => m.algorithm === algo && m.feedback === fb);
        if (m) {
            const pid = patchId({ voice_data: m.voice_data, algorithm: m.algorithm, feedback: m.feedback });
            if (resultPatchNames[pid]) {
                const nameEl = row.querySelector('.patch-name');
                if (nameEl) nameEl.textContent = resultPatchNames[pid];
                m.name = resultPatchNames[pid];
            }
            addRenameHandlers(row, m, () => {
                resultPatchNames[pid] = m.name;
                savePatchNames(resultPatchNames);
            });
        }
    });

    const syxLink = document.createElement('a');
    syxLink.className = 'syx-link';
    syxLink.href = data.syx_url;
    syxLink.target = '_blank';
    syxLink.rel = 'noopener';
    syxLink.textContent = 'Download bank.syx';
    resultsArea.appendChild(syxLink);

    let timbralList = null;
    if (data.timbral_matches && data.timbral_matches.length > 0) {
        timbralList = createPatchList(data.timbral_matches, {
            onSelect: onSelectFn,
            onStar: onStarFn,
            onPlay: onPlayFn,
            onEdit: onEditFn,
            baseUrl,
            selectedMatch: job.selectedMatch,
            favIds,
        });
        const details = document.createElement('details');
        details.className = 'collapsible-section';
        const summary = document.createElement('summary');
        summary.textContent = 'Timbral Top 5 — Best static timbre match, filtered by spectral similarity';
        details.appendChild(summary);
        details.appendChild(timbralList);
        resultsArea.appendChild(details);
    }

    let fullList = null;
    if (data.full_matches && data.full_matches.length > 0) {
        fullList = createPatchList(data.full_matches, {
            onSelect: onSelectFn,
            onStar: onStarFn,
            onPlay: onPlayFn,
            onEdit: onEditFn,
            baseUrl,
            selectedMatch: job.selectedMatch,
            favIds,
        });
        const details = document.createElement('details');
        details.className = 'collapsible-section';
        const summary = document.createElement('summary');
        summary.textContent = 'Full Top 5 — Best temporal match, DiffOptimizer-refined envelopes';
        details.appendChild(summary);
        details.appendChild(fullList);
        resultsArea.appendChild(details);
    }

    // Auto-select first row if nothing selected yet
    if (!job.selectedMatch && data.matches && data.matches.length > 0) {
        job.selectedMatch = data.matches[0];
    }
    // Highlight the selected row
    if (job.selectedMatch) {
        highlightPatchRow(primaryList, job.selectedMatch);
        highlightPatchRow(timbralList, job.selectedMatch);
        highlightPatchRow(fullList, job.selectedMatch);
    }

    // Piano — anchored above footer in piano-dock
    rebuildPianoDock(data.midi_note, job);

    // Batch-render waveforms for match results
    const matchEls = [];
    [primaryList, timbralList, fullList].forEach(list => {
        if (!list) return;
        list.querySelectorAll('.patch-row').forEach(row => {
            const algo = parseInt(row.dataset.algo);
            const fb = parseInt(row.dataset.fb);
            const m = (data.matches || []).concat(data.timbral_matches || []).concat(data.full_matches || [])
                .find(mm => mm.algorithm === algo && mm.feedback === fb);
            if (m && m.voice_data) {
                const cvs = row.querySelector('.patch-canvas');
                if (cvs) matchEls.push({ voiceData: m.voice_data, canvas: cvs });
            }
        });
    });
    if (matchEls.length > 0) {
        const midiNote = data.midi_note || 60;
        batchRenderWaveforms(matchEls, (canvas, blob) => {
            blob.arrayBuffer().then(buf => {
                const samples = decodeWavSamples(buf);
                if (samples && samples.length > 0) drawMiniWaveform(canvas, samples);
            });
        }, midiNote);
    }

    tabContent.classList.add('visible');
    showImportedPanel();
}

function appendLog(message) {
    const log = document.getElementById("log");
    const timestamp = new Date().toLocaleTimeString();
    log.value += `[${timestamp}] ${message}\n`;
    log.scrollTop = log.scrollHeight;
    const label = document.getElementById('progress-label');
    if (label) {
        const short = message.replace(/^\[.*?\]\s*/, '').substring(0, 80);
        label.textContent = short;
    }
}

// --- Menu Bar ---

// Menu dropdown toggle
document.querySelectorAll('.menu-item').forEach(item => {
    item.querySelector('.menu-label').addEventListener('click', (e) => {
        e.stopPropagation();
        const wasOpen = item.classList.contains('open');
        document.querySelectorAll('.menu-item.open').forEach(i => i.classList.remove('open'));
        if (!wasOpen) item.classList.add('open');
    });
});
document.addEventListener('click', () => {
    document.querySelectorAll('.menu-item.open').forEach(i => i.classList.remove('open'));
});

// File menu
document.getElementById('menu-import-wav').addEventListener('click', () => {
    fileInput.accept = '.wav';
    fileInput.click();
});
document.getElementById('menu-import-syx').addEventListener('click', () => {
    fileInput.accept = '.syx';
    fileInput.click();
});
document.getElementById('menu-export-syx').addEventListener('click', () => {
    exportFavorites();
});

// History menu
document.getElementById('menu-show-log').addEventListener('click', () => {
    const dock = document.getElementById('piano-dock');
    if (dock && dock._showLog) dock._showLog();
});
document.getElementById('menu-clear-history').addEventListener('click', async () => {
    if (!confirm('Clear all session history?')) return;
    await clearAllSessions();
});

// Help menu
document.getElementById('menu-help').addEventListener('click', () => {
    document.getElementById('help-modal').style.display = '';
});
document.getElementById('menu-about').addEventListener('click', () => {
    document.getElementById('about-modal').style.display = '';
});
document.getElementById('menu-credits').addEventListener('click', () => {
    document.getElementById('credits-modal').style.display = '';
});
document.getElementById('menu-legal').addEventListener('click', () => {
    document.getElementById('legal-modal').style.display = '';
});
document.getElementById('menu-donate').addEventListener('click', () => {
    window.open('https://github.com/sponsors/kompadre', '_blank');
});

// Modal close handlers
['credits', 'about', 'help', 'legal'].forEach(id => {
    document.getElementById(id + '-close').addEventListener('click', () => {
        document.getElementById(id + '-modal').style.display = 'none';
    });
    document.getElementById(id + '-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) e.currentTarget.style.display = 'none';
    });
});

// Favorites header toggle
document.getElementById('fav-header').addEventListener('click', () => {
    const body = document.getElementById('fav-body');
    const header = document.getElementById('fav-header');
    const collapsed = body.classList.toggle('collapsed');
    header.classList.toggle('collapsed', collapsed);
    if (!collapsed) {
        body.style.maxHeight = body.scrollHeight + 'px';
    }
});

// Edit panel events
document.getElementById('edit-go').addEventListener('click', runEditAction);
document.getElementById('edit-clear').addEventListener('click', () => {
    editSlot = null;
    editSlotB = null;
    currentMutation = null;
    renderEditPanel();
});
document.getElementById('edit-close').addEventListener('click', () => {
    document.getElementById('edit-panel').classList.remove('visible');
});

// Mode tab buttons
document.querySelectorAll('.edit-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setEditMode(btn.dataset.mode));
});

// Shared slider
const editSlider = document.getElementById('edit-slider');
editSlider.addEventListener('input', () => {
    const valEl = document.getElementById('edit-slider-val');
    if (valEl) valEl.textContent = editMode === 'random' ? editSlider.value : parseFloat(editSlider.value).toFixed(2);

    // Update status for morph mode
    if (editMode === 'morph' && editSlot && editSlotB) {
        const status = document.getElementById('edit-status');
        status.textContent = `Ready: ${editSlot.name} ${ratioLabel(parseFloat(editSlider.value))} ${editSlotB.name}`;
    }
});

// Restore sessions on page load
restoreSessions();
restoreSyxTabs();
loadFavorites();
rebuildPianoDock();
initSyxUpload();
initNightSky({
    onPlay: async (match) => {
        if (match.voice_data) {
            await ensureDx7Synth();
            if (isDx7Ready()) {
                const note = activeKeyboardNote || 60;
                const stop = playNoteRealtime(match.voice_data, note);
                if (stop) setTimeout(stop, 3000);
                const blob = await renderOffline(match.voice_data, note, 3000);
                return URL.createObjectURL(blob);
            }
        }
        return null;
    },
    onSelect: (match) => {
        if (match.voice_data) {
            activeSyxVoiceData = match.voice_data;
            activePianorollPatchId = match.name;
            rebuildPianoDock();
        }
    },
    onEdit: (match) => {
        openEdit({
            id: 'sky_' + Date.now(),
            name: match.name,
            algorithm: match.algorithm,
            feedback: match.feedback,
            distance: match.distance || 0,
            voice_data: match.voice_data || null,
        });
    },
    baseUrl,
});
