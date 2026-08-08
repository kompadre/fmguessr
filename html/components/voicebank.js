// VoiceBank — patch selector for pianoroll
// Each entry has a color that maps to pianoroll note colors.
// Volume overlay: click+drag horizontally to adjust channel volume (0-127).

const STORAGE_KEY = 'fmguessr_voicebank';
const MAX_SLOTS = 8;
const DEFAULT_VOLUME = 100;

const PATCH_COLORS = ['#e94560', '#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#00bcd4', '#ff5722', '#607d8b'];

export function createVoiceBank(opts) {
    const { onSelect, onVoiceAdded, onVolumeChange } = opts;
    let entries = [];
    let activeId = null;

    const container = document.createElement('div');
    container.className = 'voicebank';

    const header = document.createElement('div');
    header.className = 'voicebank-header';
    header.textContent = 'Voices';
    container.appendChild(header);

    const list = document.createElement('div');
    list.className = 'voicebank-list';
    container.appendChild(list);

    // Drop target for canvas patches
    container.addEventListener('dragover', (e) => {
        if (!window._draggedPatch) return;
        e.preventDefault();
        e.stopPropagation();
        container.classList.add('voicebank-drag-over');
        const items = [...list.querySelectorAll('.voicebank-item')];
        items.forEach(i => i.classList.remove('voicebank-drop-target'));
        const afterItem = getInsertAfterItem(items, e.clientY);
        if (afterItem) {
            afterItem.classList.add('voicebank-drop-target');
        } else if (items.length > 0) {
            items[items.length - 1].classList.add('voicebank-drop-target');
        }
    });

    container.addEventListener('dragleave', (e) => {
        if (!container.contains(e.relatedTarget)) {
            container.classList.remove('voicebank-drag-over');
            list.querySelectorAll('.voicebank-drop-target').forEach(i => i.classList.remove('voicebank-drop-target'));
        }
    });

    container.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        container.classList.remove('voicebank-drag-over');
        list.querySelectorAll('.voicebank-drop-target').forEach(i => i.classList.remove('voicebank-drop-target'));
        if (!window._draggedPatch) return;

        const patch = window._draggedPatch;
        const items = [...list.querySelectorAll('.voicebank-item')];
        const afterItem = getInsertAfterItem(items, e.clientY);

        if (afterItem) {
            const idx = parseInt(afterItem.dataset.vbIdx);
            if (entries[idx]) {
                entries[idx].name = patch.name || entries[idx].name;
                entries[idx].voiceData = patch.voice_data || entries[idx].voiceData;
                entries[idx].patchId = patch.name || entries[idx].patchId;
                activeId = entries[idx].id;
                render();
                fireSelect();
                saveState();
            }
        } else {
            addEntry(patch.name || 'Untitled', patch.voice_data || null, patch.name);
        }

        window._draggedPatch = null;
    });

    function getInsertAfterItem(items, clientY) {
        for (const item of items) {
            const rect = item.getBoundingClientRect();
            if (clientY < rect.top + rect.height / 2) return item;
        }
        return null;
    }

    function fireSelect() {
        const active = entries.find(e => e.id === activeId);
        if (onSelect && active) onSelect(active);
    }

    function render() {
        list.innerHTML = '';
        for (let i = 0; i < MAX_SLOTS; i++) {
            const entry = entries[i];
            const item = document.createElement('div');
            item.className = 'voicebank-item' + (entry && entry.id === activeId ? ' active' : '');
            item.dataset.vbIdx = i;

            if (entry) {
                // Volume overlay (fills from left)
                const overlay = document.createElement('div');
                overlay.className = 'voicebank-volume-overlay';
                overlay.style.background = entry.color;
                overlay.style.width = ((entry.volume || DEFAULT_VOLUME) / 127 * 100) + '%';
                item.appendChild(overlay);

                const dot = document.createElement('span');
                dot.className = 'voicebank-dot';
                dot.style.background = entry.color;
                item.appendChild(dot);

                const nameEl = document.createElement('span');
                nameEl.className = 'voicebank-name';
                nameEl.textContent = entry.name;
                item.appendChild(nameEl);

                // Volume label
                const volEl = document.createElement('span');
                volEl.className = 'voicebank-vol';
                volEl.textContent = entry.volume || DEFAULT_VOLUME;
                item.appendChild(volEl);

                // Click to select
                item.addEventListener('click', (e) => {
                    // Don't select if dragging volume
                    if (item._dragging) return;
                    activeId = entry.id;
                    render();
                    fireSelect();
                    saveState();
                });

                // Volume drag: mousedown → track mousemove → mouseup
                item.addEventListener('mousedown', (e) => {
                    if (e.button !== 0) return;
                    const rect = item.getBoundingClientRect();
                    const startX = e.clientX;
                    let didDrag = false;

                    function onMove(ev) {
                        const x = ev.clientX - rect.left;
                        const pct = Math.max(0, Math.min(1, x / rect.width));
                        const vol = Math.round(pct * 127);
                        entry.volume = vol;
                        overlay.style.width = (pct * 100) + '%';
                        volEl.textContent = vol;
                        didDrag = true;
                        if (onVolumeChange) onVolumeChange(entry.channel, vol);
                    }

                    function onUp() {
                        document.removeEventListener('mousemove', onMove);
                        document.removeEventListener('mouseup', onUp);
                        if (didDrag) {
                            item._dragging = true;
                            setTimeout(() => { item._dragging = false; }, 50);
                            saveState();
                        }
                    }

                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup', onUp);
                });
            } else {
                item.classList.add('voicebank-empty');
                const placeholder = document.createElement('span');
                placeholder.className = 'voicebank-placeholder';
                placeholder.textContent = '—';
                item.appendChild(placeholder);
            }

            list.appendChild(item);
        }
    }

    function addEntry(name, voiceData, patchId) {
        const existing = entries.find(e => e.voiceData === voiceData);
        if (existing) {
            activeId = existing.id;
            render();
            fireSelect();
            saveState();
            return;
        }

        if (entries.length >= MAX_SLOTS) return;

        const color = PATCH_COLORS[entries.length % PATCH_COLORS.length];
        entries.push({
            id: 'vb_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
            name: name || 'Untitled',
            voiceData: voiceData || null,
            patchId: patchId || name,
            color,
            channel: entries.length,
            volume: DEFAULT_VOLUME,
        });
        activeId = entries[entries.length - 1].id;
        render();
        fireSelect();
        saveState();

        if (onVoiceAdded && voiceData) {
            onVoiceAdded(voiceData, entries.length - 1);
        }
    }

    function removeEntry(id) {
        entries = entries.filter(e => e.id !== id);
        if (activeId === id) {
            activeId = entries.length > 0 ? entries[0].id : null;
            if (activeId) fireSelect();
        }
        render();
        saveState();
    }

    function getActive() {
        return entries.find(e => e.id === activeId) || null;
    }

    function setActive(id) {
        if (entries.find(e => e.id === id)) {
            activeId = id;
            render();
            fireSelect();
            saveState();
        }
    }

    function saveState() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({ entries, activeId }));
        } catch (e) {
            console.warn('VoiceBank save failed:', e);
        }
    }

    function loadState() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return;
            const data = JSON.parse(raw);
            entries = data.entries || [];
            activeId = data.activeId || null;
            if (activeId && !entries.find(e => e.id === activeId)) {
                activeId = entries.length > 0 ? entries[0].id : null;
            }
            // Backfill volume for entries saved before volume feature
            for (const entry of entries) {
                if (entry.volume === undefined) entry.volume = DEFAULT_VOLUME;
            }
        } catch (e) {
            console.warn('VoiceBank load failed:', e);
        }
    }

    // Init
    loadState();
    render();
    // If entries were restored, fire onSelect to initialize pianoroll's activeSlot
    if (entries.length > 0 && activeId) fireSelect();

    container._addEntry = addEntry;
    container._removeEntry = removeEntry;
    container._clear = () => { entries = []; activeId = null; render(); saveState(); };
    container._getActive = getActive;
    container._setActive = setActive;
    container._getEntries = () => entries;
    container._loadAllVoices = (loadVoiceFn) => {
        for (let i = 0; i < entries.length; i++) {
            if (entries[i] && entries[i].voiceData) {
                entries[i].channel = i;
                loadVoiceFn(entries[i].voiceData, i);
            }
        }
    };

    return container;
}
