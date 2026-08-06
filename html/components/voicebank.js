// VoiceBank — patch selector for pianoroll
// Each entry has a color that maps to pianoroll note colors.

const STORAGE_KEY = 'fmguessr_voicebank';
const MAX_SLOTS = 8;

const PATCH_COLORS = ['#e94560', '#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#00bcd4', '#ff5722', '#607d8b'];

export function createVoiceBank(opts) {
    const { onSelect } = opts;
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

    // Drop target for night sky patches
    container.addEventListener('dragover', (e) => {
        if (!window._draggedMatch) return;
        e.preventDefault();
        e.stopPropagation();
        container.classList.add('voicebank-drag-over');
        // Highlight drop target slot
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
        if (!window._draggedMatch) return;

        const match = window._draggedMatch;
        const items = [...list.querySelectorAll('.voicebank-item')];
        const afterItem = getInsertAfterItem(items, e.clientY);

        if (afterItem) {
            // Replace the entry at drop position
            const idx = parseInt(afterItem.dataset.vbIdx);
            if (entries[idx]) {
                entries[idx].name = match.name || entries[idx].name;
                entries[idx].voiceData = match.voice_data || entries[idx].voiceData;
                entries[idx].patchId = match.name || entries[idx].patchId;
                activeId = entries[idx].id;
                render();
                fireSelect();
                saveState();
            }
        } else {
            // Append to first empty slot
            addEntry(match.name || 'Untitled', match.voice_data || null, match.name);
        }

        window._draggedRow = null;
        window._draggedMatch = null;
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
                const dot = document.createElement('span');
                dot.className = 'voicebank-dot';
                dot.style.background = entry.color;
                item.appendChild(dot);

                const nameEl = document.createElement('span');
                nameEl.className = 'voicebank-name';
                nameEl.textContent = entry.name;
                item.appendChild(nameEl);

                item.addEventListener('click', () => {
                    activeId = entry.id;
                    render();
                    fireSelect();
                    saveState();
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
        // Dedup by voiceData
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
        });
        activeId = entries[entries.length - 1].id;
        render();
        fireSelect();
        saveState();
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
            // Ensure activeId is valid
            if (activeId && !entries.find(e => e.id === activeId)) {
                activeId = entries.length > 0 ? entries[0].id : null;
            }
        } catch (e) {
            console.warn('VoiceBank load failed:', e);
        }
    }

    // Init
    loadState();
    render();

    container._addEntry = addEntry;
    container._removeEntry = removeEntry;
    container._getActive = getActive;
    container._setActive = setActive;
    container._getEntries = () => entries;

    return container;
}
