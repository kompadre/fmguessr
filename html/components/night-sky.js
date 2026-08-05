// Night Sky — free-floating patch canvas
// Patches can be dragged out of lists onto a persistent canvas.

const STORAGE_KEY = 'fmguessr_night_sky';

let skyEl = null;
let canvasEl = null;
let untethered = []; // { id, left, top, minified, match, containerId }
let containers = []; // { id, name, left, top, width, height }
let onPlayRef = null;
let onSelectRef = null;
let onEditRef = null;
let baseUrlRef = null;

export function initNightSky(opts) {
    skyEl = document.getElementById('night-sky');
    onPlayRef = opts.onPlay;
    onSelectRef = opts.onSelect;
    onEditRef = opts.onEdit;
    baseUrlRef = opts.baseUrl;

    skyEl.innerHTML = '';
    skyEl.classList.add('visible');

    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.className = 'sky-toolbar';
    const ctrBtn = document.createElement('button');
    ctrBtn.className = 'sky-ctr-btn';
    ctrBtn.textContent = '+ Container';
    ctrBtn.title = 'Create a container to group patches';
    ctrBtn.addEventListener('click', () => addContainer('Group', 50, 50));
    toolbar.appendChild(ctrBtn);
    skyEl.appendChild(toolbar);

    canvasEl = document.createElement('div');
    canvasEl.className = 'night-sky-canvas';
    skyEl.appendChild(canvasEl);

    // Load persisted state
    loadState();

    // Re-render
    renderAll();

    // Make canvas a drop target for patches from lists
    canvasEl.addEventListener('dragover', (e) => {
        if (!window._draggedMatch) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        canvasEl.classList.add('drag-over-sky');
    });
    canvasEl.addEventListener('dragleave', (e) => {
        if (!canvasEl.contains(e.relatedTarget)) {
            canvasEl.classList.remove('drag-over-sky');
        }
    });
    canvasEl.addEventListener('drop', (e) => {
        e.preventDefault();
        canvasEl.classList.remove('drag-over-sky');
        if (!window._draggedMatch) return;

        const rect = canvasEl.getBoundingClientRect();
        const left = e.clientX - rect.left + canvasEl.scrollLeft;
        const top = e.clientY - rect.top + canvasEl.scrollTop;

        createFloating(window._draggedMatch, left, top);
        window._draggedRow = null;
        window._draggedMatch = null;
    });
}

function createFloating(match, left, top, id) {
    const entry = {
        id: id || ('sky_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5)),
        left: Math.max(0, left),
        top: Math.max(0, top),
        minified: true,
        match: {
            name: match.name,
            algorithm: match.algorithm,
            feedback: match.feedback,
            distance: match.distance || 0,
            voice_data: match.voice_data || null,
            wav_url: match.wav_url || null,
            rank: match.rank || '',
        },
        containerId: null,
    };
    untethered.push(entry);
    renderFloating(entry);
    saveState();
    return entry;
}

function renderFloating(entry) {
    const el = document.createElement('div');
    el.className = 'patch-float' + (entry.minified ? ' minified' : '');
    el.style.left = entry.left + 'px';
    el.style.top = entry.top + 'px';
    el.dataset.skyId = entry.id;

    // Star
    const star = document.createElement('button');
    star.className = 'patch-star fav-active';
    star.textContent = '★';
    star.title = 'Drag to move';
    el.appendChild(star);

    // Name
    const nameEl = document.createElement('span');
    nameEl.className = 'patch-name';
    nameEl.textContent = entry.match.name || ('#' + entry.match.rank);
    el.appendChild(nameEl);

    if (!entry.minified) {
        appendFullFields(el, entry.match);
    }

    // Drag to reposition (from star)
    let dragging = false;
    let startX, startY, origLeft, origTop;

    star.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
        origLeft = entry.left;
        origTop = entry.top;
        el.style.cursor = 'grabbing';
        el.style.zIndex = 100;
        el.style.transition = 'none';
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        entry.left = Math.max(0, origLeft + dx);
        entry.top = Math.max(0, origTop + dy);
        el.style.left = entry.left + 'px';
        el.style.top = entry.top + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        el.style.cursor = '';
        el.style.zIndex = '';
        el.style.transition = '';
        saveState();
    });

    // Click to expand/collapse (not on star)
    el.addEventListener('click', (e) => {
        if (e.target === star || dragging) return;
        entry.minified = !entry.minified;
        el.classList.toggle('minified', entry.minified);
        if (entry.minified) {
            // Remove full fields
            el.querySelectorAll('.patch-spec, .patch-col, .patch-assign, .patch-edit-btn').forEach(n => n.remove());
        } else {
            appendFullFields(el, entry.match);
        }
        saveState();
    });

    // Double-click star to re-tether (remove from sky)
    star.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        removeFloating(entry.id);
    });

    canvasEl.appendChild(el);
}

function appendFullFields(el, match) {
    const nameEl = el.querySelector('.patch-name');
    if (!nameEl) return;

    // Spec
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

    // Algo
    const algoEl = document.createElement('span');
    algoEl.className = 'patch-col patch-algo';
    algoEl.textContent = match.algorithm;

    // FB
    const fbEl = document.createElement('span');
    fbEl.className = 'patch-col patch-fb';
    fbEl.textContent = match.feedback;

    // Dist
    const distEl = document.createElement('span');
    distEl.className = 'patch-col patch-dist';
    distEl.textContent = (match.distance || 0).toFixed(4);

    // Assign
    const assignBtn = document.createElement('button');
    assignBtn.className = 'patch-assign';
    assignBtn.textContent = '♫';
    assignBtn.title = 'Assign to pianoroll';
    assignBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (onSelectRef) onSelectRef(match);
        const dock = document.getElementById('piano-dock');
        if (dock) dock.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });

    // Append in order after name: spec, algo, fb, dist, assign
    nameEl.insertAdjacentElement('afterend', assignBtn);
    nameEl.insertAdjacentElement('afterend', distEl);
    nameEl.insertAdjacentElement('afterend', fbEl);
    nameEl.insertAdjacentElement('afterend', algoEl);
    nameEl.insertAdjacentElement('afterend', specWrap);

    // Play handler
    if (onPlayRef) {
        playBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            onPlayRef(match);
        });
    }
}

function removeFloating(id) {
    untethered = untethered.filter(u => u.id !== id);
    const el = canvasEl.querySelector(`[data-sky-id="${id}"]`);
    if (el) el.remove();
    saveState();
}

function renderAll() {
    canvasEl.querySelectorAll('.patch-float, .sky-container').forEach(el => el.remove());
    for (const c of containers) renderContainer(c);
    for (const u of untethered) renderFloating(u);
}

function saveState() {
    const data = {
        untethered: untethered.map(u => ({
            id: u.id,
            left: u.left,
            top: u.top,
            minified: u.minified,
            match: u.match,
            containerId: u.containerId,
        })),
        containers: containers.map(c => ({
            id: c.id,
            name: c.name,
            left: c.left,
            top: c.top,
            width: c.width,
            height: c.height,
        })),
    };
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (e) {
        console.warn('Night sky save failed:', e);
    }
}

function loadState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);
        untethered = data.untethered || [];
        containers = data.containers || [];
    } catch (e) {
        console.warn('Night sky load failed:', e);
        untethered = [];
        containers = [];
    }
}

export function addContainer(name, left, top) {
    const c = {
        id: 'ctr_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
        name: name || 'Container',
        left: left || 50,
        top: top || 50,
        width: 220,
        height: 120,
    };
    containers.push(c);
    renderContainer(c);
    saveState();
    return c;
}

function renderContainer(c) {
    const el = document.createElement('div');
    el.className = 'sky-container';
    el.style.left = c.left + 'px';
    el.style.top = c.top + 'px';
    el.style.width = c.width + 'px';
    el.style.height = c.height + 'px';
    el.dataset.containerId = c.id;

    const header = document.createElement('div');
    header.className = 'sky-container-header';
    header.textContent = c.name;
    el.appendChild(header);

    const body = document.createElement('div');
    body.className = 'sky-container-body';
    el.appendChild(body);

    // Drag container by header
    let dragging = false;
    let startX, startY, origLeft, origTop;
    header.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
        origLeft = c.left;
        origTop = c.top;
        el.style.zIndex = 50;
    });
    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        c.left = Math.max(0, origLeft + (e.clientX - startX));
        c.top = Math.max(0, origTop + (e.clientY - startY));
        el.style.left = c.left + 'px';
        el.style.top = c.top + 'px';
    });
    document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        el.style.zIndex = '';
        saveState();
    });

    // Drop patches into container
    body.addEventListener('dragover', (e) => {
        if (!window._draggedMatch) return;
        e.preventDefault();
        e.stopPropagation();
        el.classList.add('drag-over-container');
    });
    body.addEventListener('dragleave', () => {
        el.classList.remove('drag-over-container');
    });
    body.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.classList.remove('drag-over-container');
        if (!window._draggedMatch) return;

        const entry = createFloating(window._draggedMatch, 0, 0);
        entry.containerId = c.id;
        entry.minified = true;

        // Move DOM element into container body
        const floatEl = canvasEl.querySelector(`[data-sky-id="${entry.id}"]`);
        if (floatEl) {
            floatEl.style.left = '';
            floatEl.style.top = '';
            body.appendChild(floatEl);
        }

        window._draggedRow = null;
        window._draggedMatch = null;
        saveState();
    });

    canvasEl.appendChild(el);
}

export function isNightSkyVisible() {
    return skyEl && skyEl.classList.contains('visible');
}
