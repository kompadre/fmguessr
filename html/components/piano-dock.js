import { createPianoroll } from './pianoroll.js';

const PIANO_NOTES = [
    { midi: 36, name: 'C2',  white: true },
    { midi: 37, name: 'C#2', white: false },
    { midi: 38, name: 'D2',  white: true },
    { midi: 39, name: 'D#2', white: false },
    { midi: 40, name: 'E2',  white: true },
    { midi: 41, name: 'F2',  white: true },
    { midi: 42, name: 'F#2', white: false },
    { midi: 43, name: 'G2',  white: true },
    { midi: 44, name: 'G#2', white: false },
    { midi: 45, name: 'A2',  white: true },
    { midi: 46, name: 'A#2', white: false },
    { midi: 47, name: 'B2',  white: true },
    { midi: 48, name: 'C3',  white: true },
    { midi: 49, name: 'C#3', white: false },
    { midi: 50, name: 'D3',  white: true },
    { midi: 51, name: 'D#3', white: false },
    { midi: 52, name: 'E3',  white: true },
    { midi: 53, name: 'F3',  white: true },
    { midi: 54, name: 'F#3', white: false },
    { midi: 55, name: 'G3',  white: true },
    { midi: 56, name: 'G#3', white: false },
    { midi: 57, name: 'A3',  white: true },
    { midi: 58, name: 'A#3', white: false },
    { midi: 59, name: 'B3',  white: true },
    { midi: 60, name: 'C4',  white: true },
    { midi: 61, name: 'C#4', white: false },
    { midi: 62, name: 'D4',  white: true },
    { midi: 63, name: 'D#4', white: false },
    { midi: 64, name: 'E4',  white: true },
    { midi: 65, name: 'F4',  white: true },
    { midi: 66, name: 'F#4', white: false },
    { midi: 67, name: 'G4',  white: true },
    { midi: 68, name: 'G#4', white: false },
    { midi: 69, name: 'A4',  white: true },
    { midi: 70, name: 'A#4', white: false },
    { midi: 71, name: 'B4',  white: true },
    { midi: 72, name: 'C5',  white: true },
    { midi: 73, name: 'C#5', white: false },
    { midi: 74, name: 'D5',  white: true },
    { midi: 75, name: 'D#5', white: false },
    { midi: 76, name: 'E5',  white: true },
    { midi: 77, name: 'F5',  white: true },
    { midi: 78, name: 'F#5', white: false },
    { midi: 79, name: 'G5',  white: true },
    { midi: 80, name: 'G#5', white: false },
    { midi: 81, name: 'A5',  white: true },
    { midi: 82, name: 'A#5', white: false },
    { midi: 83, name: 'B5',  white: true },
];

export function createPianoDock(opts) {
    const { matchedMidi, playNoteFn, pianorollNotes, activePatchId, initialTab } = opts;
    const disabled = !playNoteFn;

    const wrapper = document.createElement('div');
    wrapper.className = 'piano-dock-inner' + (disabled ? ' piano-dock-disabled' : '');

    // Tab bar
    const tabBar = document.createElement('div');
    tabBar.className = 'dock-tabs';

    const tabKeyboard = document.createElement('button');
    tabKeyboard.className = 'dock-tab' + (initialTab === 'keyboard' ? ' active' : '');
    tabKeyboard.textContent = 'Keyboard';

    const tabPianoroll = document.createElement('button');
    tabPianoroll.className = 'dock-tab' + (initialTab !== 'keyboard' ? ' active' : '');
    tabPianoroll.textContent = 'Piano Roll';

    tabBar.appendChild(tabKeyboard);
    tabBar.appendChild(tabPianoroll);
    wrapper.appendChild(tabBar);

    // Panels
    const keyboardPanel = document.createElement('div');
    keyboardPanel.className = 'dock-panel' + (initialTab === 'keyboard' ? ' active' : '');

    const pianorollPanel = document.createElement('div');
    pianorollPanel.className = 'dock-panel' + (initialTab !== 'keyboard' ? ' active' : '');

    wrapper.appendChild(keyboardPanel);
    wrapper.appendChild(pianorollPanel);

    function showKeyboard() {
        tabKeyboard.classList.add('active');
        tabPianoroll.classList.remove('active');
        keyboardPanel.classList.add('active');
        pianorollPanel.classList.remove('active');
    }
    function showPianoroll() {
        tabPianoroll.classList.add('active');
        tabKeyboard.classList.remove('active');
        pianorollPanel.classList.add('active');
        keyboardPanel.classList.remove('active');
    }

    tabKeyboard.addEventListener('click', showKeyboard);
    tabPianoroll.addEventListener('click', showPianoroll);

    // --- Keyboard panel ---
    const note = matchedMidi || 60;

    const label = document.createElement('div');
    label.className = 'keyboard-label';
    label.textContent = disabled ? 'Load a patch to play' : 'Click to preview note';
    keyboardPanel.appendChild(label);

    const kb = document.createElement('div');
    kb.className = 'keyboard';

    const keyElements = {};
    PIANO_NOTES.forEach(n => {
        const key = document.createElement('div');
        key.className = 'key ' + (n.white ? 'white' : 'black');
        key.title = n.name;
        if (n.midi === note) key.classList.add('matched');
        if (n.white) key.textContent = n.name;

        if (!disabled) {
            let pressed = false;
            key.addEventListener('mousedown', (e) => {
                e.preventDefault();
                pressed = true;
                key.classList.add('pressed');
                playNoteFn(n.midi);
            });
            key.addEventListener('mouseup', () => { pressed = false; key.classList.remove('pressed'); });
            key.addEventListener('mouseleave', () => { if (pressed) key.classList.remove('pressed'); });
        }

        keyElements[n.midi] = key;
        kb.appendChild(key);
    });

    keyboardPanel.appendChild(kb);

    function updateHighlights() {
        const midiSet = new Set((pianorollNotes || []).map(n => n.midi));
        for (const [midi, key] of Object.entries(keyElements)) {
            const m = parseInt(midi);
            if (m >= 60 && m <= 77) {
                key.classList.toggle('pianoroll-active', midiSet.has(m));
            }
        }
    }
    updateHighlights();

    // --- Pianoroll panel ---
    let pianorollRef = null;
    if (!disabled) {
        pianorollRef = createPianoroll({
            notes: pianorollNotes,
            playNoteFn,
            onNotesChange: updateHighlights,
            activePatchId,
        });
        pianorollPanel.appendChild(pianorollRef.element);
    } else {
        const placeholder = document.createElement('div');
        placeholder.className = 'pianoroll-placeholder';
        placeholder.textContent = 'Load a patch to use Piano Roll';
        pianorollPanel.appendChild(placeholder);
    }

    wrapper._updateHighlights = updateHighlights;
    wrapper._pianoroll = pianorollRef;
    wrapper._showKeyboard = showKeyboard;
    wrapper._showPianoroll = showPianoroll;
    return wrapper;
}
