// Tutorial — guided tour of app features.
// Each step highlights an element, opens a menu, and shows a modal.
// Steps are extensible: add objects to STEPS array for new tutorial pages.

const STORAGE_KEY = 'fmguessr_tutorial';

const STEPS = [
    {
        id: 'import-wav',
        title: 'Import a .WAV file',
        openMenu: 'file',
        illustration: null,
        description:
            'Import any audio recording to find its closest DX7 match.<br><br>' +
            '<strong>The process:</strong><br>' +
            '1. Select a <strong>.WAV</strong> file from your computer<br>' +
            '2. The server <em>probes</em> it — analyzing attack, decay, brightness, and harmonicity<br>' +
            '3. You review and adjust the detected parameters<br>' +
            '4. <em>Matching</em> begins — genetic algorithms search for the best DX7 patch<br>' +
            '5. Results appear as patches in a container on the canvas<br><br>' +
            'Each patch is a 128-byte DX7 voice that reproduces the original sound using FM synthesis.',
    },
    // Future steps:
    // { id: 'load-syx', title: 'Load .SYX', openMenu: 'file', ... },
    // { id: 'canvas', title: 'The Canvas', highlight: '#night-sky', ... },
    // { id: 'pianoroll', title: 'Piano Roll', ... },
    // { id: 'edit', title: 'Edit Patches', ... },
    // { id: 'export', title: 'Export .SYX', openMenu: 'file', ... },
];

let currentStep = 0;
let active = false;

export function isTutorialActive() {
    return active;
}

export function startTutorial() {
    if (localStorage.getItem(STORAGE_KEY)) return;
    if (document.querySelectorAll('.menu-item.open').length > 0) return;
    currentStep = 0;
    active = true;
    showStep(currentStep);
}

export function skipTutorial() {
    localStorage.setItem(STORAGE_KEY, 'skipped');
    hide();
}

function completeTutorial() {
    localStorage.setItem(STORAGE_KEY, 'completed');
    hide();
}

function showStep(index) {
    const step = STEPS[index];
    if (!step) { completeTutorial(); return; }

    const modal = document.getElementById('tutorial-modal');
    document.getElementById('tutorial-title').textContent = step.title;
    document.getElementById('tutorial-description').innerHTML = step.description;
    document.getElementById('tutorial-step-indicator').textContent =
        'Step ' + (index + 1) + ' of ' + STEPS.length;

    // Illustration
    const illust = document.getElementById('tutorial-illustration');
    if (step.illustration) {
        illust.innerHTML = step.illustration;
        illust.style.display = '';
    } else {
        illust.innerHTML = '<div class="tutorial-illustration-placeholder">\uD83D\uDCCA</div>';
        illust.style.display = '';
    }

    // Clear previous state
    clearHighlights();

    // Blur page content
    var pageContent = document.getElementById('page-content');
    if (pageContent) pageContent.classList.add('page-blurred');

    // Lift menu bar above the overlay
    var menuBar = document.querySelector('.menu-bar');
    if (menuBar) menuBar.classList.add('tutorial-highlight');

    // Open menu if specified
    if (step.openMenu) {
        var menuItem = document.querySelector('.menu-item[data-menu="' + step.openMenu + '"]');
        if (menuItem) {
            menuItem.classList.add('open');
            menuItem.dataset.tutorialOpen = '1';
        }
    }

    // Update nav buttons
    var nextBtn = document.getElementById('tutorial-next');
    nextBtn.textContent = index < STEPS.length - 1 ? 'Next \u2192' : 'Got it!';

    modal.style.display = '';
}

function hide() {
    active = false;
    var modal = document.getElementById('tutorial-modal');
    modal.style.display = 'none';
    clearHighlights();
}

function clearHighlights() {
    // Remove blur from page content
    var pageContent = document.getElementById('page-content');
    if (pageContent) pageContent.classList.remove('page-blurred');

    // Remove menu bar highlight
    var menuBar = document.querySelector('.menu-bar');
    if (menuBar) menuBar.classList.remove('tutorial-highlight');

    // Close menus we opened (but not ones the user opened manually)
    document.querySelectorAll('.menu-item').forEach(function(el) {
        if (el.dataset.tutorialOpen) {
            el.classList.remove('open');
            delete el.dataset.tutorialOpen;
        }
    });
}

export function initTutorial() {
    document.getElementById('tutorial-close').addEventListener('click', skipTutorial);
    document.getElementById('tutorial-skip').addEventListener('click', skipTutorial);
    document.getElementById('tutorial-next').addEventListener('click', function() {
        currentStep++;
        if (currentStep >= STEPS.length) {
            completeTutorial();
        } else {
            showStep(currentStep);
        }
    });

    // Click outside modal dismisses
    document.getElementById('tutorial-modal').addEventListener('click', function(e) {
        if (e.target === e.currentTarget) skipTutorial();
    });
}
