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
    {
        id: 'load-syx',
        title: 'Load a .SYX bank',
        openMenu: 'file',
        illustration: null,
        description:
            'Load a standard DX7 SysEx file (.SYX) containing up to <strong>32 voices</strong>.<br><br>' +
            '<strong>What happens:</strong><br>' +
            '1. The .SYX file is parsed into individual 128-byte patches<br>' +
            '2. All non-empty voices appear in a new <strong>container</strong> on the canvas<br>' +
            '3. Click any patch to preview its sound<br>' +
            '4. Use the <strong>♫</strong> button to assign patches to the voice bank for use in the piano roll<br><br>' +
            'Try loading the built-in <strong>DEMO.SYX</strong> if you haven\'t already!',
    },
    {
        id: 'containers',
        title: 'Patch Containers',
        highlight: '.sky-container',
        fallbackHighlight: '#night-sky',
        noBlur: true,
        illustration: null,
        description:
            'Patches live inside <strong>containers</strong> on the canvas — the dark starfield area.<br><br>' +
            '<strong>Interactions:</strong><br>' +
            '• <strong>Click</strong> a patch to preview its sound<br>' +
            '• <strong>♫</strong> assigns patches to the voice bank (up to 8) for the piano roll<br>' +
            '• <strong>★</strong> drag handle — drag patches between containers or onto the canvas<br>' +
            '• <strong>↕</strong> sorts patches by algorithm number<br>' +
            '• <strong>💾</strong> exports the container as a .SYX file<br>' +
            '• Drag the <strong>header</strong> to move a container, drag the <strong>bottom-right corner</strong> to resize<br><br>' +
            'Your DEMO container is a good starting point — try clicking some patches!',
    },
    // Future steps:
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

export function restartTutorial() {
    localStorage.removeItem(STORAGE_KEY);
    currentStep = 0;
    active = true;
    showStep(currentStep);
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

    // Blur page content (unless step opts out — e.g. canvas steps)
    var pageContent = document.getElementById('page-content');
    if (pageContent && !step.noBlur) pageContent.classList.add('page-blurred');

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

    // Highlight target element (with fallback)
    var highlightTarget = null;
    var usedFallback = false;
    if (step.highlight) {
        highlightTarget = document.querySelector(step.highlight);
        if (!highlightTarget && step.fallbackHighlight) {
            highlightTarget = document.querySelector(step.fallbackHighlight);
            usedFallback = true;
        }
        if (highlightTarget) {
            highlightTarget.classList.add('tutorial-highlight');
        }
    }

    // Position modal relative to highlight target
    var dialog = modal.querySelector('.tutorial-dialog');
    if (dialog) {
        // Reset position classes/styles
        dialog.classList.remove('tutorial-centered');
        dialog.style.margin = '';

        if (usedFallback) {
            // No container found — center the modal
            dialog.classList.add('tutorial-centered');
        } else if (highlightTarget && highlightTarget.classList.contains('sky-container')) {
            // Position modal to the left or right of the container
            var rect = highlightTarget.getBoundingClientRect();
            var screenCenter = window.innerWidth / 2;
            var containerCenter = rect.left + rect.width / 2;

            if (containerCenter < screenCenter) {
                // Container on the left — modal to the right
                dialog.style.margin = '48px 0 0 ' + (rect.right + 16) + 'px';
            } else {
                // Container on the right — modal to the left
                dialog.style.margin = '48px 0 0 ' + Math.max(8, rect.left - 436) + 'px';
            }
        }
        // else: default positioning (near File menu) — no changes needed
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
