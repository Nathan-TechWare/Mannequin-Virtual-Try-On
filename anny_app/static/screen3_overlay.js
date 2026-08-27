/* Screen 3 overlay — enhances existing Anny mesh UI without modifying index.html.
   Injects a top nav bar, hides the Cup Size slider, and adds human-readable
   labels alongside Weight/Muscle/Proportions sliders (updated live as user drags). */

(function () {
    'use strict';

    // ── 1. Nav bar ──────────────────────────────────────────────
    function insertNavBar() {
        if (document.getElementById('app-nav')) return;
        const nav = document.createElement('div');
        nav.id = 'app-nav';
        nav.innerHTML = `
            <a href="/">← Upload</a>
            <a href="/measurements">Measurements</a>
            <span class="sep">/</span>
            <span class="current">3D Mesh</span>
        `;
        document.body.insertBefore(nav, document.body.firstChild);
    }

    // ── 2. Formatting helpers (same formulas as Screen 2) ────
    function muscleLabel(v) {
        if (v < 0.2) return "Slim";
        if (v < 0.4) return "Lean";
        if (v < 0.6) return "Average";
        if (v < 0.8) return "Athletic";
        return "Very Muscular";
    }
    function proportionsFormat(v) {
        const torsoRatio = v * 0.2 + 0.4;
        const torsoPct = Math.round(torsoRatio * 100);
        return `${torsoPct}% torso · ${100 - torsoPct}% legs`;
    }
    function kgFromPhenotype(v) {
        return Math.round(v * 80 + 40);
    }

    // ── 3. Slider enhancement ────────────────────────────────
    // Finds the row for a given label substring by scanning slider labels.
    function findRowByLabel(needle) {
        const sliders = document.querySelectorAll('input[type="range"]');
        for (const slider of sliders) {
            // Walk up ancestors to find a row-like container
            let node = slider;
            for (let depth = 0; depth < 6 && node && node !== document.body; depth++) {
                if (node.textContent && node.textContent.toLowerCase().includes(needle.toLowerCase())) {
                    // Ensure this ancestor contains ONLY this slider (not multiple)
                    if (node.querySelectorAll('input[type="range"]').length === 1) {
                        return { row: node, slider: slider };
                    }
                }
                node = node.parentElement;
            }
        }
        return null;
    }

    // Attach a "live label" to a slider that updates as it moves.
    // transformFn takes the slider value (0-1) and returns the display string.
    function attachLiveLabel(row, slider, transformFn, mode = 'append') {
        // Try to find the existing numeric value display element
        // (a leaf element in the row whose text starts with a number)
        let valueEl = null;
        const leaves = row.querySelectorAll('*');
        for (const el of leaves) {
            if (el.children.length === 0 && el !== slider) {
                const t = el.textContent.trim();
                if (/^-?[\d.]+/.test(t) && t.length < 20) {
                    valueEl = el;
                    break;
                }
            }
        }

        // Create our own label element
        const overlayLabel = document.createElement('span');
        overlayLabel.className = 'overlay-label';

        // Placement: prefer next to the numeric value; else append to row
        if (valueEl && valueEl.parentElement) {
            valueEl.parentElement.appendChild(overlayLabel);
        } else {
            row.appendChild(overlayLabel);
        }

        const update = () => {
            const v = parseFloat(slider.value);
            const text = transformFn(v);
            overlayLabel.textContent = text;
        };
        slider.addEventListener('input', update);
        update();

        // Some UIs update value display asynchronously — re-check on click too
        slider.addEventListener('change', update);
    }

    // ── 4. Cup size hide ─────────────────────────────────────
    function hideCupSize() {
        const hit = findRowByLabel('cup size');
        if (hit) hit.row.classList.add('overlay-hidden');
    }

    // ── 5. Set up all enhancements once sliders exist ────────
    function enhance() {
        const sliders = document.querySelectorAll('input[type="range"]');
        if (sliders.length === 0) return false;

        // Weight
        const weight = findRowByLabel('weight');
        if (weight) attachLiveLabel(weight.row, weight.slider,
            v => `${kgFromPhenotype(v)} kg`);

        // Muscle
        const muscle = findRowByLabel('muscle');
        if (muscle) attachLiveLabel(muscle.row, muscle.slider,
            v => muscleLabel(v));

        // Proportions
        const prop = findRowByLabel('proportions');
        if (prop) attachLiveLabel(prop.row, prop.slider,
            v => proportionsFormat(v));

        hideCupSize();
        return true;
    }

    // ── 6. Bootstrap: wait for sliders to appear ─────────────
    // Sliders may be created asynchronously (after /initial_values fetch).
    // Poll every 100ms for up to 15 seconds.
    function init() {
        insertNavBar();
        let attempts = 0;
        const maxAttempts = 150;
        const timer = setInterval(() => {
            attempts++;
            if (enhance() || attempts >= maxAttempts) {
                clearInterval(timer);
                if (attempts >= maxAttempts) {
                    console.warn('[overlay] Could not find sliders after 15s; enhancements skipped.');
                }
            }
        }, 100);

        // Also watch for future DOM changes (e.g. sliders re-rendered on "Reset")
        const observer = new MutationObserver(() => {
            // Only re-enhance if our labels are gone
            if (!document.querySelector('.overlay-label')) enhance();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
