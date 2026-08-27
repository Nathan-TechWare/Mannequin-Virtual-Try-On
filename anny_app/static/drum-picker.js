/* Drum picker — iOS-style scrolling wheel selector.
   Two classes: DrumWheel (single wheel), DrumPicker (two wheels: feet + inches). */

class DrumWheel {
    constructor(container, values, initialValue) {
        this.container = container;
        this.values = values;
        this.itemHeight = 40; // must match .drum-item height in CSS
        this._render();
        this.setValue(initialValue);

        this._scrollTimer = null;
        this.container.addEventListener('scroll', () => {
            clearTimeout(this._scrollTimer);
            // Update the "selected" highlight live while scrolling
            this._updateHighlight();
            // Snap after scrolling stops
            this._scrollTimer = setTimeout(() => this._snap(), 120);
        });
    }

    _render() {
        this.container.innerHTML = this.values
            .map(v => `<div class="drum-item" data-value="${v}">${v}</div>`)
            .join('');
    }

    _centeredIndex() {
        return Math.round(this.container.scrollTop / this.itemHeight);
    }

    _clampIndex(i) {
        return Math.max(0, Math.min(this.values.length - 1, i));
    }

    _updateHighlight() {
        const idx = this._clampIndex(this._centeredIndex());
        const items = this.container.querySelectorAll('.drum-item');
        items.forEach((el, i) => el.classList.toggle('selected', i === idx));
    }

    _snap() {
        const idx = this._clampIndex(this._centeredIndex());
        const targetTop = idx * this.itemHeight;
        // Only scroll if actually off-target (prevents infinite bounce)
        if (Math.abs(this.container.scrollTop - targetTop) > 1) {
            this.container.scrollTo({ top: targetTop, behavior: 'smooth' });
        }
        this._value = this.values[idx];
        this._updateHighlight();
        if (this.onChange) this.onChange(this._value);
    }

    setValue(value) {
        const idx = this.values.indexOf(value);
        if (idx === -1) return;
        this._value = value;
        this.container.scrollTop = idx * this.itemHeight;
        this._updateHighlight();
    }

    getValue() {
        return this._value;
    }
}


class DrumPicker {
    constructor(container, options) {
        // options: { feet: {values, initial}, inches: {values, initial} }
        this.container = container;
        container.classList.add('drum-picker');
        container.innerHTML = `
            <div class="drum-wheel" data-name="feet"></div>
            <span class="drum-label">ft</span>
            <div class="drum-wheel" data-name="inches"></div>
            <span class="drum-label">in</span>
        `;
        const feetEl   = container.querySelector('[data-name="feet"]');
        const inchesEl = container.querySelector('[data-name="inches"]');
        this.feet   = new DrumWheel(feetEl,   options.feet.values,   options.feet.initial);
        this.inches = new DrumWheel(inchesEl, options.inches.values, options.inches.initial);

        // Forward changes on either wheel
        const forward = () => { if (this.onChange) this.onChange(this.getCm()); };
        this.feet.onChange   = forward;
        this.inches.onChange = forward;
    }

    getFeetInches() {
        return { feet: this.feet.getValue(), inches: this.inches.getValue() };
    }

    getCm() {
        const { feet, inches } = this.getFeetInches();
        return Math.round((feet * 12 + inches) * 2.54);
    }
}
