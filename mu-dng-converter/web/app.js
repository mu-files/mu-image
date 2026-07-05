// mu DNG Converter - JavaScript UI Controller

const WB_PRESETS = {
    as_shot: [null, null],
    d50: [5000, 0],
    daylight: [5500, 10],
    cloudy: [6500, 10],
    shade: [7500, 10],
    tungsten: [2850, 0],
    fluorescent: [3800, 21],
    flash: [5500, 0],
    custom: [null, null],
};

class DNGConverter {
    constructor() {
        this.currentTab = 'create-dng';
        this.inputPaths = {};   // tab -> array of selected paths
        this.inputModes = {};   // tab -> 'folder' | 'files'
        this.outputPaths = {};  // tab -> output folder
        this.metadataOps = {};  // tab -> array of {type, name, value, ...}
        this.init();
    }

    init() {
        this.setupTabSwitching();
        this.setupButtonHandlers();
        this.setupCollapsibleSections();
        this.setupCreateDNGHandlers();
        this.setupRenderDNGHandlers();
        console.log('DNG Converter initialized');
    }

    setupTabSwitching() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabPanes = document.querySelectorAll('.tab-pane');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;
                
                // Remove active class from all buttons and panes
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabPanes.forEach(pane => pane.classList.remove('active'));
                
                // Add active class to clicked button and corresponding pane
                button.classList.add('active');
                document.getElementById(targetTab).classList.add('active');
                
                this.currentTab = targetTab;
                console.log(`Switched to tab: ${targetTab}`);
            });
        });
    }

    setupButtonHandlers() {
        ['create-dng', 'render-dng'].forEach(tab => {
            document.getElementById(`${tab}-run`).addEventListener('click', () => this.handleRun(tab));
            document.getElementById(`${tab}-cancel`).addEventListener('click', () => this.handleCancel(tab));
            document.getElementById(`${tab}-input-btn`).addEventListener('click', () => this.handleInputSelect(tab));
            document.getElementById(`${tab}-output-btn`).addEventListener('click', () => this.handleOutputSelect(tab));
        });
    }

    gatherSettings(tab) {
        const val = id => document.getElementById(id).value;
        const chk = id => document.getElementById(id).checked;
        const settings = {
            input: this.inputPaths[tab] || [],
            inputMode: this.inputModes[tab] || 'folder',
            output: this.outputPaths[tab] || '',
        };
        if (tab === 'create-dng') {
            settings.inputType = val('create-dng-input-type');
            settings.transcode = chk('create-dng-transcode');
            settings.compression = val('create-dng-compression');
            settings.jxlDistance = val('create-dng-jxl-distance');
            settings.jxlEffort = val('create-dng-jxl-effort');
            settings.demosaic = chk('create-dng-demosaic');
            settings.scale = val('create-dng-scale');
            settings.preview = chk('create-dng-preview');
            settings.fastLoad = chk('create-dng-fast-load');
            settings.numWorkers = val('create-dng-num-workers');
            settings.metadataOps = this.metadataOps[tab] || [];
            // FITS rendering parameters
            settings.autoExposure = chk('create-dng-auto-exposure');
            settings.toneCurve = chk('create-dng-tone-curve');
            settings.wbPreset = val('create-dng-wb');
            settings.temperature = val('create-dng-temperature');
            settings.tint = val('create-dng-tint');
            settings.exposure = val('create-dng-exposure');
        } else if (tab === 'render-dng') {
            settings.mode = val('render-dng-conversion-mode');
            settings.useXmp = chk('render-dng-use-xmp');
            settings.wbPreset = val('render-dng-wb');
            settings.temperature = val('render-dng-temperature');
            settings.tint = val('render-dng-tint');
            settings.exposure = val('render-dng-exposure');
            settings.bitDepth = val('render-dng-bit-depth');
            settings.scale = val('render-dng-scale');
            settings.numWorkers = val('render-dng-num-workers');
            settings.resolution = val('render-dng-resolution');
            settings.codec = val('render-dng-codec');
            settings.crf = val('render-dng-crf');
            settings.frameRate = val('render-dng-frame-rate');
            settings.videoBitDepth = val('render-dng-video-bit-depth');
            settings.overlay = chk('render-dng-overlay');
        }
        return settings;
    }

    async handleRun(tab) {
        console.log(`Run clicked for ${tab}`);
        const runBtn = document.getElementById(`${tab}-run`);
        const cancelBtn = document.getElementById(`${tab}-cancel`);
        
        // Show cancel button
        runBtn.style.display = 'none';
        cancelBtn.style.display = 'flex';
        
        // Call Python backend
        try {
            if (window.pywebview && window.pywebview.api) {
                const settings = this.gatherSettings(tab);
                let result = await window.pywebview.api.run_conversion(tab, settings);
                if (result && result.status === 'confirm-overwrite') {
                    const action = await this.showOverwriteDialog(result.existing, result.total);
                    if (action !== 'cancel') {
                        settings.overwriteAction = action;
                        result = await window.pywebview.api.run_conversion(tab, settings);
                    }
                }
            } else {
                // Fallback for testing without pywebview
                await new Promise(r => setTimeout(r, 2000));
            }
        } catch (error) {
            console.error('Error running:', error);
            this.appendLog(tab, 'ERROR: ' + error.message);
        }
        // Restore buttons and reset progress bar to pre-run state
        runBtn.style.display = 'flex';
        cancelBtn.style.display = 'none';
        this.updateProgressBar(tab, 0);
    }

    showOverwriteDialog(existing, total) {
        return new Promise(resolve => {
            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay';
            overlay.innerHTML =
                '<div class="modal">' +
                '<p class="modal-title">Files already exist</p>' +
                `<p class="modal-message">${existing} of ${total} output files already exist in the destination folder.</p>` +
                '<div class="modal-buttons">' +
                '<button class="modal-overwrite">Overwrite All</button>' +
                '<button class="modal-skip">Skip Existing</button>' +
                '<button class="modal-cancel">Cancel</button>' +
                '</div></div>';
            document.body.appendChild(overlay);
            const done = action => { overlay.remove(); resolve(action); };
            overlay.querySelector('.modal-overwrite').addEventListener('click', () => done('overwrite'));
            overlay.querySelector('.modal-skip').addEventListener('click', () => done('skip'));
            overlay.querySelector('.modal-cancel').addEventListener('click', () => done('cancel'));
        });
    }

    handleCancel(tab) {
        console.log(`Cancel clicked for ${tab}`);
        // Signal Python backend; run_conversion will return and handleRun restores the UI
        if (window.pywebview && window.pywebview.api) {
            window.pywebview.api.cancel_conversion(tab);
        }
    }

    async handleInputSelect(tab) {
        console.log(`Input select clicked for ${tab}`);
        const modeEl = document.getElementById(`${tab}-mode`);
        const mode = modeEl ? modeEl.value : 'folder';
        const fileType = tab === 'create-dng'
            ? document.getElementById('create-dng-input-type').value
            : 'dng';
        
        try {
            if (window.pywebview && window.pywebview.api) {
                const result = await window.pywebview.api.select_input(tab, mode, fileType);
                this.updateInputPath(tab, result);
            } else {
                // Fallback for testing
                this.updateInputPath(tab, '/mock/input/path');
            }
        } catch (error) {
            console.error('Error selecting input:', error);
        }
    }

    async handleOutputSelect(tab) {
        console.log(`Output select clicked for ${tab}`);
        let outMode = 'folder';
        if (tab === 'render-dng' &&
            document.getElementById('render-dng-conversion-mode').value === 'video') {
            outMode = 'file';
        }
        
        try {
            if (window.pywebview && window.pywebview.api) {
                const result = await window.pywebview.api.select_output(tab, outMode);
                this.updateOutputPath(tab, result);
            } else {
                // Fallback for testing
                this.updateOutputPath(tab, '/mock/output/path');
            }
        } catch (error) {
            console.error('Error selecting output:', error);
        }
    }

    updateInputPath(tab, path) {
        const pathElement = document.getElementById(`${tab}-input-path`);
        if (pathElement) {
            if (!path) {
                pathElement.textContent = 'No folder selected';
                pathElement.title = '';
                return;
            }
            const paths = path.split('\n').filter(p => p.trim());
            this.inputPaths[tab] = paths;
            const modeEl = document.getElementById(`${tab}-mode`);
            this.inputModes[tab] = modeEl ? modeEl.value : 'folder';
            if (paths.length === 1) {
                pathElement.textContent = paths[0];
                pathElement.title = paths[0];
            } else {
                const folder = this.commonFolder(paths);
                pathElement.textContent = `${paths.length} files selected (${folder})`;
                pathElement.title = folder;
            }
        }
    }

    commonFolder(paths) {
        if (paths.length === 0) return '';
        if (paths.length === 1) return paths[0];
        const parts = paths.map(p => p.split('/').filter(Boolean));
        let common = [];
        const minLen = Math.min(...parts.map(p => p.length));
        for (let i = 0; i < minLen; i++) {
            const part = parts[0][i];
            if (parts.every(p => p[i] === part)) {
                common.push(part);
            } else {
                break;
            }
        }
        if (common.length === 0) return '';
        // On macOS/Linux absolute paths start with /; reconstruct with leading slash
        const isAbsolute = paths[0].startsWith('/');
        return (isAbsolute ? '/' : '') + common.join('/');
    }

    updateOutputPath(tab, path) {
        const pathElement = document.getElementById(`${tab}-output-path`);
        if (pathElement) {
            if (path) this.outputPaths[tab] = path;
            pathElement.textContent = path || 'No folder selected';
            pathElement.title = path || '';
        }
    }

    // Method to update progress from Python backend
    updateProgress(tab, message) {
        const progressText = document.getElementById(`${tab}-progress`);
        if (progressText) {
            progressText.textContent = message;
        }
    }

    // Method to update progress bar from Python backend
    updateProgressBar(tab, value) {
        const progressBar = document.getElementById(`${tab}-progress-bar`);
        if (progressBar) {
            progressBar.style.width = `${value}%`;
        }
    }

    // Method to append log messages from Python backend
    appendLog(tab, message) {
        const logArea = document.getElementById(`${tab}-log`);
        if (logArea) {
            logArea.textContent += message + '\n';
            logArea.scrollTop = logArea.scrollHeight;
        }
    }

    setupCollapsibleSections() {
        const sectionTitles = document.querySelectorAll('.section-title');
        
        sectionTitles.forEach(title => {
            title.addEventListener('click', () => {
                const content = title.nextElementSibling;
                const isExpanded = title.classList.contains('expanded');
                
                if (isExpanded) {
                    title.classList.remove('expanded');
                    content.style.display = 'none';
                } else {
                    title.classList.add('expanded');
                    content.style.display = 'flex';
                }
            });
        });
    }

    setupCreateDNGHandlers() {
        document.getElementById('create-dng-apply-metadata').addEventListener('click', () => {
            this.handleApplyMetadata();
        });

        const transcodeCheckbox = document.getElementById('create-dng-transcode');
        const transcodeControls = [
            'create-dng-compression', 'create-dng-jxl-distance', 'create-dng-jxl-effort',
            'create-dng-demosaic', 'create-dng-demosaic-algo', 'create-dng-scale',
        ];
        const updateTranscodeState = () => {
            const enabled = transcodeCheckbox.checked;
            transcodeControls.forEach(id => {
                document.getElementById(id).disabled = !enabled;
            });
            if (enabled) {
                updateCompressionState();
                updateDemosaicState();
            }
        };

        const updateCompressionState = () => {
            const compression = document.getElementById('create-dng-compression').value;
            const isLossy = compression === 'jxl_lossy';
            const isJxl = compression !== 'uncompressed';
            const distField = document.getElementById('create-dng-jxl-distance');
            const effortField = document.getElementById('create-dng-jxl-effort');
            distField.disabled = !isLossy;
            if (!isLossy) distField.value = '0';
            effortField.disabled = !isJxl;
        };

        const updateDemosaicState = () => {
            const demosaicEnabled = document.getElementById('create-dng-demosaic').checked;
            document.getElementById('create-dng-demosaic-algo').disabled = !demosaicEnabled;
            document.getElementById('create-dng-scale').disabled = !demosaicEnabled;
        };

        transcodeCheckbox.addEventListener('change', updateTranscodeState);
        document.getElementById('create-dng-compression').addEventListener('change', updateCompressionState);
        document.getElementById('create-dng-demosaic').addEventListener('change', updateDemosaicState);

        // Input type (DNG / FITS) switching
        const typeSel = document.getElementById('create-dng-input-type');
        typeSel.addEventListener('change', () => {
            const isFits = typeSel.value === 'fits';
            document.getElementById('create-dng-rendering-section')
                .classList.toggle('section-hidden', !isFits);
            document.getElementById('create-dng-transcode-label').textContent =
                isFits ? 'Encode' : 'Re-encode';
            document.getElementById('create-dng-transcode-section-title').textContent =
                isFits ? 'Encode Options' : 'Transcode Options';
            // Clear input selection since the file type changed
            this.inputPaths['create-dng'] = [];
            this.updateInputPath('create-dng', '');
        });

        // FITS rendering parameter handlers
        const autoExp = document.getElementById('create-dng-auto-exposure');
        autoExp.addEventListener('change', () => {
            document.getElementById('create-dng-exposure').disabled = autoExp.checked;
        });
        document.getElementById('create-dng-wb').addEventListener('change', () => {
            this.applyWBPreset('create-dng');
        });

        // Value clamping — mirrors Flet's _clamp_float / _clamp_workers
        this.addClamp('create-dng-jxl-distance', 0.0, 25.0, 1.0);
        this.addClamp('create-dng-jxl-effort',   1,   9,   5);
        this.addClamp('create-dng-scale',         0.125, 1.0, 1.0);
        this.addClamp('create-dng-num-workers',   1,   8,   4);
    }

    setupRenderDNGHandlers() {
        // Conversion mode (tif / jpg / video) switching
        const modeSel = document.getElementById('render-dng-conversion-mode');
        modeSel.addEventListener('change', () => {
            const mode = modeSel.value;
            const isVideo = mode === 'video';
            document.getElementById('render-dng-video-section')
                .classList.toggle('section-hidden', !isVideo);
            document.getElementById('render-dng-bit-depth-field').style.display = isVideo ? 'none' : '';
            document.getElementById('render-dng-scale-field').style.display = isVideo ? 'none' : '';
            document.getElementById('render-dng-output-btn').textContent =
                isVideo ? 'Select Output File' : 'Select Output Folder';
            // Reset output selection since the target type changed
            this.outputPaths['render-dng'] = '';
            this.updateOutputPath('render-dng', '');
            const bitDepth = document.getElementById('render-dng-bit-depth');
            if (mode === 'jpg') {
                bitDepth.value = '8';
                bitDepth.disabled = true;
            } else {
                bitDepth.disabled = false;
            }
        });

        // Use XMP gating of manual rendering parameters
        const useXmp = document.getElementById('render-dng-use-xmp');
        useXmp.addEventListener('change', () => {
            const on = useXmp.checked;
            document.getElementById('render-dng-wb').disabled = on;
            document.getElementById('render-dng-exposure').disabled = on;
            if (on) {
                document.getElementById('render-dng-temperature').disabled = true;
                document.getElementById('render-dng-tint').disabled = true;
            } else {
                this.applyWBPreset('render-dng');
            }
        });

        document.getElementById('render-dng-wb').addEventListener('change', () => {
            this.applyWBPreset('render-dng');
        });

        this.addClamp('render-dng-scale',       0.125, 1.0, 1.0);
        this.addClamp('render-dng-num-workers', 1,     8,   4);
    }

    applyWBPreset(tab) {
        const wb = document.getElementById(`${tab}-wb`);
        const temp = document.getElementById(`${tab}-temperature`);
        const tint = document.getElementById(`${tab}-tint`);
        const preset = wb.value;
        if (preset === 'custom') {
            temp.disabled = false;
            tint.disabled = false;
        } else {
            const [t, ti] = WB_PRESETS[preset] || [null, null];
            temp.value = t !== null ? String(t) : '';
            tint.value = ti !== null ? String(ti) : '';
            temp.disabled = true;
            tint.disabled = true;
        }
    }

    addClamp(id, lo, hi, defaultVal) {
        const el = document.getElementById(id);
        if (!el) return;
        const clamp = () => {
            const v = parseFloat(el.value);
            if (isNaN(v)) {
                el.value = defaultVal;
            } else {
                el.value = Math.min(hi, Math.max(lo, v));
            }
        };
        el.addEventListener('blur', clamp);
        el.addEventListener('keydown', e => { if (e.key === 'Enter') clamp(); });
    }

    showAlert(title, message) {
        return new Promise(resolve => {
            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay';
            overlay.innerHTML =
                '<div class="modal">' +
                `<p class="modal-title">${title}</p>` +
                `<p class="modal-message">${message}</p>` +
                '<div class="modal-buttons">' +
                '<button class="modal-ok">OK</button>' +
                '</div></div>';
            document.body.appendChild(overlay);
            overlay.querySelector('.modal-ok').addEventListener('click', () => {
                overlay.remove();
                resolve();
            });
        });
    }

    async handleApplyMetadata() {
        const op = document.getElementById('create-dng-metadata-op').value;
        const name = document.getElementById('create-dng-tag-name').value.trim();
        const value = document.getElementById('create-dng-tag-value').value;
        
        if (op === 'set' || op === 'strip') {
            if (!name) {
                await this.showAlert('Missing tag name', 'Please enter a tag name.');
                return;
            }
            // Validate against the TIFF tag registry on the Python side
            if (window.pywebview && window.pywebview.api) {
                const known = await window.pywebview.api.validate_tag(name);
                if (!known) {
                    await this.showAlert(
                        'Unknown tag',
                        `Tag '${name}' is not in the TIFF tag registry and was not added.`
                    );
                    return;
                }
            }
        } else if (!value.trim()) {
            const hint = op === 'shift-time'
                ? 'Enter a time offset in the Value field (e.g. "+1:30" or "-2 04:00:00").'
                : 'Enter a timezone in the Value field (e.g. "+05:00").';
            await this.showAlert('Missing value', hint);
            return;
        }
        
        const opsList = document.getElementById('create-dng-metadata-ops');
        const opText = `${op.toUpperCase()}: ${name}${value ? ' = ' + value : ''}`;
        
        // Record op data for run_conversion
        if (!this.metadataOps['create-dng']) this.metadataOps['create-dng'] = [];
        const opData = { type: op, name: name, value: value };
        this.metadataOps['create-dng'].push(opData);
        
        // Add to operations list with a remove button
        const opItem = document.createElement('div');
        opItem.className = 'metadata-op-item';
        const text = document.createElement('span');
        text.textContent = opText;
        const removeBtn = document.createElement('button');
        removeBtn.className = 'metadata-op-remove';
        removeBtn.title = 'Remove';
        removeBtn.textContent = '✕';
        removeBtn.addEventListener('click', () => {
            const ops = this.metadataOps['create-dng'] || [];
            const idx = ops.indexOf(opData);
            if (idx !== -1) ops.splice(idx, 1);
            opItem.remove();
        });
        opItem.append(text, removeBtn);
        opsList.appendChild(opItem);
        
        // Clear inputs
        document.getElementById('create-dng-tag-name').value = '';
        document.getElementById('create-dng-tag-value').value = '';
        
        console.log('Applied metadata operation:', opText);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dngConverter = new DNGConverter();
});

// Expose methods for Python backend to call
window.updateProgress = (tab, message) => {
    if (window.dngConverter) {
        window.dngConverter.updateProgress(tab, message);
    }
};

window.updateProgressBar = (tab, value) => {
    if (window.dngConverter) {
        window.dngConverter.updateProgressBar(tab, value);
    }
};

window.appendLog = (tab, message) => {
    if (window.dngConverter) {
        window.dngConverter.appendLog(tab, message);
    }
};
