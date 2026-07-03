// mu DNG Converter - JavaScript UI Controller

class DNGConverter {
    constructor() {
        this.currentTab = 'dng-image';
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
        this.setupDNGDNGHandlers();
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
        // DNG → Image buttons
        document.getElementById('dng-image-run').addEventListener('click', () => {
            this.handleRun('dng-image');
        });
        
        document.getElementById('dng-image-cancel').addEventListener('click', () => {
            this.handleCancel('dng-image');
        });
        
        document.getElementById('dng-image-input-btn').addEventListener('click', () => {
            this.handleInputSelect('dng-image');
        });
        
        document.getElementById('dng-image-output-btn').addEventListener('click', () => {
            this.handleOutputSelect('dng-image');
        });

        // FITS → DNG buttons
        document.getElementById('fits-dng-run').addEventListener('click', () => {
            this.handleRun('fits-dng');
        });
        
        document.getElementById('fits-dng-cancel').addEventListener('click', () => {
            this.handleCancel('fits-dng');
        });
        
        document.getElementById('fits-dng-input-btn').addEventListener('click', () => {
            this.handleInputSelect('fits-dng');
        });
        
        document.getElementById('fits-dng-output-btn').addEventListener('click', () => {
            this.handleOutputSelect('fits-dng');
        });

        // DNG → DNG buttons
        document.getElementById('dng-dng-run').addEventListener('click', () => {
            this.handleRun('dng-dng');
        });
        
        document.getElementById('dng-dng-cancel').addEventListener('click', () => {
            this.handleCancel('dng-dng');
        });
        
        document.getElementById('dng-dng-input-btn').addEventListener('click', () => {
            this.handleInputSelect('dng-dng');
        });
        
        document.getElementById('dng-dng-output-btn').addEventListener('click', () => {
            this.handleOutputSelect('dng-dng');
        });
    }

    gatherSettings(tab) {
        const settings = {
            input: this.inputPaths[tab] || [],
            inputMode: this.inputModes[tab] || 'folder',
            output: this.outputPaths[tab] || '',
        };
        if (tab === 'dng-dng') {
            settings.transcode = document.getElementById('dng-dng-transcode').checked;
            settings.compression = document.getElementById('dng-dng-compression').value;
            settings.jxlDistance = document.getElementById('dng-dng-jxl-distance').value;
            settings.jxlEffort = document.getElementById('dng-dng-jxl-effort').value;
            settings.demosaic = document.getElementById('dng-dng-demosaic').checked;
            settings.scale = document.getElementById('dng-dng-scale').value;
            settings.preview = document.getElementById('dng-dng-preview').checked;
            settings.fastLoad = document.getElementById('dng-dng-fast-load').checked;
            settings.numWorkers = document.getElementById('dng-dng-num-workers').value;
            settings.metadataOps = this.metadataOps[tab] || [];
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
                    const ok = await this.showConfirm(
                        `${result.existing} of ${result.total} output files already exist. Overwrite?`
                    );
                    if (ok) {
                        settings.overwriteConfirmed = true;
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

    showConfirm(message) {
        return new Promise(resolve => {
            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay';
            overlay.innerHTML =
                '<div class="modal">' +
                `<p class="modal-message">${message}</p>` +
                '<div class="modal-buttons">' +
                '<button class="modal-cancel">Cancel</button>' +
                '<button class="modal-ok">OK</button>' +
                '</div></div>';
            document.body.appendChild(overlay);
            overlay.querySelector('.modal-ok').addEventListener('click', () => {
                overlay.remove();
                resolve(true);
            });
            overlay.querySelector('.modal-cancel').addEventListener('click', () => {
                overlay.remove();
                resolve(false);
            });
        });
    }

    handleCancel(tab) {
        console.log(`Cancel clicked for ${tab}`);
        // Signal Python backend; run_conversion will return and handleRun restores the UI
        if (window.pywebview && window.pywebview.api) {
            window.pywebview.api.cancel_conversion(tab);
        }
    }

    handleComplete(tab) {
        const runBtn = document.getElementById(`${tab}-run`);
        const cancelBtn = document.getElementById(`${tab}-cancel`);
        const progressText = document.getElementById(`${tab}-progress`);
        
        runBtn.style.display = 'flex';
        cancelBtn.style.display = 'none';
        progressText.textContent = 'Complete';
    }

    async handleInputSelect(tab) {
        console.log(`Input select clicked for ${tab}`);
        const modeEl = document.getElementById(`${tab}-mode`);
        const mode = modeEl ? modeEl.value : 'folder';
        
        try {
            if (window.pywebview && window.pywebview.api) {
                const result = await window.pywebview.api.select_input(tab, mode);
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
        
        try {
            if (window.pywebview && window.pywebview.api) {
                const result = await window.pywebview.api.select_output(tab);
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

    setupDNGDNGHandlers() {
        document.getElementById('dng-dng-apply-metadata').addEventListener('click', () => {
            this.handleApplyMetadata();
        });

        const transcodeCheckbox = document.getElementById('dng-dng-transcode');
        const transcodeControls = [
            'dng-dng-compression', 'dng-dng-jxl-distance', 'dng-dng-jxl-effort',
            'dng-dng-demosaic', 'dng-dng-demosaic-algo', 'dng-dng-scale',
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
            const compression = document.getElementById('dng-dng-compression').value;
            const isLossy = compression === 'jxl_lossy';
            const isJxl = compression !== 'uncompressed';
            const distField = document.getElementById('dng-dng-jxl-distance');
            const effortField = document.getElementById('dng-dng-jxl-effort');
            distField.disabled = !isLossy;
            if (!isLossy) distField.value = '0';
            effortField.disabled = !isJxl;
        };

        const updateDemosaicState = () => {
            const demosaicEnabled = document.getElementById('dng-dng-demosaic').checked;
            document.getElementById('dng-dng-demosaic-algo').disabled = !demosaicEnabled;
            document.getElementById('dng-dng-scale').disabled = !demosaicEnabled;
        };

        transcodeCheckbox.addEventListener('change', updateTranscodeState);
        document.getElementById('dng-dng-compression').addEventListener('change', updateCompressionState);
        document.getElementById('dng-dng-demosaic').addEventListener('change', updateDemosaicState);

        // Value clamping — mirrors Flet's _clamp_float / _clamp_workers
        this.addClamp('dng-dng-jxl-distance', 0.0, 25.0, 1.0);
        this.addClamp('dng-dng-jxl-effort',   1,   9,   5);
        this.addClamp('dng-dng-scale',         0.125, 1.0, 1.0);
        this.addClamp('dng-dng-num-workers',   1,   8,   4);
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

    handleApplyMetadata() {
        const op = document.getElementById('dng-dng-metadata-op').value;
        const name = document.getElementById('dng-dng-tag-name').value;
        const value = document.getElementById('dng-dng-tag-value').value;
        
        if (!name) {
            alert('Please enter a tag name');
            return;
        }
        
        const opsList = document.getElementById('dng-dng-metadata-ops');
        const opText = `${op.toUpperCase()}: ${name}${value ? ' = ' + value : ''}`;
        
        // Record op data for run_conversion
        if (!this.metadataOps['dng-dng']) this.metadataOps['dng-dng'] = [];
        this.metadataOps['dng-dng'].push({ type: op, name: name, value: value });
        
        // Add to operations list
        const opItem = document.createElement('div');
        opItem.textContent = opText;
        opItem.style.padding = '2px 0';
        opsList.appendChild(opItem);
        
        // Clear inputs
        document.getElementById('dng-dng-tag-name').value = '';
        document.getElementById('dng-dng-tag-value').value = '';
        
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
