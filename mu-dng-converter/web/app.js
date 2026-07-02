// mu DNG Converter - JavaScript UI Controller

class DNGConverter {
    constructor() {
        this.currentTab = 'dng-image';
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

    async handleRun(tab) {
        console.log(`Run clicked for ${tab}`);
        const runBtn = document.getElementById(`${tab}-run`);
        const cancelBtn = document.getElementById(`${tab}-cancel`);
        const progressText = document.getElementById(`${tab}-progress`);
        
        // Show cancel button and progress
        runBtn.style.display = 'none';
        cancelBtn.style.display = 'flex';
        progressText.textContent = 'Starting...';
        
        // Call Python backend
        try {
            if (window.pywebview && window.pywebview.api) {
                await window.pywebview.api.handleRun(tab);
            } else {
                // Fallback for testing without pywebview
                progressText.textContent = 'Running (simulation)...';
                setTimeout(() => {
                    this.handleComplete(tab);
                }, 2000);
            }
        } catch (error) {
            console.error('Error running:', error);
            progressText.textContent = 'Error: ' + error.message;
            this.handleComplete(tab);
        }
    }

    handleCancel(tab) {
        console.log(`Cancel clicked for ${tab}`);
        const runBtn = document.getElementById(`${tab}-run`);
        const cancelBtn = document.getElementById(`${tab}-cancel`);
        const progressText = document.getElementById(`${tab}-progress`);
        
        // Call Python backend
        if (window.pywebview && window.pywebview.api) {
            window.pywebview.api.handleCancel(tab);
        }
        
        // Reset UI
        runBtn.style.display = 'flex';
        cancelBtn.style.display = 'none';
        progressText.textContent = 'Cancelled';
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
        
        try {
            if (window.pywebview && window.pywebview.api) {
                const result = await window.pywebview.api.selectInput(tab);
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
                const result = await window.pywebview.api.selectOutput(tab);
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
            pathElement.textContent = path || 'No folder selected';
        }
    }

    updateOutputPath(tab, path) {
        const pathElement = document.getElementById(`${tab}-output-path`);
        if (pathElement) {
            pathElement.textContent = path || 'No folder selected';
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
        // DNG → DNG specific handlers
        document.getElementById('dng-dng-apply-metadata').addEventListener('click', () => {
            this.handleApplyMetadata();
        });
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
