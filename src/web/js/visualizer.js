/**
 * Audio Stem Visualizer
 * 3D visualization of audio stems using Three.js and Web Audio API
 */

// Global variables
let scene, camera, renderer;
let audioContext, bassAnalyzer, drumsAnalyzer, vocalsAnalyzer, otherAnalyzer;
let bassFrequencyData, drumsFrequencyData, vocalsFrequencyData, otherFrequencyData;
let bassSource, drumsSource, vocalsSource, otherSource;
let isPlaying = false;
let playButton;

// Backward compatibility variables
let analyserBass, analyserDrums, analyserVocals, analyserOther;
let bassData, drumsData, vocalsData, otherData;
let stemBuffers = {};

// Three.js objects
let vocalsVisual, drumsVisual, bassVisual, otherVisual;
let clock;

// Buffer objects
let bassBuffer, drumsBuffer, vocalsBuffer, otherBuffer;

// Audio config for long song handling
let masterGain;
let seekPosition = 0; // Current position in seconds
let songDuration = 0;  // Total duration in seconds
let appConfig = window.appConfig || {
    audioConfig: {
        chunkMode: false,
        sampleRate: 22050,
        duration: 0
    }
};

// Main initialization function
function init() {
    console.log("Main initialization started");
    
    // Initialize app configuration if available
    if (window.appConfig) {
        appConfig = window.appConfig;
        console.log("App configuration loaded:", appConfig);
        
        if (appConfig.audioConfig && appConfig.audioConfig.duration) {
            songDuration = appConfig.audioConfig.duration;
            console.log("Song duration from config:", songDuration, "seconds");
        }
    }
    
    // Create loading indicator if it doesn't exist
    let loading = document.getElementById('loading-indicator');
    if (!loading) {
        loading = document.createElement('div');
        loading.id = 'loading-indicator';
        loading.textContent = 'Loading audio stems...';
        loading.style.display = 'none'; // Hide initially since loadStems will show it
        
        // Find the container
        const container = document.getElementById('visualization-container');
        if (container) {
            container.appendChild(loading);
        } else {
            document.body.appendChild(loading);
            console.warn("Visualization container not found, added loading indicator to body");
        }
    }
    
    try {
        // Display debug info
        displayStemPathsDebug();
        
        // Set up Three.js scene
        initThreeJs();
        
        // Set up event listeners - this needs to come before loadStems
        // for cases where stems might be already cached
        setupEventListeners();
        
        // Load stems - do this on a timeout to let the scene initialize first
        // This helps with memory usage by separating the intensive tasks
        setTimeout(() => {
            loadStems();
        }, 500);
    } catch (error) {
        console.error("Error during initialization:", error);
        alert("Error initializing visualizer: " + error.message);
    }
}

// Initialize Three.js scene
function initThreeJs() {
    const container = document.getElementById('canvas-container');
    
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x051012);
    scene.fog = new THREE.FogExp2(0x05101f, 0.025);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 10, 30);
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x1a2a3a, 0.5);
    scene.add(ambientLight);
    
    // Add directional light (sun-like)
    const sunLight = new THREE.DirectionalLight(0xf5c1b1, 1);
    sunLight.position.set(30, 50, 30);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 1024;
    sunLight.shadow.mapSize.height = 1024;
    scene.add(sunLight);
    
    // Initialize clock for animations
    clock = new THREE.Clock();
    
    // Create a simple test visualization (just a cube for each stem)
    console.log("Creating simple test visualizations");
    createSimpleTestVisuals();
    
    // Start animation loop
    animate();
}

// Create simple test visualizations to isolate the loading problem
function createSimpleTestVisuals() {
    // Create a group for all objects
    const testGroup = new THREE.Group();
    
    // Create simple cubes for each stem
    const geometry = new THREE.BoxGeometry(5, 5, 5);
    
    // Vocals - Red cube
    const vocalsMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    vocalsVisual = new THREE.Mesh(geometry, vocalsMaterial);
    vocalsVisual.position.set(-10, 0, 0);
    testGroup.add(vocalsVisual);
    
    // Drums - Blue cube
    const drumsMaterial = new THREE.MeshStandardMaterial({ color: 0x0000ff });
    drumsVisual = new THREE.Mesh(geometry, drumsMaterial);
    drumsVisual.position.set(10, 0, 0);
    testGroup.add(drumsVisual);
    
    // Bass - Green cube
    const bassMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
    bassVisual = new THREE.Mesh(geometry, bassMaterial);
    bassVisual.position.set(0, -10, 0);
    testGroup.add(bassVisual);
    
    // Other - Yellow cube
    const otherMaterial = new THREE.MeshStandardMaterial({ color: 0xffff00 });
    otherVisual = new THREE.Mesh(geometry, otherMaterial);
    otherVisual.position.set(0, 10, 0);
    testGroup.add(otherVisual);
    
    scene.add(testGroup);
    console.log("Simple test visualizations created");
}

// Initialize Web Audio API
function initAudio() {
    // Create audio context
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Create analyzers for each stem
    analyserVocals = audioContext.createAnalyser();
    analyserVocals.fftSize = 1024;
    vocalsData = new Uint8Array(analyserVocals.frequencyBinCount);
    
    analyserDrums = audioContext.createAnalyser();
    analyserDrums.fftSize = 1024;
    drumsData = new Uint8Array(analyserDrums.frequencyBinCount);
    
    analyserBass = audioContext.createAnalyser();
    analyserBass.fftSize = 1024;
    bassData = new Uint8Array(analyserBass.frequencyBinCount);
    
    analyserOther = audioContext.createAnalyser();
    analyserOther.fftSize = 1024;
    otherData = new Uint8Array(analyserOther.frequencyBinCount);
    
    // Load stem audio files
    loadStems();
}

// Load audio stems with improved memory management for long songs
function loadStems() {
    console.log("Starting loadStems() function - optimized for high-quality long songs");
    
    // Show loading message
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'flex';
        loadingIndicator.textContent = 'Loading high-quality audio stems...';
    } else {
        console.error("No loading indicator element found");
    }
    
    try {
        // Check available memory
        if (performance && performance.memory) {
            const memoryInfo = performance.memory;
            console.log("Memory info:", {
                jsHeapSizeLimit: Math.round(memoryInfo.jsHeapSizeLimit / (1024 * 1024)) + " MB",
                totalJSHeapSize: Math.round(memoryInfo.totalJSHeapSize / (1024 * 1024)) + " MB",
                usedJSHeapSize: Math.round(memoryInfo.usedJSHeapSize / (1024 * 1024)) + " MB"
            });
        } else {
            console.log("Performance.memory API not available");
        }
        
        // First try to get stems from the data attribute
        const stemPathsElement = document.getElementById('stem-paths');
        if (stemPathsElement) {
            console.log("Found stem-paths element, parsing data-stems attribute");
            try {
                const dataStemPaths = JSON.parse(stemPathsElement.getAttribute('data-stems'));
                if (dataStemPaths && Object.keys(dataStemPaths).length > 0) {
                    console.log("Using stems from data-stems attribute");
                    for (const stem in dataStemPaths) {
                        console.log(`Found ${stem} stem with URL length: ${dataStemPaths[stem].length}`);
                    }
                    stemPaths = dataStemPaths;
                }
            } catch (e) {
                console.error("Error parsing data-stems attribute:", e);
            }
        }
        
        // Fallback to global stemPaths
        if (!stemPaths || Object.keys(stemPaths).length === 0) {
            console.log("Using global stemPaths variable:", stemPaths);
        }
        
        // Check if we have valid stem paths
        if (!stemPaths || Object.keys(stemPaths).length === 0) {
            if (loadingIndicator) {
                loadingIndicator.textContent = 'Error: No audio stems found. Please check the console for details.';
            }
            console.error("No valid stem paths found");
            return;
        }
        
        // Create audio context
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Create master gain
        masterGain = audioContext.createGain();
        masterGain.gain.value = 0.5;
        masterGain.connect(audioContext.destination);
        
        // Create analyzers (simplified)
        const createAnalyzer = () => {
            const analyzer = audioContext.createAnalyser();
            analyzer.fftSize = 1024; // Reduced from 2048 to save memory
            return {
                analyzer: analyzer,
                data: new Uint8Array(analyzer.frequencyBinCount)
            };
        };
        
        // Create all analyzers
        const analyzers = {
            bass: createAnalyzer(),
            drums: createAnalyzer(),
            vocals: createAnalyzer(),
            other: createAnalyzer()
        };
        
        // Assign to global variables
        bassAnalyzer = analyzers.bass.analyzer;
        bassFrequencyData = analyzers.bass.data;
        
        drumsAnalyzer = analyzers.drums.analyzer;
        drumsFrequencyData = analyzers.drums.data;
        
        vocalsAnalyzer = analyzers.vocals.analyzer;
        vocalsFrequencyData = analyzers.vocals.data;
        
        otherAnalyzer = analyzers.other.analyzer;
        otherFrequencyData = analyzers.other.data;
        
        // Also assign to the older variable names for backward compatibility
        analyserBass = bassAnalyzer;
        bassData = bassFrequencyData;
        
        analyserDrums = drumsAnalyzer;
        drumsData = drumsFrequencyData;
        
        analyserVocals = vocalsAnalyzer;
        vocalsData = vocalsFrequencyData;
        
        analyserOther = otherAnalyzer;
        otherData = otherFrequencyData;
        
        if (loadingIndicator) {
            loadingIndicator.textContent = 'Audio context ready, loading stems...';
        }
        
        // Track completion
        let stemCount = Object.keys(stemPaths).length;
        let loadedCount = 0;
        
        // Load each stem sequentially to avoid memory pressure
        const loadStemsSequentially = async () => {
            const stems = Object.keys(stemPaths);
            
            for (let i = 0; i < stems.length; i++) {
                const stem = stems[i];
                const path = stemPaths[stem];
                
                if (loadingIndicator) {
                    loadingIndicator.textContent = `Loading ${stem} stem (${i+1}/${stems.length})...`;
                }
                
                console.log(`Loading ${stem} stem (${i+1}/${stems.length}), URL length: ${path.length}`);
                
                try {
                    // Fetch and decode in one go
                    const response = await fetch(path);
                    if (!response.ok) {
                        throw new Error(`Network response was not ok: ${response.status}`);
                    }
                    
                    // Get array buffer
                    const arrayBuffer = await response.arrayBuffer();
                    console.log(`${stem} data received, size:`, arrayBuffer.byteLength);
                    
                    // Force garbage collection if possible
                    if (window.gc) window.gc();
                    
                    // Decode the audio
                    const buffer = await audioContext.decodeAudioData(arrayBuffer);
                    console.log(`${stem} decode successful, duration:`, buffer.duration, "sampleRate:", buffer.sampleRate);
                    
                    // Store buffer based on stem type
                    switch (stem) {
                        case 'bass': bassBuffer = buffer; break;
                        case 'drums': drumsBuffer = buffer; break;
                        case 'vocals': vocalsBuffer = buffer; break;
                        case 'other': otherBuffer = buffer; break;
                    }
                    
                    // For backward compatibility
                    stemBuffers[stem] = buffer;
                    
                    // Update song duration if not set and this is the first stem
                    if (!songDuration && buffer.duration) {
                        songDuration = buffer.duration;
                        console.log("Set song duration to", songDuration, "seconds");
                    }
                    
                    // Force another garbage collection
                    if (window.gc) window.gc();
                    
                    // Increment loaded count
                    loadedCount++;
                    console.log(`${stem} stem loaded (${loadedCount}/${stemCount})`);
                    
                } catch (error) {
                    console.error(`Error loading ${stem} stem:`, error);
                    if (loadingIndicator) {
                        loadingIndicator.textContent = `Error loading ${stem} stem: ${error.message}`;
                    }
                    loadedCount++;
                }
                
                // Add a small delay between loading stems to avoid memory pressure
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
            // All stems have been loaded (or attempted)
            console.log("All stems processed");
            
            if (loadingIndicator) {
                if (bassBuffer || drumsBuffer || vocalsBuffer || otherBuffer) {
                    loadingIndicator.style.display = 'none';
                } else {
                    loadingIndicator.textContent = 'Failed to load any stems. Check console for errors.';
                }
            }
            
            // Create seek bar if we're dealing with a long song
            if (songDuration > 60) {
                createSeekBar();
            }
        };
        
        // Start the sequential loading process
        loadStemsSequentially();
        
    } catch (error) {
        console.error("Fatal error in loadStems function:", error);
        if (loadingIndicator) {
            loadingIndicator.textContent = `Fatal error: ${error.message}`;
        }
    }
}

// Set up event listeners
function setupEventListeners() {
    console.log("Setting up event listeners");
    
    // Find the play button by various possible IDs
    const playButtonIds = ['play-pause', 'play-button'];
    let foundButton = false;
    
    for (const id of playButtonIds) {
        const button = document.getElementById(id);
        if (button) {
            playButton = button;
            
            // Remove existing listeners to avoid duplicates
            const newButton = button.cloneNode(true);
            button.parentNode.replaceChild(newButton, button);
            playButton = newButton;
            
            playButton.addEventListener('click', togglePlayback);
            console.log(`Set up click listener for play button with ID: ${id}`);
            foundButton = true;
            
            // Ensure button is enabled
            playButton.disabled = false;
            break;
        }
    }
    
    if (!foundButton) {
        console.error("Could not find play button with IDs:", playButtonIds);
        
        // Try to find any button in the visualization container
        const container = document.getElementById('visualization-container');
        if (container) {
            const buttons = container.querySelectorAll('button');
            if (buttons.length > 0) {
                console.log("Found alternate button:", buttons[0]);
                playButton = buttons[0];
                
                // Remove existing listeners
                const newButton = playButton.cloneNode(true);
                playButton.parentNode.replaceChild(newButton, playButton);
                playButton = newButton;
                
                playButton.addEventListener('click', togglePlayback);
                playButton.disabled = false;
            }
        }
    }
    
    // Set up stem toggles if they exist
    document.querySelectorAll('.stem-toggle input').forEach(checkbox => {
        checkbox.checked = true;
        
        // Remove existing listeners
        const newCheckbox = checkbox.cloneNode(true);
        checkbox.parentNode.replaceChild(newCheckbox, checkbox);
        
        newCheckbox.addEventListener('change', updateStemVisibility);
    });
    
    // Listen for window resize
    window.addEventListener('resize', onWindowResize);
}

// Function to update stem visibility based on checkboxes
function updateStemVisibility() {
    document.querySelectorAll('.stem-toggle input').forEach(checkbox => {
        const stemName = checkbox.name;
        const isChecked = checkbox.checked;
        
        // Update visibility of corresponding visual element
        switch(stemName) {
            case 'vocals':
                if (vocalsVisual) vocalsVisual.visible = isChecked;
                break;
            case 'drums':
                if (drumsVisual) drumsVisual.visible = isChecked;
                break;
            case 'bass':
                if (bassVisual) bassVisual.visible = isChecked;
                break;
            case 'other':
                if (otherVisual) otherVisual.visible = isChecked;
                break;
        }
    });
}

// Play or pause all stems
function togglePlayback() {
    try {
        console.log("Toggle playback called, current state:", isPlaying ? "playing" : "stopped");
        
        // Debug what button was pressed
        console.log("Button pressed:", this);
        
        if (isPlaying) {
            stopAllStems();
            if (playButton) {
                playButton.textContent = 'Play';
                playButton.classList.remove('playing');
            }
        } else {
            // First check if we have any buffers loaded
            const hasBuffers = bassBuffer || drumsBuffer || vocalsBuffer || otherBuffer;
            
            if (!hasBuffers) {
                console.error("No audio buffers loaded, cannot play");
                alert("No audio data loaded. Please make sure the stems have been processed correctly.");
                return;
            }
            
            // Check audio context
            if (!audioContext) {
                console.error("No audio context created");
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            if (audioContext.state === 'closed') {
                console.error("Audio context is closed, creating a new one");
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                // Recreate analyzers
                if (!bassAnalyzer) {
                    bassAnalyzer = audioContext.createAnalyser();
                    bassAnalyzer.fftSize = 1024;
                    bassFrequencyData = new Uint8Array(bassAnalyzer.frequencyBinCount);
                }
                
                if (!drumsAnalyzer) {
                    drumsAnalyzer = audioContext.createAnalyser();
                    drumsAnalyzer.fftSize = 1024;
                    drumsFrequencyData = new Uint8Array(drumsAnalyzer.frequencyBinCount);
                }
                
                if (!vocalsAnalyzer) {
                    vocalsAnalyzer = audioContext.createAnalyser();
                    vocalsAnalyzer.fftSize = 1024;
                    vocalsFrequencyData = new Uint8Array(vocalsAnalyzer.frequencyBinCount);
                }
                
                if (!otherAnalyzer) {
                    otherAnalyzer = audioContext.createAnalyser();
                    otherAnalyzer.fftSize = 1024;
                    otherFrequencyData = new Uint8Array(otherAnalyzer.frequencyBinCount);
                }
                
                // Create master gain
                masterGain = audioContext.createGain();
                masterGain.gain.value = 0.5;
                masterGain.connect(audioContext.destination);
            }
            
            // Try to play from current seek position
            try {
                playAllStems(seekPosition);
                if (playButton) {
                    playButton.textContent = 'Pause';
                    playButton.classList.add('playing');
                }
            } catch (e) {
                console.error("Error in playAllStems:", e);
                alert("Error playing audio: " + e.message);
                return;
            }
        }
        
        isPlaying = !isPlaying;
    } catch (error) {
        console.error("Fatal error in togglePlayback:", error);
        alert("Error toggling playback: " + error.message);
    }
}

// Play all stems with seek position support
function playAllStems(startPosition = 0) {
    try {
        console.log("Starting playAllStems function at position", startPosition);
        
        // Resume audio context if suspended
        if (audioContext.state === 'suspended') {
            console.log("Audio context was suspended, resuming...");
            audioContext.resume();
        }
        
        // Check for quality settings
        const qualityToggle = document.getElementById('quality-toggle');
        const highQuality = qualityToggle ? qualityToggle.checked : true;
        
        // Log status of audio buffers
        console.log("Playing stems with buffers:", 
            { bass: bassBuffer ? `loaded (${bassBuffer.duration}s)` : "missing", 
              drums: drumsBuffer ? `loaded (${drumsBuffer.duration}s)` : "missing",
              vocals: vocalsBuffer ? `loaded (${vocalsBuffer.duration}s)` : "missing",
              other: otherBuffer ? `loaded (${otherBuffer.duration}s)` : "missing" },
            "Quality:", highQuality ? "high" : "low");
        
        // Fix incorrect analyzer references
        let bassAnalyzerLocal = bassAnalyzer || analyserBass;
        let drumsAnalyzerLocal = drumsAnalyzer || analyserDrums;
        let vocalsAnalyzerLocal = vocalsAnalyzer || analyserVocals;
        let otherAnalyzerLocal = otherAnalyzer || analyserOther;
        
        // Set FFT size based on quality
        const fftSize = highQuality ? 2048 : 512;
        if (bassAnalyzerLocal) bassAnalyzerLocal.fftSize = fftSize;
        if (drumsAnalyzerLocal) drumsAnalyzerLocal.fftSize = fftSize;
        if (vocalsAnalyzerLocal) vocalsAnalyzerLocal.fftSize = fftSize;
        if (otherAnalyzerLocal) otherAnalyzerLocal.fftSize = fftSize;
        
        // Create master gain if it doesn't exist
        if (!masterGain) {
            console.log("Creating master gain node");
            masterGain = audioContext.createGain();
            masterGain.gain.value = 0.5;
            masterGain.connect(audioContext.destination);
        }
        
        // Store the current time for tracking playback position
        const currentTime = audioContext.currentTime;
        
        // Helper function to create and start a source
        const createAndStartSource = (buffer, analyzer, sourceName) => {
            if (!buffer) return null;
            
            try {
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                
                // Connect through analyzer if available
                if (analyzer) {
                    source.connect(analyzer);
                    analyzer.connect(masterGain);
                } else {
                    source.connect(masterGain);
                }
                
                // Start from the specified position
                source.start(0, startPosition);
                source.startTime = currentTime - startPosition; // For tracking
                console.log(`${sourceName} playback started at position ${startPosition}`);
                return source;
            } catch (e) {
                console.error(`Error starting ${sourceName} playback:`, e);
                return null;
            }
        };
        
        // Create sources for each stem using the helper function
        bassSource = createAndStartSource(bassBuffer, bassAnalyzerLocal, "Bass");
        drumsSource = createAndStartSource(drumsBuffer, drumsAnalyzerLocal, "Drums");
        vocalsSource = createAndStartSource(vocalsBuffer, vocalsAnalyzerLocal, "Vocals");
        otherSource = createAndStartSource(otherBuffer, otherAnalyzerLocal, "Other");
        
        // Set end event to update UI when playback finishes
        let setEndEvent = false;
        
        // Try to set end event on any available source, prioritizing vocals
        if (vocalsSource) {
            vocalsSource.onended = function() {
                console.log("Vocals playback ended");
                if (isPlaying) {
                    togglePlayback();
                }
            };
            setEndEvent = true;
        } else if (drumsSource) {
            drumsSource.onended = function() {
                console.log("Drums playback ended");
                if (isPlaying) {
                    togglePlayback();
                }
            };
            setEndEvent = true;
        } else if (bassSource) {
            bassSource.onended = function() {
                console.log("Bass playback ended");
                if (isPlaying) {
                    togglePlayback();
                }
            };
            setEndEvent = true;
        } else if (otherSource) {
            otherSource.onended = function() {
                console.log("Other playback ended");
                if (isPlaying) {
                    togglePlayback();
                }
            };
            setEndEvent = true;
        }
        
        if (!setEndEvent) {
            console.warn("No sources available to set end event");
        }
        
        // Schedule garbage collection if available
        setTimeout(() => {
            if (window.gc) window.gc();
        }, 1000);
        
        console.log("All playback started successfully");
    } catch (error) {
        console.error("Fatal error in playAllStems:", error);
        alert("Error playing audio: " + error.message);
    }
}

// Stop all stems
function stopAllStems() {
    console.log("Stopping all stems");
    
    // Helper to safely stop and disconnect a source
    const stopSource = (source) => {
        if (source) {
            try {
                source.stop();
                source.disconnect();
            } catch (e) {
                console.error("Error stopping source:", e);
            }
        }
    };
    
    // Stop each source
    stopSource(vocalsSource);
    stopSource(drumsSource);
    stopSource(bassSource);
    stopSource(otherSource);
    
    // Reset sources
    vocalsSource = null;
    drumsSource = null;
    bassSource = null;
    otherSource = null;
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Get time delta
    const delta = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();
    
    // Update visualizations if playing audio
    if (isPlaying) {
        // Update frequency data from analyzers
        if (bassAnalyzer && bassFrequencyData) {
            bassAnalyzer.getByteFrequencyData(bassFrequencyData);
        }
        
        if (drumsAnalyzer && drumsFrequencyData) {
            drumsAnalyzer.getByteFrequencyData(drumsFrequencyData);
        }
        
        if (vocalsAnalyzer && vocalsFrequencyData) {
            vocalsAnalyzer.getByteFrequencyData(vocalsFrequencyData);
        }
        
        if (otherAnalyzer && otherFrequencyData) {
            otherAnalyzer.getByteFrequencyData(otherFrequencyData);
        }
        
        // Simple test animation for cubes
        if (bassVisual) {
            bassVisual.rotation.y += 0.01;
            // Scale with bass frequencies if analyzer is available
            if (bassAnalyzer && bassFrequencyData) {
                const bassEnergy = getAverageFrequency(bassFrequencyData, 0, 10);
                bassVisual.scale.set(1 + bassEnergy/100, 1 + bassEnergy/100, 1 + bassEnergy/100);
            }
        }
        
        if (drumsVisual) {
            drumsVisual.rotation.x += 0.01;
            // Scale with drum frequencies if analyzer is available
            if (drumsAnalyzer && drumsFrequencyData) {
                const drumsEnergy = getAverageFrequency(drumsFrequencyData, 0, 30);
                drumsVisual.scale.set(1 + drumsEnergy/100, 1 + drumsEnergy/100, 1 + drumsEnergy/100);
            }
        }
        
        if (vocalsVisual) {
            vocalsVisual.rotation.z += 0.01;
            // Scale with vocals frequencies if analyzer is available
            if (vocalsAnalyzer && vocalsFrequencyData) {
                const vocalsEnergy = getAverageFrequency(vocalsFrequencyData, 10, 40);
                vocalsVisual.scale.set(1 + vocalsEnergy/100, 1 + vocalsEnergy/100, 1 + vocalsEnergy/100);
            }
        }
        
        if (otherVisual) {
            otherVisual.rotation.y += 0.005;
            otherVisual.rotation.x += 0.005;
            // Scale with other frequencies if analyzer is available
            if (otherAnalyzer && otherFrequencyData) {
                const otherEnergy = getAverageFrequency(otherFrequencyData, 10, 40);
                otherVisual.scale.set(1 + otherEnergy/100, 1 + otherEnergy/100, 1 + otherEnergy/100);
            }
        }
    } else {
        // Simple idle animation for test cubes
        if (bassVisual) {
            bassVisual.rotation.y += 0.005;
        }
        
        if (drumsVisual) {
            drumsVisual.rotation.x += 0.005;
        }
        
        if (vocalsVisual) {
            vocalsVisual.rotation.z += 0.005;
        }
        
        if (otherVisual) {
            otherVisual.rotation.y += 0.003;
            otherVisual.rotation.x += 0.002;
        }
    }
    
    // Simple camera orbit
    camera.position.x = Math.sin(elapsedTime * 0.1) * 30;
    camera.position.z = Math.cos(elapsedTime * 0.1) * 30;
    camera.lookAt(0, 0, 0);
    
    // Render the scene
    renderer.render(scene, camera);
}

// Helper function to get average frequency from frequency data
function getAverageFrequency(frequencyData, startIndex, endIndex) {
    let sum = 0;
    for (let i = startIndex; i < endIndex; i++) {
        sum += frequencyData[i];
    }
    return sum / (endIndex - startIndex);
}

// Handle window resize
function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Initialize when the page loads
window.addEventListener('load', init);

// Initialize when page is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("Document loaded - starting initial setup");
});

// Function to display stem paths for debugging
function displayStemPathsDebug() {
    console.log("Displaying stem paths debug information");
    
    // Create debug display element
    const debugDisplay = document.createElement('div');
    debugDisplay.id = 'debug-display';
    debugDisplay.style.cssText = 'background: rgba(0,0,0,0.8); color: white; padding: 15px; position: absolute; top: 10px; right: 10px; max-width: 400px; max-height: 300px; overflow: auto; z-index: 1000; font-family: monospace; font-size: 12px; border-radius: 5px;';
    
    // Add title
    const title = document.createElement('h3');
    title.textContent = 'Debug: Stem Data';
    title.style.cssText = 'margin-top: 0; color: #ff9';
    debugDisplay.appendChild(title);
    
    // Get stem paths
    const stemPathsElement = document.getElementById('stem-paths');
    let stemPaths = {};
    
    try {
        if (stemPathsElement) {
            stemPaths = JSON.parse(stemPathsElement.getAttribute('data-stems'));
            const pathList = document.createElement('ul');
            
            if (Object.keys(stemPaths).length === 0) {
                const noStems = document.createElement('p');
                noStems.textContent = 'No stem paths found in data-stems attribute!';
                noStems.style.color = '#f55';
                debugDisplay.appendChild(noStems);
            } else {
                // Display all stems found
                for (const [stem, path] of Object.entries(stemPaths)) {
                    const item = document.createElement('li');
                    item.textContent = `${stem}: Data URL found (${Math.round(path.length / 1024)} KB)`;
                    pathList.appendChild(item);
                }
                debugDisplay.appendChild(pathList);
            }
        } else {
            const noElement = document.createElement('p');
            noElement.textContent = 'No stem-paths element found in the document!';
            noElement.style.color = '#f55';
            debugDisplay.appendChild(noElement);
            
            // Debug global stemPaths variable
            const globalPaths = document.createElement('p');
            globalPaths.textContent = `Global stemPaths variable: ${JSON.stringify(stemPaths).substring(0, 100)}...`;
            debugDisplay.appendChild(globalPaths);
        }
    } catch (e) {
        const error = document.createElement('p');
        error.textContent = `Error parsing stem paths: ${e.message}`;
        error.style.color = '#f55';
        debugDisplay.appendChild(error);
    }
    
    // Show direct variables
    const directVars = document.createElement('div');
    directVars.innerHTML = `<p><strong>Direct stemPaths variable:</strong></p>
                           <p>Type: ${typeof stemPaths}</p>
                           <p>Keys: ${Object.keys(stemPaths).join(', ') || 'none'}</p>`;
    debugDisplay.appendChild(directVars);
    
    document.body.appendChild(debugDisplay);
}

// Create a seek bar for long songs
function createSeekBar() {
    console.log("Creating seek bar for long song");
    
    // Find a good place to insert the seek bar
    const container = document.getElementById('controls');
    if (!container) {
        console.error("Could not find controls container for seek bar");
        return;
    }
    
    // Create seek bar container
    const seekContainer = document.createElement('div');
    seekContainer.className = 'seek-container';
    seekContainer.style.cssText = 'width: 100%; padding: 10px 0; display: flex; align-items: center; flex-wrap: wrap;';
    
    // Create seek bar
    const seekBar = document.createElement('input');
    seekBar.type = 'range';
    seekBar.min = 0;
    seekBar.max = Math.ceil(songDuration);
    seekBar.value = 0;
    seekBar.id = 'seek-bar';
    seekBar.style.cssText = 'width: 80%; margin: 0 10px; accent-color: #9c27b0; height: 8px; cursor: pointer; background: #333;';
    
    // Create time display
    const timeDisplay = document.createElement('div');
    timeDisplay.id = 'time-display';
    timeDisplay.textContent = '0:00 / ' + formatTime(songDuration);
    timeDisplay.style.cssText = 'font-family: monospace; color: white; margin-left: 10px; min-width: 85px;';
    
    // Create quality toggle
    const qualityContainer = document.createElement('div');
    qualityContainer.style.cssText = 'margin-top: 10px; width: 100%; display: flex; justify-content: flex-end; align-items: center;';
    
    const qualityLabel = document.createElement('span');
    qualityLabel.textContent = 'High Quality: ';
    qualityLabel.style.cssText = 'color: white; font-size: 12px; margin-right: 5px;';
    
    const qualityToggle = document.createElement('input');
    qualityToggle.type = 'checkbox';
    qualityToggle.id = 'quality-toggle';
    qualityToggle.checked = true;
    
    qualityToggle.addEventListener('change', function() {
        // When quality is toggled, we can adjust the FFT size
        const fftSize = this.checked ? 2048 : 512;
        
        if (bassAnalyzer) bassAnalyzer.fftSize = fftSize;
        if (drumsAnalyzer) drumsAnalyzer.fftSize = fftSize;
        if (vocalsAnalyzer) vocalsAnalyzer.fftSize = fftSize;
        if (otherAnalyzer) otherAnalyzer.fftSize = fftSize;
        
        console.log(`Audio quality set to ${this.checked ? 'high' : 'low'}, FFT size: ${fftSize}`);
    });
    
    // Add event listener for seeking
    seekBar.addEventListener('input', function() {
        seekPosition = parseFloat(this.value);
        timeDisplay.textContent = formatTime(seekPosition) + ' / ' + formatTime(songDuration);
    });
    
    // Add event listener for seeking completion (mouseup)
    seekBar.addEventListener('change', function() {
        // If we're playing, stop and restart at new position
        if (isPlaying) {
            stopAllStems();
            playAllStems(seekPosition);
        }
    });
    
    // Add elements to container
    seekContainer.appendChild(seekBar);
    seekContainer.appendChild(timeDisplay);
    qualityContainer.appendChild(qualityLabel);
    qualityContainer.appendChild(qualityToggle);
    seekContainer.appendChild(qualityContainer);
    
    // Add to page
    container.appendChild(seekContainer);
    
    // Update the seek bar during playback
    setInterval(() => {
        if (isPlaying && vocalsSource) {
            seekPosition = audioContext.currentTime - vocalsSource.startTime;
            if (seekPosition > songDuration) seekPosition = songDuration;
            
            seekBar.value = seekPosition;
            timeDisplay.textContent = formatTime(seekPosition) + ' / ' + formatTime(songDuration);
        }
    }, 250); // Update more frequently for smoother UI
}

// Format time in seconds to MM:SS format
function formatTime(seconds) {
    seconds = Math.floor(seconds);
    const minutes = Math.floor(seconds / 60);
    seconds = seconds % 60;
    return minutes + ':' + (seconds < 10 ? '0' : '') + seconds;
} 