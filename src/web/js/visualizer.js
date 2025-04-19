/**
 * Audio Stem Visualizer
 * 3D visualization of audio stems using Three.js and Web Audio API
 */

// Global variables
let scene, camera, renderer;
let audioContext, analyserVocals, analyserDrums, analyserBass, analyserOther;
let vocalsData, drumsData, bassData, otherData;
let vocalsSource, drumsSource, bassSource, otherSource;
let isPlaying = false;
let stemBuffers = {};
let playButton;

// Three.js objects
let vocalsVisual, drumsVisual, bassVisual, otherVisual;

// Initialization
function init() {
    // Create loading indicator
    const loading = document.createElement('div');
    loading.id = 'loading';
    loading.textContent = 'Loading audio stems...';
    document.getElementById('visualization-container').appendChild(loading);

    // Set up Three.js scene
    initThreeJs();
    
    // Set up audio context
    initAudio();

    // Button event listeners
    playButton = document.getElementById('play-pause');
    playButton.addEventListener('click', togglePlayback);
    
    // Mark stems as active by default
    document.querySelectorAll('.stem-toggle input').forEach(checkbox => {
        checkbox.checked = true;
    });
    
    // Listen for window resize
    window.addEventListener('resize', onWindowResize);
}

// Initialize Three.js scene
function initThreeJs() {
    const container = document.getElementById('canvas-container');
    
    // Create scene
    scene = new THREE.Scene();
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 30;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Initialize stem visualizations
    initVocalsVisual();
    initDrumsVisual();
    initBassVisual();
    initOtherVisual();
    
    // Start animation loop
    animate();
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

// Load stem audio files from the paths provided by Python
async function loadStems() {
    try {
        // Load each stem file
        const stems = Object.keys(stemPaths);
        const fetchPromises = stems.map(stem => fetch(stemPaths[stem])
            .then(response => response.arrayBuffer())
            .then(arrayBuffer => audioContext.decodeAudioData(arrayBuffer))
            .then(audioBuffer => {
                stemBuffers[stem] = audioBuffer;
                return audioBuffer;
            })
        );
        
        await Promise.all(fetchPromises);
        
        // Hide loading indicator
        document.getElementById('loading').style.display = 'none';
        
        console.log('All stems loaded successfully');
    } catch (error) {
        console.error('Error loading stems:', error);
        document.getElementById('loading').textContent = 'Error loading audio stems. Please try again.';
    }
}

// Play or pause all stems
function togglePlayback() {
    if (isPlaying) {
        stopAllStems();
        playButton.textContent = 'Play';
        playButton.classList.remove('playing');
    } else {
        playAllStems();
        playButton.textContent = 'Pause';
        playButton.classList.add('playing');
    }
    
    isPlaying = !isPlaying;
}

// Play all stems
function playAllStems() {
    // Resume audio context if suspended
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
    
    // Create sources for each stem
    if (stemBuffers.vocals) {
        vocalsSource = audioContext.createBufferSource();
        vocalsSource.buffer = stemBuffers.vocals;
        vocalsSource.connect(analyserVocals);
        analyserVocals.connect(audioContext.destination);
        vocalsSource.start(0);
    }
    
    if (stemBuffers.drums) {
        drumsSource = audioContext.createBufferSource();
        drumsSource.buffer = stemBuffers.drums;
        drumsSource.connect(analyserDrums);
        analyserDrums.connect(audioContext.destination);
        drumsSource.start(0);
    }
    
    if (stemBuffers.bass) {
        bassSource = audioContext.createBufferSource();
        bassSource.buffer = stemBuffers.bass;
        bassSource.connect(analyserBass);
        analyserBass.connect(audioContext.destination);
        bassSource.start(0);
    }
    
    if (stemBuffers.other) {
        otherSource = audioContext.createBufferSource();
        otherSource.buffer = stemBuffers.other;
        otherSource.connect(analyserOther);
        analyserOther.connect(audioContext.destination);
        otherSource.start(0);
    }
    
    // Set end event to update UI when playback finishes
    if (vocalsSource) {
        vocalsSource.onended = function() {
            if (isPlaying) {
                togglePlayback();
            }
        };
    }
}

// Stop all stems
function stopAllStems() {
    if (vocalsSource) {
        vocalsSource.stop();
        vocalsSource.disconnect();
    }
    
    if (drumsSource) {
        drumsSource.stop();
        drumsSource.disconnect();
    }
    
    if (bassSource) {
        bassSource.stop();
        bassSource.disconnect();
    }
    
    if (otherSource) {
        otherSource.stop();
        otherSource.disconnect();
    }
}

// Initialize vocals visualization (particles)
function initVocalsVisual() {
    // Create particle system for vocals (nature-inspired)
    const particleCount = 2000;
    const particles = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 50;
        positions[i + 1] = (Math.random() - 0.5) * 50;
        positions[i + 2] = (Math.random() - 0.5) * 50;
    }
    
    particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const particleMaterial = new THREE.PointsMaterial({
        color: 0xff9900,
        size: 0.5,
        transparent: true,
        blending: THREE.AdditiveBlending
    });
    
    vocalsVisual = new THREE.Points(particles, particleMaterial);
    scene.add(vocalsVisual);
}

// Initialize drums visualization (cubes)
function initDrumsVisual() {
    // Create grid of cubes for drums (noir-inspired)
    const gridSize = 8;
    const cubeSize = 1;
    const spacing = 1.5;
    
    drumsVisual = new THREE.Group();
    
    const material = new THREE.MeshPhongMaterial({
        color: 0xff0000,
        specular: 0xffffff,
        shininess: 100,
        flatShading: true
    });
    
    for (let x = 0; x < gridSize; x++) {
        for (let z = 0; z < gridSize; z++) {
            const geometry = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
            const cube = new THREE.Mesh(geometry, material);
            
            cube.position.x = (x - gridSize / 2) * spacing;
            cube.position.z = (z - gridSize / 2) * spacing;
            cube.position.y = -10;
            
            cube.userData = {
                originalY: -10,
                index: x * gridSize + z
            };
            
            drumsVisual.add(cube);
        }
    }
    
    scene.add(drumsVisual);
}

// Initialize bass visualization (waves)
function initBassVisual() {
    // Create wave surface for bass (dance-inspired)
    const geometry = new THREE.PlaneGeometry(40, 40, 32, 32);
    const material = new THREE.MeshPhongMaterial({
        color: 0x0000ff,
        wireframe: true,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.8
    });
    
    bassVisual = new THREE.Mesh(geometry, material);
    bassVisual.rotation.x = Math.PI / 2;
    bassVisual.position.y = -15;
    
    scene.add(bassVisual);
}

// Initialize other instruments visualization (orbiting spheres)
function initOtherVisual() {
    // Create orbiting spheres for other instruments (epic fusion)
    otherVisual = new THREE.Group();
    
    const sphereCount = 12;
    const orbitRadius = 15;
    
    for (let i = 0; i < sphereCount; i++) {
        const geometry = new THREE.SphereGeometry(1, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: 0x00cc00,
            emissive: 0x003300
        });
        
        const sphere = new THREE.Mesh(geometry, material);
        
        const angle = (i / sphereCount) * Math.PI * 2;
        sphere.position.x = Math.cos(angle) * orbitRadius;
        sphere.position.z = Math.sin(angle) * orbitRadius;
        
        sphere.userData = {
            angle: angle,
            radius: orbitRadius,
            speed: 0.005 + Math.random() * 0.01,
            pulseSpeed: 0.05 + Math.random() * 0.05
        };
        
        otherVisual.add(sphere);
    }
    
    scene.add(otherVisual);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Update visualizations based on audio data
    if (isPlaying) {
        updateVocalsVisual();
        updateDrumsVisual();
        updateBassVisual();
        updateOtherVisual();
    }
    
    // Rotate camera slowly
    camera.position.x = Math.sin(Date.now() * 0.0002) * 30;
    camera.position.z = Math.cos(Date.now() * 0.0002) * 30;
    camera.lookAt(scene.position);
    
    renderer.render(scene, camera);
}

// Update vocals visualization
function updateVocalsVisual() {
    if (!analyserVocals) return;
    
    analyserVocals.getByteFrequencyData(vocalsData);
    
    const positions = vocalsVisual.geometry.attributes.position.array;
    const count = positions.length / 3;
    
    for (let i = 0; i < count; i++) {
        const freqIndex = i % vocalsData.length;
        const freqValue = vocalsData[freqIndex] / 255;
        
        const scale = 1 + freqValue * 2;
        
        // Apply frequency data to particle positions
        positions[i * 3 + 1] += (freqValue * 0.3) * (Math.random() - 0.5);
        
        // Contain particles within bounds
        if (Math.abs(positions[i * 3]) > 25) {
            positions[i * 3] *= 0.95;
        }
        if (Math.abs(positions[i * 3 + 1]) > 25) {
            positions[i * 3 + 1] *= 0.95;
        }
        if (Math.abs(positions[i * 3 + 2]) > 25) {
            positions[i * 3 + 2] *= 0.95;
        }
    }
    
    vocalsVisual.geometry.attributes.position.needsUpdate = true;
    
    // Update particle size based on average frequency
    const avgFreq = vocalsData.reduce((a, b) => a + b, 0) / vocalsData.length;
    vocalsVisual.material.size = 0.5 + (avgFreq / 255) * 1.5;
}

// Update drums visualization
function updateDrumsVisual() {
    if (!analyserDrums) return;
    
    analyserDrums.getByteFrequencyData(drumsData);
    
    // Update each cube in the grid
    drumsVisual.children.forEach((cube, index) => {
        const freqIndex = index % drumsData.length;
        const freqValue = drumsData[freqIndex] / 255;
        
        // Scale height based on frequency
        cube.scale.y = 1 + freqValue * 5;
        
        // Update position to compensate for scaling
        cube.position.y = cube.userData.originalY + (cube.scale.y / 2);
        
        // Rotate cube
        cube.rotation.x += 0.02 * freqValue;
        cube.rotation.z += 0.02 * freqValue;
    });
}

// Update bass visualization
function updateBassVisual() {
    if (!analyserBass) return;
    
    analyserBass.getByteFrequencyData(bassData);
    
    // Update wave geometry vertices
    const positions = bassVisual.geometry.attributes.position.array;
    const count = positions.length / 3;
    
    for (let i = 0; i < count; i++) {
        const freqIndex = i % bassData.length;
        const freqValue = bassData[freqIndex] / 255;
        
        // Skip x and z coordinates, only modify y
        positions[i * 3 + 2] = Math.sin(i / 30 + Date.now() * 0.001) * freqValue * 5;
    }
    
    bassVisual.geometry.attributes.position.needsUpdate = true;
    
    // Add subtle rotation
    bassVisual.rotation.z += 0.001;
}

// Update other instruments visualization
function updateOtherVisual() {
    if (!analyserOther) return;
    
    analyserOther.getByteFrequencyData(otherData);
    
    // Update each sphere in the orbit
    otherVisual.children.forEach((sphere, index) => {
        const freqIndex = index % otherData.length;
        const freqValue = otherData[freqIndex] / 255;
        
        // Update orbit position
        sphere.userData.angle += sphere.userData.speed * (1 + freqValue);
        
        // Scale sphere based on frequency
        sphere.scale.set(
            1 + freqValue * 1.5,
            1 + freqValue * 1.5,
            1 + freqValue * 1.5
        );
        
        // Update position
        sphere.position.x = Math.cos(sphere.userData.angle) * sphere.userData.radius;
        sphere.position.z = Math.sin(sphere.userData.angle) * sphere.userData.radius;
        
        // Add some y-axis movement
        sphere.position.y = Math.sin(Date.now() * sphere.userData.pulseSpeed) * 2 * freqValue;
    });
    
    // Rotate the entire orbit group
    otherVisual.rotation.y += 0.003;
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