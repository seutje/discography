import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';

const ALBUM_ORDER = [
  'UNMEASURED GEOMETRY',
  'Algorithmically Optimized Collection of Keywords',
  'stderr',
  'Failure to Converge',
  'Residual Instabilities',
  'Closed Doors',
  'Net Worthless',
  'Noise Floor',
  'Orbital Garden',
  'Driftwood',
  'Broken Temple',
  'Lost Souls',
];

const ROOM_SPACING = 19;
const ROOM_WIDTH = 15;
const ROOM_DEPTH = 15;
const CAMERA_HEIGHT = 1.65;
const WALK_SPEED = 6.2;
const LOOK_SPEED = 0.0022;

const museum = {
  overlay: null,
  canvas: null,
  renderer: null,
  scene: null,
  camera: null,
  clock: new THREE.Clock(),
  raycaster: new THREE.Raycaster(),
  pointer: new THREE.Vector2(0, 0),
  rooms: [],
  stationObjects: [],
  materials: null,
  focusedStation: null,
  activeStation: null,
  yaw: 0,
  pitch: 0,
  keys: new Set(),
  touchMove: { x: 0, z: 0 },
  pointerLookId: null,
  previousLook: null,
  running: false,
  animationFrame: 0,
};

function el(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function createOverlay() {
  const overlay = el('div', 'museum-overlay');
  overlay.setAttribute('aria-label', '3D discography museum');
  overlay.innerHTML = `
    <canvas class="museum-canvas" aria-label="3D discography"></canvas>
    <div class="museum-hud">
      <div class="museum-panel">
        <div class="museum-room-title" data-museum-room>Loading museum</div>
        <div class="museum-track-title" data-museum-track>Chronological album rooms</div>
        <div class="museum-prompt" data-museum-prompt>Museum under construction</div>
      </div>
      <button class="secondary museum-exit" type="button" data-museum-exit>Exit</button>
    </div>
    <div class="museum-center" aria-hidden="true"></div>
    <div class="museum-mobile-controls" aria-label="Mobile movement controls">
      <div class="museum-move-pad">
        <button type="button" data-move="forward" aria-label="Move forward">↑</button>
        <button type="button" data-move="left" aria-label="Move left">←</button>
        <button type="button" data-move="right" aria-label="Move right">→</button>
        <button type="button" data-move="backward" aria-label="Move backward">↓</button>
      </div>
      <div class="museum-look-pad" data-look-pad>Look</div>
    </div>
    <div class="museum-lyrics" aria-live="polite">
      <div class="museum-lyric-side" data-lyric-prev></div>
      <div class="museum-lyric-current" data-lyric-current>Timed lyrics will appear here.</div>
      <div class="museum-lyric-side" data-lyric-next></div>
    </div>
    <div class="museum-transport" aria-label="Museum player controls">
      <div class="museum-transport-copy">
        <strong data-player-title>No track selected</strong>
        <span data-player-time>--:--</span>
      </div>
      <div class="museum-transport-buttons">
        <button class="secondary" type="button" data-player-prev aria-label="Previous track">Prev</button>
        <button class="secondary" type="button" data-player-rewind aria-label="Rewind 10 seconds">-10s</button>
        <button class="secondary" type="button" data-player-toggle aria-label="Play or pause">Play</button>
        <button class="secondary" type="button" data-player-forward aria-label="Forward 10 seconds">+10s</button>
        <button class="secondary" type="button" data-player-next aria-label="Next track">Next</button>
      </div>
    </div>
    <div class="museum-rotate-hint">Rotate your device for the 3D museum.</div>
  `;
  document.body.appendChild(overlay);
  museum.overlay = overlay;
  museum.canvas = overlay.querySelector('.museum-canvas');
  overlay.querySelector('[data-museum-exit]').addEventListener('click', closeMuseum);
  overlay.querySelector('[data-player-toggle]').addEventListener('click', () => window.discographyApp?.togglePlayback?.());
  overlay.querySelector('[data-player-prev]').addEventListener('click', () => window.discographyApp?.previousTrack?.());
  overlay.querySelector('[data-player-next]').addEventListener('click', () => window.discographyApp?.nextTrack?.());
  overlay.querySelector('[data-player-rewind]').addEventListener('click', () => window.discographyApp?.seekBy?.(-10));
  overlay.querySelector('[data-player-forward]').addEventListener('click', () => window.discographyApp?.seekBy?.(10));
  wireControls(overlay);
}

function createWoodTexture(base = '#6f4325', dark = '#3d2414', light = '#aa7442') {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y += 44) {
    const plankHeight = 40 + (y % 3) * 5;
    ctx.fillStyle = y % 88 === 0 ? base : '#7a4a2a';
    ctx.fillRect(0, y, canvas.width, plankHeight);
    ctx.strokeStyle = dark;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, y + plankHeight);
    ctx.lineTo(canvas.width, y + plankHeight);
    ctx.stroke();
    for (let x = 0; x < canvas.width; x += 90) {
      const wave = Math.sin((x + y) * 0.03) * 8;
      ctx.strokeStyle = x % 180 === 0 ? 'rgba(38, 22, 11, .38)' : 'rgba(217, 166, 100, .26)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y + 8 + wave);
      ctx.bezierCurveTo(x + 30, y + 22, x + 62, y + 2, x + 96, y + 25 + wave * 0.5);
      ctx.stroke();
    }
    ctx.fillStyle = light;
    ctx.globalAlpha = 0.16;
    ctx.fillRect(0, y + 5, canvas.width, 3);
    ctx.globalAlpha = 1;
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  return texture;
}

function createTextTexture(lines, options = {}) {
  const width = options.width || 1024;
  const height = options.height || 256;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = options.background || 'rgba(35, 21, 11, .88)';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = options.border || 'rgba(217, 166, 100, .7)';
  ctx.lineWidth = 8;
  ctx.strokeRect(4, 4, width - 8, height - 8);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const textLines = Array.isArray(lines) ? lines : [lines];
  const mainSize = options.size || 58;
  const gap = mainSize * 1.02;
  const startY = height / 2 - ((textLines.length - 1) * gap) / 2;
  textLines.forEach((line, index) => {
    ctx.font = `${index === 0 && options.title ? 800 : 700} ${index === 0 ? mainSize : Math.round(mainSize * 0.7)}px Inter, Arial, sans-serif`;
    ctx.fillStyle = index === 0 ? options.color || '#f8ecd9' : options.subColor || '#d8c1a1';
    ctx.fillText(String(line || '').slice(0, 52), width / 2, startY + index * gap);
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function labelPlane(text, width, height, options = {}) {
  const material = new THREE.MeshBasicMaterial({
    map: createTextTexture(text, options),
    transparent: true,
    side: THREE.DoubleSide,
  });
  return new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
}

function initThree() {
  museum.scene = new THREE.Scene();
  museum.scene.background = new THREE.Color(0x120c07);
  museum.scene.fog = new THREE.FogExp2(0x120c07, 0.027);

  museum.camera = new THREE.PerspectiveCamera(72, 1, 0.1, 420);
  museum.camera.position.set(0, CAMERA_HEIGHT, 5);

  museum.renderer = new THREE.WebGLRenderer({
    canvas: museum.canvas,
    antialias: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: true,
  });
  museum.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
  museum.renderer.shadowMap.enabled = true;
  museum.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const ambient = new THREE.HemisphereLight(0xd6a65f, 0x130b05, 0.26);
  museum.scene.add(ambient);

  const flashlight = new THREE.SpotLight(0xffd79a, 6, 34, Math.PI / 7, 0.55, 1.2);
  flashlight.position.set(0, -0.08, 0);
  flashlight.target.position.set(0, -0.12, -1);
  museum.camera.add(flashlight, flashlight.target);
  museum.scene.add(museum.camera);

  buildMuseum();
  resizeMuseum();
  window.addEventListener('resize', resizeMuseum);
}

function materialSet() {
  return {
    floor: new THREE.MeshStandardMaterial({ map: createWoodTexture('#67401f', '#2a180c', '#c68a4e'), roughness: 0.86 }),
    wall: new THREE.MeshStandardMaterial({ color: 0x4c2e19, roughness: 0.92 }),
    beam: new THREE.MeshStandardMaterial({ color: 0x2d1a0d, roughness: 0.82 }),
    plaster: new THREE.MeshStandardMaterial({ color: 0x7f6648, roughness: 0.96 }),
    station: new THREE.MeshStandardMaterial({ color: 0xa16432, roughness: 0.72, metalness: 0.08 }),
    stationActive: new THREE.MeshStandardMaterial({ color: 0xd99a4a, roughness: 0.45, emissive: 0x3d2108, emissiveIntensity: 0.55 }),
    crate: new THREE.MeshStandardMaterial({ color: 0x5a3519, roughness: 0.9 }),
  };
}

function addBox(size, position, material, cast = true) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z), material);
  mesh.position.copy(position);
  mesh.castShadow = cast;
  mesh.receiveShadow = true;
  museum.scene.add(mesh);
  return mesh;
}

function buildMuseum() {
  const app = window.discographyApp;
  const byName = new Map((app?.getAlbums?.() || []).map(album => [album.name, album]));
  const albums = ALBUM_ORDER.map(name => byName.get(name)).filter(Boolean);
  const mats = materialSet();
  museum.materials = mats;

  const corridorLength = Math.max(ROOM_SPACING * albums.length + 18, 40);
  addBox(new THREE.Vector3(ROOM_WIDTH + 4, 0.24, corridorLength), new THREE.Vector3(0, -0.12, -corridorLength / 2 + 8), mats.floor, false);

  albums.forEach((album, index) => {
    const z = -index * ROOM_SPACING;
    museum.rooms.push({ album, zMin: z - ROOM_DEPTH / 2, zMax: z + ROOM_DEPTH / 2 });
    buildRoom(album, index, z, mats);
  });

}

function buildRoom(album, index, z, mats) {
  const halfW = ROOM_WIDTH / 2;
  const halfD = ROOM_DEPTH / 2;
  const doorWidth = 4.2;
  const doorHeight = 2.55;
  const sideWallWidth = (ROOM_WIDTH - doorWidth) / 2;
  const backZ = z - halfD;
  addBox(new THREE.Vector3(ROOM_WIDTH, 0.18, ROOM_DEPTH), new THREE.Vector3(0, 0, z), mats.floor, false);
  addBox(new THREE.Vector3(0.28, 3.7, ROOM_DEPTH), new THREE.Vector3(-halfW, 1.85, z), mats.wall, false);
  addBox(new THREE.Vector3(0.28, 3.7, ROOM_DEPTH), new THREE.Vector3(halfW, 1.85, z), mats.wall, false);
  addBox(new THREE.Vector3(sideWallWidth, 3.7, 0.28), new THREE.Vector3(-(doorWidth / 2 + sideWallWidth / 2), 1.85, backZ), mats.wall, false);
  addBox(new THREE.Vector3(sideWallWidth, 3.7, 0.28), new THREE.Vector3(doorWidth / 2 + sideWallWidth / 2, 1.85, backZ), mats.wall, false);
  addBox(new THREE.Vector3(doorWidth, 3.7 - doorHeight, 0.28), new THREE.Vector3(0, doorHeight + (3.7 - doorHeight) / 2, backZ), mats.wall, false);
  addBox(new THREE.Vector3(ROOM_WIDTH, 0.22, 0.28), new THREE.Vector3(0, 3.7, z + halfD), mats.beam, false);
  addBox(new THREE.Vector3(0.18, doorHeight, 0.22), new THREE.Vector3(-doorWidth / 2, doorHeight / 2, backZ + 0.03), mats.beam);
  addBox(new THREE.Vector3(0.18, doorHeight, 0.22), new THREE.Vector3(doorWidth / 2, doorHeight / 2, backZ + 0.03), mats.beam);
  addBox(new THREE.Vector3(doorWidth + 0.36, 0.18, 0.22), new THREE.Vector3(0, doorHeight, backZ + 0.03), mats.beam);

  for (let beam = -2; beam <= 2; beam += 1) {
    addBox(new THREE.Vector3(0.16, 3.9, 0.16), new THREE.Vector3(-halfW + 1.1, 1.95, z + beam * 2.6), mats.beam);
    addBox(new THREE.Vector3(0.16, 3.9, 0.16), new THREE.Vector3(halfW - 1.1, 1.95, z + beam * 2.6), mats.beam);
  }

  const title = labelPlane([album.name, `${album.track_count || (album.tracks || []).length} songs`], 6.5, 1.3, {
    title: true,
    size: album.name.length > 22 ? 44 : 54,
    background: 'rgba(49, 29, 13, .94)',
  });
  title.position.set(0, 3.12, backZ + 0.18);
  museum.scene.add(title);

  const workLight = new THREE.PointLight(0xd88a3c, 0.75, 8, 1.5);
  workLight.position.set(index % 2 ? -3.4 : 3.4, 2.8, z - 2);
  museum.scene.add(workLight);

  addConstructionClutter(z, mats, index);
  addSongStations(album, z, mats);
}

function addConstructionClutter(z, mats, index) {
  const baseX = index % 2 ? 4.8 : -4.8;
  for (let i = 0; i < 3; i += 1) {
    addBox(new THREE.Vector3(1.2, 0.55 + i * 0.08, 0.85), new THREE.Vector3(baseX + i * 0.7, 0.28 + i * 0.04, z - 5 + i * 1.2), mats.crate);
  }
  addBox(new THREE.Vector3(4.5, 0.12, 0.18), new THREE.Vector3(-baseX * 0.65, 1.35, z + 4.6), mats.beam);
  addBox(new THREE.Vector3(0.18, 1.8, 0.18), new THREE.Vector3(-baseX * 0.65 - 2.1, 0.9, z + 4.6), mats.beam);
  addBox(new THREE.Vector3(0.18, 1.8, 0.18), new THREE.Vector3(-baseX * 0.65 + 2.1, 0.9, z + 4.6), mats.beam);
}

function addSongStations(album, z, mats) {
  const tracks = (album.tracks || []).filter(track => track.audio_url);
  const left = tracks.filter((_, index) => index % 2 === 0);
  const right = tracks.filter((_, index) => index % 2 === 1);
  placeStationColumn(album, left, -6.35, z, 1, mats);
  placeStationColumn(album, right, 6.35, z, -1, mats);
}

function placeStationColumn(album, tracks, x, z, facing, mats) {
  const gap = Math.min(2.2, 11 / Math.max(1, tracks.length - 1));
  const startZ = z + ((tracks.length - 1) * gap) / 2;
  tracks.forEach((track, index) => {
    const stationZ = startZ - index * gap;
    const station = addBox(new THREE.Vector3(0.55, 1.15, 1.25), new THREE.Vector3(x, 0.65, stationZ), mats.station);
    station.userData = { albumName: album.name, trackTitle: track.title, kind: 'station' };
    museum.stationObjects.push(station);
    const label = labelPlane([`${track.track_number ?? track.index}. ${track.title}`], 1.72, 0.58, {
      size: track.title.length > 24 ? 24 : 29,
      background: 'rgba(43, 26, 13, .92)',
    });
    label.position.set(x - facing * 0.42, 1.55, stationZ);
    label.rotation.y = facing > 0 ? Math.PI / 2 : -Math.PI / 2;
    label.userData = { albumName: album.name, trackTitle: track.title, kind: 'station-label' };
    museum.stationObjects.push(label);
    museum.scene.add(label);
  });
}

function wireControls(overlay) {
  document.addEventListener('keydown', event => {
    if (!museum.running) return;
    if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.code)) {
      museum.keys.add(event.code);
      event.preventDefault();
    }
    if (event.code === 'Space' || event.code === 'KeyP') {
      event.preventDefault();
      window.discographyApp?.togglePlayback?.();
    }
    if (event.code === 'KeyN') {
      event.preventDefault();
      window.discographyApp?.nextTrack?.();
    }
    if (event.code === 'KeyB') {
      event.preventDefault();
      window.discographyApp?.previousTrack?.();
    }
    if (event.code === 'BracketLeft') {
      event.preventDefault();
      window.discographyApp?.seekBy?.(-10);
    }
    if (event.code === 'BracketRight') {
      event.preventDefault();
      window.discographyApp?.seekBy?.(10);
    }
    if (event.code === 'Escape' && document.pointerLockElement !== museum.canvas) closeMuseum();
  });
  document.addEventListener('keyup', event => museum.keys.delete(event.code));
  document.addEventListener('pointerlockchange', () => {
    museum.canvas.classList.toggle('locked', document.pointerLockElement === museum.canvas);
  });
  document.addEventListener('mousemove', event => {
    if (!museum.running || document.pointerLockElement !== museum.canvas) return;
    lookBy(event.movementX, event.movementY);
  });
  museum.canvas.addEventListener('click', event => {
    if (!museum.running) return;
    updateFocusFromClient(event.clientX, event.clientY);
    if (museum.focusedStation) {
      playFocusedStation();
      return;
    }
    if (document.pointerLockElement !== museum.canvas && museum.canvas.requestPointerLock) {
      museum.canvas.requestPointerLock();
    }
  });
  museum.canvas.addEventListener('pointermove', event => {
    if (!museum.running || document.pointerLockElement === museum.canvas) return;
    updateFocusFromClient(event.clientX, event.clientY);
    if (event.buttons === 1) lookBy(event.movementX, event.movementY);
  });

  overlay.querySelectorAll('[data-move]').forEach(button => {
    const setMove = pressed => {
      const direction = button.dataset.move;
      const value = pressed ? 1 : 0;
      if (direction === 'forward') museum.touchMove.z = -value;
      if (direction === 'backward') museum.touchMove.z = value;
      if (direction === 'left') museum.touchMove.x = -value;
      if (direction === 'right') museum.touchMove.x = value;
    };
    button.addEventListener('pointerdown', event => {
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      setMove(true);
    });
    button.addEventListener('pointerup', () => setMove(false));
    button.addEventListener('pointercancel', () => setMove(false));
    button.addEventListener('lostpointercapture', () => setMove(false));
  });

  const lookPad = overlay.querySelector('[data-look-pad]');
  lookPad.addEventListener('pointerdown', event => {
    event.preventDefault();
    museum.pointerLookId = event.pointerId;
    museum.previousLook = { x: event.clientX, y: event.clientY };
    lookPad.setPointerCapture(event.pointerId);
  });
  lookPad.addEventListener('pointermove', event => {
    if (museum.pointerLookId !== event.pointerId || !museum.previousLook) return;
    lookBy(event.clientX - museum.previousLook.x, event.clientY - museum.previousLook.y);
    museum.previousLook = { x: event.clientX, y: event.clientY };
  });
  lookPad.addEventListener('pointerup', () => {
    museum.pointerLookId = null;
    museum.previousLook = null;
  });
}

function lookBy(deltaX, deltaY) {
  museum.yaw -= deltaX * LOOK_SPEED;
  museum.pitch -= deltaY * LOOK_SPEED;
  museum.pitch = Math.max(-1.18, Math.min(1.1, museum.pitch));
}

function resizeMuseum() {
  if (!museum.renderer) return;
  const width = museum.overlay.clientWidth || window.innerWidth;
  const height = museum.overlay.clientHeight || window.innerHeight;
  museum.camera.aspect = width / height;
  museum.camera.updateProjectionMatrix();
  museum.renderer.setSize(width, height, false);
}

function updateFocusFromClient(clientX, clientY) {
  const rect = museum.canvas.getBoundingClientRect();
  museum.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  museum.pointer.y = -(((clientY - rect.top) / rect.height) * 2 - 1);
  updateFocusedStation(museum.pointer);
}

function updateFocusedStation(pointer = new THREE.Vector2(0, 0)) {
  museum.raycaster.setFromCamera(pointer, museum.camera);
  const hits = museum.raycaster.intersectObjects(museum.stationObjects, false);
  const station = hits.find(hit => hit.distance < 8)?.object || null;
  const data = ['station', 'station-label'].includes(station?.userData?.kind) ? station.userData : null;
  museum.focusedStation = data?.trackTitle ? data : null;
  const prompt = museum.overlay.querySelector('[data-museum-prompt]');
  prompt.textContent = museum.focusedStation
    ? `Station: ${museum.focusedStation.trackTitle}`
    : 'Museum under construction';
}

function playFocusedStation() {
  const station = museum.focusedStation;
  if (!station) return;
  const played = window.discographyApp?.playCatalogTrack?.(station.albumName, station.trackTitle);
  if (played) {
    museum.activeStation = station;
    museum.overlay.querySelector('[data-museum-track]').textContent = station.trackTitle;
    museum.stationObjects.forEach(object => {
      if (object.userData?.kind === 'station') {
        object.material = object.userData.trackTitle === station.trackTitle && object.userData.albumName === station.albumName
          ? museum.materials.stationActive
          : museum.materials.station;
      }
    });
  }
}

function currentRoom() {
  const z = museum.camera.position.z;
  return museum.rooms.find(room => z >= room.zMin && z <= room.zMax) || museum.rooms[0];
}

function moveCamera(delta) {
  const forward = Number(museum.keys.has('KeyW') || museum.keys.has('ArrowUp')) -
    Number(museum.keys.has('KeyS') || museum.keys.has('ArrowDown')) -
    museum.touchMove.z;
  const strafe = Number(museum.keys.has('KeyD') || museum.keys.has('ArrowRight')) -
    Number(museum.keys.has('KeyA') || museum.keys.has('ArrowLeft')) +
    museum.touchMove.x;
  if (!forward && !strafe) return;
  const speed = WALK_SPEED * delta;
  const direction = new THREE.Vector3(strafe, 0, -forward);
  if (direction.lengthSq() > 1) direction.normalize();
  direction.applyAxisAngle(new THREE.Vector3(0, 1, 0), museum.yaw);
  museum.camera.position.addScaledVector(direction, speed);
  museum.camera.position.x = Math.max(-ROOM_WIDTH / 2 + 0.8, Math.min(ROOM_WIDTH / 2 - 0.8, museum.camera.position.x));
  const minZ = -((museum.rooms.length - 1) * ROOM_SPACING) - ROOM_DEPTH / 2 + 1;
  museum.camera.position.z = Math.max(minZ, Math.min(ROOM_DEPTH / 2 - 1, museum.camera.position.z));
  museum.camera.position.y = CAMERA_HEIGHT;
}

function formatPlayerTime(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value <= 0) return '0:00';
  const minutes = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${minutes}:${String(secs).padStart(2, '0')}`;
}

function updateHud() {
  const room = currentRoom();
  if (room) {
    museum.overlay.querySelector('[data-museum-room]').textContent = room.album.name;
  }
  const snapshot = window.discographyApp?.getLyricSnapshot?.() || {};
  museum.overlay.querySelector('[data-lyric-prev]').textContent = snapshot.previous || '';
  museum.overlay.querySelector('[data-lyric-current]').textContent = snapshot.current || 'Timed lyrics will appear here.';
  museum.overlay.querySelector('[data-lyric-next]').textContent = snapshot.next || '';
  if (snapshot.title) museum.overlay.querySelector('[data-museum-track]').textContent = snapshot.title;

  const player = window.discographyApp?.getPlayerSnapshot?.() || {};
  museum.overlay.querySelector('[data-player-title]').textContent = player.title || 'No track selected';
  museum.overlay.querySelector('[data-player-time]').textContent = player.hasTrack
    ? `${formatPlayerTime(player.currentTime)} / ${formatPlayerTime(player.duration)}`
    : '--:--';
  museum.overlay.querySelector('[data-player-toggle]').textContent = player.hasTrack && !player.paused ? 'Pause' : 'Play';
  museum.overlay.querySelectorAll('[data-player-prev], [data-player-rewind], [data-player-toggle], [data-player-forward], [data-player-next]')
    .forEach(button => {
      button.disabled = !player.hasTrack;
    });

}

function animate() {
  if (!museum.running) return;
  const delta = Math.min(0.05, museum.clock.getDelta());
  museum.camera.rotation.order = 'YXZ';
  museum.camera.rotation.y = museum.yaw;
  museum.camera.rotation.x = museum.pitch;
  moveCamera(delta);
  updateFocusedStation();
  updateHud();
  museum.renderer.render(museum.scene, museum.camera);
  museum.animationFrame = requestAnimationFrame(animate);
}

async function openMuseum() {
  if (!museum.overlay) createOverlay();
  await waitForAlbums();
  if (!museum.renderer) initThree();
  museum.overlay.classList.add('active');
  document.body.classList.add('museum-active');
  museum.running = true;
  museum.clock.start();
  resizeMuseum();
  try {
    if (!document.fullscreenElement && museum.overlay.requestFullscreen) {
      await museum.overlay.requestFullscreen({ navigationUI: 'hide' });
    }
    await screen.orientation?.lock?.('landscape');
  } catch {
    // Browsers may reject fullscreen or orientation lock outside supported contexts.
  }
  cancelAnimationFrame(museum.animationFrame);
  animate();
}

async function waitForAlbums() {
  const started = performance.now();
  while (!(window.discographyApp?.getAlbums?.() || []).length && performance.now() - started < 8000) {
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

async function closeMuseum() {
  museum.running = false;
  cancelAnimationFrame(museum.animationFrame);
  museum.keys.clear();
  museum.touchMove.x = 0;
  museum.touchMove.z = 0;
  museum.overlay?.classList.remove('active');
  document.body.classList.remove('museum-active');
  if (document.pointerLockElement === museum.canvas) document.exitPointerLock();
  try {
    if (document.fullscreenElement === museum.overlay) await document.exitFullscreen();
  } catch {
    // Ignore browser fullscreen teardown errors.
  }
}

document.getElementById('museumBtn')?.addEventListener('click', openMuseum);
