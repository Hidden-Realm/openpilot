// DriveSight World: live 3D view of the map the comma is building, drawn on the viewer's own GPU.
//
// Frames: the server streams the car pose in a local east/north frame (meters) and everything the car sees in
// its own frame (x forward, y right). In three.js, x = east, y = up, z = -north. A car-frame point (x, y) is a
// child of the car group at (x, height, y): the group's yaw turns its +x to the heading and its +z to the right.
import * as THREE from "three";
import { OrbitControls } from "/static/OrbitControls.js";

const COLORS = {
  teal: 0x2dd4bf, amber: 0xfbbf24, coral: 0xfb7185, steel: 0x94a3b8, white: 0xe8ecf6, slate: 0x1e293b,
  red: 0xef4444, yellow: 0xfacc15, green: 0x22c55e, ink: 0x070b18,
};
const MPH = 2.23694;
const TILE_POLL_MS = 3000;
const FOLLOW_RESUME_MS = 8000;

// ---------------------------------------------------------------- renderer, cameras, lights
const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x000000, 0);
const scene = new THREE.Scene();

const chase = new THREE.PerspectiveCamera(55, 1, 0.5, 1500);
chase.position.set(-14, 7, 0);
const ortho = new THREE.OrthographicCamera(-100, 100, 100, -100, -1000, 1000);
ortho.up.set(0, 0, -1);  // north up
ortho.position.set(0, 300, 0);
ortho.lookAt(0, 0, 0);

scene.add(new THREE.HemisphereLight(0x9fb4ff, 0x0b1020, 1.1));
const sun = new THREE.DirectionalLight(0xffe7c2, 1.2);
sun.position.set(-40, 80, 30);
scene.add(sun);

let mode = "drive";
let camera = chase;
const controls = new OrbitControls(chase, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 6;
controls.maxDistance = 400;
const mapControls = new OrbitControls(ortho, canvas);
mapControls.enableRotate = false;
mapControls.screenSpacePanning = true;
mapControls.enabled = false;
let manualUntil = 0;
controls.addEventListener("start", () => { manualUntil = performance.now() + FOLLOW_RESUME_MS; });
controls.addEventListener("end", () => { manualUntil = performance.now() + FOLLOW_RESUME_MS; });

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  chase.aspect = w / h;
  chase.updateProjectionMatrix();
  const half = ortho.userData.half || 120;
  ortho.left = -half * w / h; ortho.right = half * w / h; ortho.top = half; ortho.bottom = -half;
  ortho.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------- ground: a dot grid that fades with distance
const groundMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false,
  uniforms: { uCar: { value: new THREE.Vector2() } },
  vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `uniform vec2 uCar; varying vec3 vW;
    void main(){
      vec2 g = fract(vW.xz / 4.0) - 0.5;
      float d = smoothstep(0.07, 0.0, length(g));
      float fade = exp(-length(vW.xz - uCar) / 160.0);
      gl_FragColor = vec4(vec3(0.42, 0.52, 0.80), d * fade * 0.55);
    }`,
});
const ground = new THREE.Mesh(new THREE.PlaneGeometry(1200, 1200), groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.02;
ground.renderOrder = 0;
scene.add(ground);

// ---------------------------------------------------------------- map tiles
// R road, G edges, B lane lines, A how well observed. Unobserved cells are left out entirely.
const tileMat = (tex) => new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  uniforms: { map: { value: tex }, uCar: { value: new THREE.Vector2() }, uFade: { value: 1.0 } },
  vertexShader: `varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `uniform sampler2D map; uniform vec2 uCar; uniform float uFade; varying vec2 vUv; varying vec3 vW;
    void main(){
      vec4 t = texture2D(map, vUv);
      float road = t.r, edge = t.g, lane = t.b, seen = t.a;
      if (seen < 0.03) discard;
      float band = 0.5 + 0.5 * sin(dot(vW.xz, vec2(0.7, 0.7)) * 1.6);          // faint contour texture on the road
      float edgeV = smoothstep(0.2, 0.9, edge);   // the core of a mapped line, not its blurred skirt
      float laneV = smoothstep(0.2, 0.9, lane);
      vec3 col = vec3(0.09, 0.13, 0.24) * (0.45 + 0.9 * road) * (0.9 + 0.1 * band);
      col = mix(col, vec3(0.18, 0.83, 0.75), edgeV * 0.85);
      col = mix(col, vec3(0.80, 0.85, 0.95), laneV * 0.55);
      float a = clamp(max(road * 0.8, max(edgeV * 0.85, laneV * 0.6)), 0.0, 1.0) * smoothstep(0.03, 0.35, seen);
      float fade = mix(1.0, exp(-length(vW.xz - uCar) / 260.0), uFade);
      gl_FragColor = vec4(col, a * fade);
    }`,
});
const tiles = new Map();   // key -> { mesh, rev }
let tilesOrigin = null;
const loader = new THREE.TextureLoader();

function tileMesh(t) {
  const c = t.corners;  // SW, SE, NE, NW in the local east/north frame
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute([
    c[0][0], 0, -c[0][1], c[1][0], 0, -c[1][1], c[2][0], 0, -c[2][1], c[3][0], 0, -c[3][1]], 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  g.setIndex([0, 2, 1, 0, 3, 2]);
  const mesh = new THREE.Mesh(g, tileMat(null));
  mesh.renderOrder = 1;
  return mesh;
}

function clearTiles() {
  for (const { mesh } of tiles.values()) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.uniforms.map.value?.dispose(); mesh.material.dispose(); }
  tiles.clear();
}

async function pollTiles() {
  try {
    const r = await fetch(mode === "map" ? "/api/tiles?all=1" : "/api/tiles?r=500");
    if (!r.ok) return;
    const { origin, tiles: list } = await r.json();
    if (!origin) return;
    if (!tilesOrigin || tilesOrigin.id !== origin.id) { clearTiles(); tilesOrigin = origin; }
    const keep = new Set();
    for (const t of list) {
      keep.add(t.key);
      let entry = tiles.get(t.key);
      if (!entry) {
        entry = { mesh: tileMesh(t), rev: null };
        scene.add(entry.mesh);
        tiles.set(t.key, entry);
      }
      if (entry.rev !== t.rev) {
        entry.rev = t.rev;
        loader.load(`/api/tile/${t.key}.png?v=${encodeURIComponent(t.rev)}`, (tex) => {
          tex.colorSpace = THREE.NoColorSpace;
          tex.anisotropy = 4;
          const old = entry.mesh.material.uniforms.map.value;
          entry.mesh.material.uniforms.map.value = tex;
          if (old) old.dispose();
        });
      }
    }
    for (const [key, entry] of tiles) {
      if (!keep.has(key)) { scene.remove(entry.mesh); entry.mesh.geometry.dispose(); entry.mesh.material.dispose(); tiles.delete(key); }
    }
  } catch (e) { /* the car may be off; try again next time */ }
}
setInterval(pollTiles, TILE_POLL_MS);

// ---------------------------------------------------------------- the car and what it sees (car frame)
const car = new THREE.Group();
scene.add(car);
const carBody = new THREE.Group();  // scaled up on the zoomed-out map; what the car sees is not
car.add(carBody);
{
  const body = new THREE.Mesh(new THREE.BoxGeometry(4.6, 1.15, 1.85), new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.4, roughness: 0.5 }));
  body.position.set(-1.6, 0.75, 0);
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.6, 1.6), new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.3, roughness: 0.4 }));
  cabin.position.set(-1.9, 1.6, 0);
  const glow = new THREE.Mesh(new THREE.RingGeometry(2.6, 3.0, 48), new THREE.MeshBasicMaterial({ color: COLORS.teal, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
  glow.rotation.x = -Math.PI / 2;
  glow.position.set(-1.6, 0.03, 0);
  carBody.add(body, cabin, glow);
}
const live3d = new THREE.Group();  // rebuilt every update, in the car frame
car.add(live3d);

function ribbon(points, width, y = 0.05) {
  // a flat strip along a car-frame polyline [[x, y], ...]
  const pos = [];
  for (let i = 0; i < points.length - 1; i++) {
    const [x0, z0] = points[i], [x1, z1] = points[i + 1];
    const dx = x1 - x0, dz = z1 - z0, len = Math.hypot(dx, dz) || 1;
    const nx = -dz / len * width / 2, nz = dx / len * width / 2;
    pos.push(x0 + nx, y, z0 + nz, x0 - nx, y, z0 - nz, x1 + nx, y, z1 + nz);
    pos.push(x1 + nx, y, z1 + nz, x0 - nx, y, z0 - nz, x1 - nx, y, z1 - nz);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  return g;
}

function dashed(points, width, on, off) {
  // split a polyline into dashes of `on` meters every `on + off`
  const out = [];
  let acc = 0, cur = [points[0]];
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1], [x1, y1] = points[i];
    const seg = Math.hypot(x1 - x0, y1 - y0);
    let s = 0;
    while (s < seg) {
      const phase = acc % (on + off), drawing = phase < on;
      const step = Math.min(seg - s, drawing ? on - phase : on + off - phase);
      s += step; acc += step;
      const p = [x0 + (x1 - x0) * s / seg, y0 + (y1 - y0) * s / seg];
      if (drawing) cur.push(p); else { if (cur.length > 1) out.push(cur); cur = [p]; }
    }
  }
  if (cur.length > 1) out.push(cur);
  return out.map((p) => ribbon(p, width, 0.08));
}

function chevrons(points, color, every = 7) {
  const pos = [];
  let next = 6, dist = 0;
  for (let i = 1; i < points.length; i++) {
    const [x0, z0] = points[i - 1], [x1, z1] = points[i];
    const seg = Math.hypot(x1 - x0, z1 - z0);
    while (dist + seg >= next && next < 90) {
      const f = (next - dist) / seg, x = x0 + (x1 - x0) * f, z = z0 + (z1 - z0) * f;
      const ux = (x1 - x0) / seg, uz = (z1 - z0) / seg, w = 0.9, d = 1.0, t = 0.35;
      // a V pointing forward: two quads
      for (const side of [1, -1]) {
        const ax = x - ux * d - uz * w * side, az = z - uz * d + ux * w * side;
        const bx = x, bz = z;
        const nx = ux * t, nz = uz * t;
        pos.push(ax, 0.06, az, bx, 0.06, bz, ax + nx, 0.06, az + nz, ax + nx, 0.06, az + nz, bx, 0.06, bz, bx + nx, 0.06, bz + nz);
      }
      next += every;
    }
    dist += seg;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
}

function rebuildLive(L) {
  for (const c of [...live3d.children]) { live3d.remove(c); c.geometry?.dispose(); c.material?.dispose(); }
  buildLive(L);
  for (const c of live3d.children) c.renderOrder = 3;   // over the map tiles
}

function buildLive(L) {
  const m = L.model || {};
  for (const lane of m.lanes || []) {
    if (lane.pts.length < 2) continue;
    live3d.add(new THREE.Mesh(ribbon(lane.pts, 0.2), new THREE.MeshBasicMaterial({ color: COLORS.white, transparent: true, opacity: 0.25 + 0.7 * lane.prob, depthWrite: false, side: THREE.DoubleSide })));
  }
  for (const edge of m.edges || []) {
    if (edge.length < 2) continue;
    live3d.add(new THREE.Mesh(ribbon(edge, 0.22), new THREE.MeshBasicMaterial({ color: COLORS.teal, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide })));
  }
  const stopping = L.stop && L.stop.kind !== "lead";
  if ((m.path || []).length > 1 && L.speed > 0.3) live3d.add(chevrons(m.path, stopping ? COLORS.amber : COLORS.teal));
  if (L.map && L.map.valid && L.map.path.length > 1) {
    for (const g of dashed(L.map.path, 0.18, 2.0, 2.0)) {
      live3d.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: COLORS.amber, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide })));
    }
  }
  if (L.stop) {
    const d = L.stop.dist;
    const y = interpY(m.path || [], d);
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.08, 3.6), new THREE.MeshBasicMaterial({ color: L.stop.kind === "lead" ? COLORS.amber : COLORS.red }));
    bar.position.set(d, 0.06, y);
    live3d.add(bar);
  }
}

function interpY(path, x) {
  for (let i = 1; i < path.length; i++) {
    if (path[i][0] >= x) {
      const [x0, y0] = path[i - 1], [x1, y1] = path[i];
      return y0 + (y1 - y0) * (x - x0) / Math.max(x1 - x0, 1e-3);
    }
  }
  return path.length ? path[path.length - 1][1] : 0;
}

// ---------------------------------------------------------------- tracked objects (car frame, smoothed)
const objects = new Map();  // id -> { group, target }
const VEHICLE_LEN = { car: 4.6, truck: 8.5, bus: 11.5, motorcycle: 2.2, bicycle: 1.8 };

function makeObject(o) {
  const g = new THREE.Group();
  const lambert = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.2, ...extra });
  if (o.kind === "person") {
    const legs = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.18, 1.2, 12), lambert(COLORS.amber, { emissive: 0x3a2a00 }));
    legs.position.y = 0.6;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.2, 16, 12), lambert(COLORS.amber, { emissive: 0x3a2a00 }));
    head.position.y = 1.5;
    g.add(legs, head);
  } else if (o.kind === "trafficLight") {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 4.2, 8), lambert(0x475569));
    pole.position.y = 2.1;
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.0, 0.4), lambert(0x111827));
    head.position.y = 4.6;
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), new THREE.MeshBasicMaterial({ color: 0x334155 }));
    lamp.position.set(-0.22, 4.6, 0);
    lamp.name = "lamp";
    g.add(pole, head, lamp);
  } else if (o.kind === "stopSign") {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 2.2, 8), lambert(0x94a3b8));
    pole.position.y = 1.1;
    const sign = new THREE.Mesh(new THREE.CircleGeometry(0.42, 8), new THREE.MeshBasicMaterial({ color: COLORS.red, side: THREE.DoubleSide }));
    sign.position.y = 2.4;
    sign.rotation.y = -Math.PI / 2;
    sign.rotation.z = Math.PI / 8;
    g.add(pole, sign);
  } else {
    const len = VEHICLE_LEN[o.kind] || 4.6, wid = Math.max(o.w || 1.8, 0.6), h = o.kind === "truck" || o.kind === "bus" ? 3.0 : (o.kind === "car" ? 1.45 : 1.3);
    const box = new THREE.Mesh(new THREE.BoxGeometry(len, h, wid), lambert(0x1f2a44, { metalness: 0.35 }));
    box.position.set(len / 2, h / 2, 0);  // the detected point is the near end
    const rim = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineBasicMaterial({ color: COLORS.steel }));
    rim.position.copy(box.position);
    rim.name = "rim";
    const ring = new THREE.Mesh(new THREE.RingGeometry(Math.max(len, wid) * 0.55, Math.max(len, wid) * 0.62, 40), new THREE.MeshBasicMaterial({ color: COLORS.steel, transparent: true, opacity: 0.3, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(len / 2, 0.03, 0);
    ring.name = "ring";
    g.add(box, rim, ring);
  }
  return g;
}

function syncObjects(list) {
  const seen = new Set();
  for (const o of list || []) {
    seen.add(o.id);
    let entry = objects.get(o.id);
    if (!entry) {
      entry = { group: makeObject(o), target: new THREE.Vector3(o.x, 0, o.y) };
      entry.group.position.copy(entry.target);
      car.add(entry.group);
      objects.set(o.id, entry);
    }
    entry.target.set(o.x, 0, o.y);
    entry.data = o;
    const lamp = entry.group.getObjectByName("lamp");
    if (lamp) lamp.material.color.setHex({ red: COLORS.red, yellow: COLORS.yellow, green: COLORS.green }[o.light] || 0x334155);
    const near = o.x > 0 && o.x < 35 && Math.abs(o.y) < 3.5;
    const rim = entry.group.getObjectByName("rim"), ring = entry.group.getObjectByName("ring");
    const color = near ? COLORS.coral : COLORS.steel;
    if (rim) rim.material.color.setHex(color);
    if (ring) ring.material.color.setHex(color);
  }
  for (const [id, entry] of objects) {
    if (!seen.has(id)) { car.remove(entry.group); objects.delete(id); }
  }
}

// ---------------------------------------------------------------- stops the map remembers (local frame)
const stopMarks = new THREE.Group();
scene.add(stopMarks);
function syncStops(list) {
  for (const c of [...stopMarks.children]) stopMarks.remove(c);
  for (const s of list || []) {
    const color = s.kind === "trafficLight" ? COLORS.amber : s.kind === "stopSign" ? COLORS.red : COLORS.steel;
    const h = (90 - s.bearing) * Math.PI / 180;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, 3.4), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.75 }));
    bar.position.set(s.e, 0.05, -s.n);
    bar.rotation.y = h;
    stopMarks.add(bar);
  }
}

// ---------------------------------------------------------------- live stream
let L = null;
let lastMsg = 0;
let lastT = null;
const es = new EventSource("/api/live");
es.onmessage = (ev) => {
  L = JSON.parse(ev.data);
  lastMsg = performance.now();
  if (L.t !== lastT) {
    lastT = L.t;
    rebuildLive(L);
    syncObjects(L.objects);
    syncStops(L.stops);
    if (!tilesOrigin || (L.origin && tilesOrigin.id !== L.origin.id)) pollTiles();
  }
  updateHud();
};

// ---------------------------------------------------------------- HUD and info
const $ = (id) => document.getElementById(id);
function pill(id, cls, text) {
  const el = $(id);
  el.className = `pill ${cls}`;
  el.querySelector("span").textContent = text;
}
function rows(id, pairs) {
  $(id).innerHTML = pairs.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}
const fmt = (v, d = 2, unit = "") => (v === null || v === undefined || Number.isNaN(v)) ? "--" : `${Number(v).toFixed(d)}${unit}`;
const mb = (b) => b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`;

function updateHud() {
  if (!L) return;
  const fresh = performance.now() - lastMsg < 2000;
  pill("p-link", fresh ? "ok" : "bad", fresh ? "live" : "link lost");
  const poseOk = L.valid && L.pose;
  if (!poseOk) pill("p-pose", "bad", "no position");
  else if (L.align && L.align.converged && L.align.quality > 0.4) pill("p-pose", "ok", `locked to map ±${fmt(L.pose.std, 1)} m`);
  else pill("p-pose", "warn", `GPS ±${fmt(L.pose.std, 1)} m`);
  const s = L.steer || {};
  if (s.reason === "active" && s.weight > 0) pill("p-steer", "ok", `map steering ${Math.round(s.weight * 100)}%`);
  else if (s.qualified) pill("p-steer", s.reason === "off" ? "" : "warn", s.reason === "off" ? "map steering ready (off)" : `map steering: ${s.reason}`);
  else pill("p-steer", "", `map steering locked · ${fmt(s.kmToGo, 1)} km to go`);
  $("p-night").style.display = L.night ? "" : "none";
  pill("p-night", "warn", "night mode");

  $("speed").textContent = Math.round((L.speed || 0) * MPH);
  const map = L.map || {};
  if (map.latError === null || map.latError === undefined) {
    $("agree").textContent = map.valid ? "comparing" : "no map here yet";
    $("agree-bar").style.width = "0%";
  } else {
    const a = Math.exp(-((map.latError / 0.5) ** 2));
    $("agree").textContent = `${fmt(map.latError, 2)} m apart`;
    $("agree-bar").style.width = `${Math.round(a * 100)}%`;
    $("agree-bar").style.background = a > 0.6 ? "var(--teal)" : a > 0.3 ? "var(--amber)" : "var(--coral)";
  }
  $("mapconf").textContent = map.valid ? `${Math.round(map.conf * 100)}% mapped · ${map.passes} pass${map.passes === 1 ? "" : "es"}` : "new road";
  const objs = L.objects || [];
  const people = objs.filter((o) => o.kind === "person").length;
  const vehicles = objs.filter((o) => ["car", "truck", "bus", "motorcycle"].includes(o.kind)).length;
  $("around").textContent = `${vehicles} vehicle${vehicles === 1 ? "" : "s"} · ${people} ${people === 1 ? "person" : "people"}`;

  const ev = $("event");
  const walker = objs.find((o) => o.kind === "person" && o.x > 0 && o.x < 30 && Math.abs(o.y) < 3.5);
  if (walker) { ev.className = "walk"; ev.textContent = `PEDESTRIAN AHEAD · ${Math.round(walker.x)} m`; ev.style.display = "block"; }
  else if (L.stop && L.stop.kind !== "lead") {
    const what = { trafficLight: "LIGHT", stopSign: "STOP SIGN" }[L.stop.kind] || "STOP";
    ev.className = "stop"; ev.textContent = `STOPPING · ${what} · ${Math.round(L.stop.dist)} m`; ev.style.display = "block";
  } else ev.style.display = "none";

  $("offline").style.display = fresh && L.valid ? "none" : (mode === "info" ? "none" : "flex");
  $("offline-note").textContent = fresh ? "Waiting for a GPS position. The live view starts when the car is on and has a fix." : "Can't reach the device. Is your iPad on its Wi-Fi?";

  if (mode === "info") fillInfo();
}

function fillInfo() {
  const s = L.steer || {}, map = L.map || {}, d = L.detector || {}, p = L.perf || {}, a = L.align || {};
  rows("t-steer", [
    ["Status", s.qualified ? (s.reason === "active" ? "active" : s.reason) : "locked (confidence check)"],
    ["Weight now", `${Math.round((s.weight || 0) * 100)}%`],
    ["Compared", `${fmt(s.comparedKm, 1)} km`],
    ["Agreed", `${fmt(s.agreedKm, 1)} km (${Math.round((s.ratio || 0) * 100)}%)`],
    ["Needed to unlock", s.qualified ? "done" : `${fmt(s.kmToGo, 1)} more km at 90%`],
  ]);
  rows("t-map", [
    ["Position", L.pose ? `±${fmt(L.pose.std, 2)} m` : "--"],
    ["Locked to map", a.converged ? `yes (fit ${Math.round((a.quality || 0) * 100)}%)` : "no"],
    ["GPS correction", a.e !== undefined ? `${fmt(a.e, 2)} m E, ${fmt(a.n, 2)} m N` : "--"],
    ["Road ahead", map.valid ? `${Math.round(map.conf * 100)}% mapped` : "not mapped"],
    ["Map vs model", map.latError === null || map.latError === undefined ? "--" : `${fmt(map.latError, 2)} m`],
    ["Model lane confidence", fmt((L.model || {}).laneConf, 2)],
    ["Night", L.night ? "yes" : "no"],
  ]);
  rows("t-det", [
    ["Detector", d.ready ? "running" : (d.ready === false ? "no compiled model" : "off (device settings)")],
    ["Run time", fmt(d.ms, 1, " ms")],
    ["Rate", fmt(d.hz, 1, " Hz")],
    ["Skipped for driving", d.skipped ?? "--"],
    ["Map update", fmt(p.worldMs, 1, " ms")],
  ]);
  const objs = (L.objects || []).slice().sort((x, y) => Math.hypot(x.x, x.y) - Math.hypot(y.x, y.y));
  $("t-objs").innerHTML = objs.length ? objs.map((o) =>
    `<tr><td>${o.kind}${o.light && o.light !== "none" ? ` (${o.light})` : ""} · ${o.cam}</td><td>${fmt(o.x, 0)} m ahead, ${fmt(Math.abs(o.y), 1)} m ${o.y >= 0 ? "right" : "left"}</td></tr>`).join("")
    : `<tr><td>nothing tracked</td><td></td></tr>`;
  const sw = L.software || {};
  const running = sw.updaterRunning;
  const updater = sw.disabled ? "turned off (Disable Updates is on)"
    : running === null || running === undefined ? "--"
    : running ? (sw.state || "starting")
    : `not running${sw.updaterExit !== null && sw.updaterExit !== undefined ? ` (exit ${sw.updaterExit})` : ""}`;
  rows("t-sw", [
    ["Installed", sw.branch ? `${sw.branch} @ ${sw.commit}` : "--"],
    ["Switching to", sw.target && sw.target !== sw.branch ? sw.target : "--"],
    ["Updater", updater],
    ["Update", sw.installReady ? "ready: tap update now" : sw.downloadReady ? "available: tap download update" : "nothing new"],
    ["Failed checks", sw.failed ?? "--"],
    ["Last update", sw.lastUpdate || "--"],
  ]);
  $("sw-note").textContent = sw.disabled ? "Turn updates back on in sunnylink (or the Software settings) to update." :
    sw.error ? `Last updater error: ${sw.error}` :
    running === false ? "The updater only runs with the car off or in always offroad mode." : "";
  rows("t-data", [["Tiles", p.tiles ?? "--"], ["Size", p.mapBytes !== undefined ? mb(p.mapBytes) : "--"], ["Limit", "10 GB (oldest roads go first)"]]);
}

$("erase").addEventListener("click", async () => {
  if (!confirm("Erase the whole map? Everything recorded so far is deleted and map steering locks again.")) return;
  await fetch("/api/erase", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: "erase" }) });
  clearTiles();
  tilesOrigin = null;
  pollTiles();
});

for (const b of document.querySelectorAll("nav button")) {
  b.addEventListener("click", () => {
    mode = b.dataset.mode;
    for (const o of document.querySelectorAll("nav button")) o.classList.toggle("on", o === b);
    $("info").style.display = mode === "info" ? "block" : "none";
    camera = mode === "map" ? ortho : chase;
    controls.enabled = mode !== "map";
    mapControls.enabled = mode === "map";
    if (mode === "map") { mapControls.target.set(car.position.x, 0, car.position.z); ortho.position.set(car.position.x, 300, car.position.z); }
    tilesOrigin = null;
    pollTiles();
    updateHud();
  });
}

// ---------------------------------------------------------------- frame loop
const prevCar = new THREE.Vector3();
let poseH = 0;
function frame() {
  requestAnimationFrame(frame);
  step();
}

function step() {
  if (L && L.pose) {
    const target = new THREE.Vector3(L.pose.e, 0, -L.pose.n);
    if (car.position.distanceTo(target) > 50) car.position.copy(target);  // jumped (new origin or first fix)
    car.position.lerp(target, 0.25);
    let dh = L.pose.h - poseH;
    dh = Math.atan2(Math.sin(dh), Math.cos(dh));
    poseH += dh * 0.25;
    car.rotation.y = poseH;
  }
  for (const entry of objects.values()) entry.group.position.lerp(entry.target, 0.2);
  carBody.scale.setScalar(mode === "map" ? Math.max(1, 3 / ortho.zoom) : 1);  // stay findable when zoomed out
  groundMat.uniforms.uCar.value.set(car.position.x, car.position.z);
  ground.position.x = car.position.x;
  ground.position.z = car.position.z;
  for (const { mesh } of tiles.values()) {
    mesh.material.uniforms.uCar.value.set(car.position.x, car.position.z);
    mesh.material.uniforms.uFade.value = mode === "map" ? 0.0 : 1.0;
  }

  const moved = car.position.clone().sub(prevCar);
  prevCar.copy(car.position);
  if (mode !== "map") {
    const fwd = new THREE.Vector3(Math.cos(poseH), 0, -Math.sin(poseH));
    if (performance.now() > manualUntil) {
      const want = car.position.clone().addScaledVector(fwd, -15).add(new THREE.Vector3(0, 7.5, 0));
      chase.position.lerp(want, 0.08);
      controls.target.lerp(car.position.clone().addScaledVector(fwd, 20), 0.12);
    } else {
      chase.position.add(moved);       // orbiting by hand: keep orbiting around the moving car
      controls.target.add(moved);
    }
    controls.update();
  } else {
    mapControls.update();
  }
  renderer.render(scene, camera);
}
window.worldView = { scene, car, chase, ortho, controls, tiles, step };  // for poking at from the browser console
frame();
pollTiles();
