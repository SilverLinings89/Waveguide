// Waveguide solver frontend. No dependencies: the drawing, the field view and the convergence chart are painted on
// canvases by the code below.
"use strict";

const state = {
  info: null,          // defaults and limits from the server
  config: {},          // the configuration shown in the form
  derived: null,       // mesh, unknowns and memory estimate for it
  run: null,           // the run that is displayed (null: a new configuration)
  field: null,         // field data of the displayed plane
  showSetup: true,     // drawing shows the setup (true) or the computed field (false)
  poll: null,
};

const $ = (selector) => document.querySelector(selector);
const form = $("#config");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}

// ---- form -------------------------------------------------------------------------------------------------------

function writeForm(config) {
  for (const [key, value] of Object.entries(config)) {
    const inputs = form.querySelectorAll(`[name="${key}"]`);
    inputs.forEach((input) => {
      if (input.type === "radio") input.checked = String(value) === input.value;
      else if (input.type === "text" && typeof value === "number" && value !== 0 && Math.abs(value) < 1e-3) input.value = value.toExponential().replace("e-", "e-");
      else input.value = value;
    });
  }
  updateVisibility();
}

function readForm() {
  const config = {};
  for (const [key, fallback] of Object.entries(state.info.defaults)) {
    const inputs = form.querySelectorAll(`[name="${key}"]`);
    if (inputs.length === 0) { config[key] = state.config[key] ?? fallback; continue; }
    let raw = inputs[0].type === "radio" ? (form.querySelector(`[name="${key}"]:checked`) || {}).value : inputs[0].value;
    if (typeof fallback === "string") config[key] = raw ?? fallback;
    else {
      const number = Number(String(raw).replace(",", "."));
      config[key] = Number.isFinite(number) && raw !== "" ? number : NaN;
    }
  }
  return config;
}

function updateVisibility() {
  const shape = (form.querySelector('[name="shape"]:checked') || {}).value;
  form.querySelectorAll("[data-for-shape]").forEach((element) => {
    element.hidden = !element.dataset.forShape.split(" ").includes(shape);
  });
  const boundary = (form.querySelector('[name="boundary_method"]:checked') || {}).value;
  form.querySelectorAll("[data-for-boundary]").forEach((element) => { element.hidden = element.dataset.forBoundary !== boundary; });
  $("#cpw-output").textContent = form.querySelector('[name="cells_per_wavelength"]').value;
}

let estimateTimer = null;
function onFormChange() {
  updateVisibility();
  state.config = readForm();
  if (state.run) {           // editing a setting starts a new configuration based on the displayed run
    state.run = null; state.field = null; state.showSetup = true;
    $("#run-select").value = "";
    stopPolling();
    showRun(null);
  }
  clearTimeout(estimateTimer);
  estimateTimer = setTimeout(refreshEstimate, 120);
}

async function refreshEstimate() {
  form.querySelectorAll("input.invalid").forEach((input) => input.classList.remove("invalid"));
  const invalid = Object.entries(state.config).filter(([, value]) => typeof value === "number" && Number.isNaN(value));
  if (invalid.length > 0) {
    invalid.forEach(([key]) => form.querySelectorAll(`[name="${key}"]`).forEach((input) => input.classList.add("invalid")));
    return showProblems(["Enter a number for every setting."], true);
  }
  try {
    const result = await api("/api/estimate", { method: "POST", body: JSON.stringify(state.config) });
    state.derived = result.derived;
    showProblems(result.derived.warnings, false);
    $("#run").disabled = !state.info.solver_found;
  } catch (error) {
    state.derived = null;
    showProblems([error.message], true);
    $("#run").disabled = true;
  }
  renderEstimate();
  renderHints();
  drawStage();
}

function showProblems(messages, isError) {
  const list = $("#warnings");
  list.innerHTML = "";
  const all = [...messages];
  if (state.info && !state.info.solver_found) all.unshift("The solver is not built yet. Build it (see Readme.md) and restart the frontend.");
  all.forEach((message, index) => {
    const item = document.createElement("li");
    item.textContent = message;
    if (isError || (index === 0 && !state.info.solver_found)) item.className = "error";
    list.appendChild(item);
  });
  if (isError) $("#run").disabled = true;
}

const formatCount = (n) => n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + " million" : n >= 1e4 ? Math.round(n / 1e3) + " thousand" : String(Math.round(n));
const formatMemory = (gb) => gb >= 10 ? Math.round(gb) + " GB" : gb >= 1 ? gb.toFixed(1) + " GB" : Math.max(1, Math.round(gb * 1000)) + " MB";
function formatDuration(seconds) {
  if (seconds == null) return "";
  if (seconds < 90) return Math.round(seconds) + " s";
  if (seconds < 5400) return Math.round(seconds / 60) + " min";
  return (seconds / 3600).toFixed(1) + " h";
}

function renderEstimate() {
  const box = $("#estimate");
  const derived = state.run ? state.run.derived : state.derived;
  if (!derived) { box.innerHTML = ""; return; }
  const cells = derived.cells.join(" × ");
  const entries = [
    ["Unknowns", formatCount(derived.unknowns)],
    ["Cells", cells],
    ["Processes", String(derived.processes)],
    ["Memory, estimated", formatMemory(derived.memory_total_gb)],
  ];
  box.innerHTML = entries.map(([label, value]) => `<div><div class="value">${value}</div><div class="label">${label}</div></div>`).join("");
}

function renderHints() {
  const config = state.run ? state.run.config : state.config;
  const derived = state.run ? state.run.derived : state.derived;
  const defaults = state.info.defaults;
  const isDefaultGuide = ["core_width", "core_height", "epsilon_core", "epsilon_cladding"].every((key) => Math.abs(config[key] - defaults[key]) < 1e-9);
  $("#mode-hint").textContent = isDefaultGuide ? "" :
    "The input signal is the stored mode of the 2.0 × 1.8 µm core with ε 2.3409 in 1.8496. For another core it is not a mode of the waveguide, so part of it is radiated.";
  if (!derived) { $("#level-hint").textContent = ""; return; }
  const names = ["One process: the system is solved with a direct solver.", "One sweep along z.", "Sweeps along z, with sweeps along y inside.", "Sweeps along z, y and x."];
  const local = formatCount(derived.unknowns_local);
  $("#level-hint").textContent = `${names[derived.sweeping_level]} ${derived.cells_per_process.join(" × ")} cells and about ${local} unknowns per process.`;
  $("#inner-tolerance-field").hidden = derived.sweeping_level < 2;
}

// ---- geometry helpers ---------------------------------------------------------------------------------------------

// The shape function of the solver for a hump: piecewise linear slopes on 5 sectors per half, mirrored at the middle.
function humpProfile(config) {
  const L = config.length, half = L / 2, sectors = 5, h = half / sectors;
  const height = config.shape === "hump" ? config.hump_height : 0;
  const slopes = new Array(sectors + 1).fill(0);
  for (let i = 1; i <= sectors - 2; i++) slopes[i] = height / half;
  const integrate = (z) => {
    let value = 0;
    for (let i = 0; i < sectors; i++) {
      const width = Math.min(Math.max(z - i * h, 0), h);
      value += slopes[i] * width + 0.5 * (slopes[i + 1] - slopes[i]) * width * width / h;
    }
    return value;
  };
  slopes[sectors - 1] = (height - integrate((sectors - 2) * h)) / h - 0.5 * (slopes[sectors - 2] + slopes[sectors]);
  return (z) => integrate(z > half ? L - z : z);
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = Math.round(width * canvas.height / canvas.width);
  if (canvas.dataset.w !== String(width) || canvas.dataset.r !== String(ratio)) {
    canvas.dataset.w = width; canvas.dataset.r = ratio;
    canvas.dataset.cssHeight = height;
    canvas.width = width * ratio; canvas.height = height * ratio;
    canvas.style.height = height + "px";
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, Number(canvas.dataset.cssHeight));
  context.font = `12px ${css("--font")}`;
  return { context, width, height: Number(canvas.dataset.cssHeight) };
}

function hatch(context, x, y, w, h, color) {
  if (w <= 0 || h <= 0) return;
  context.save();
  context.beginPath(); context.rect(x, y, w, h); context.clip();
  context.strokeStyle = color; context.lineWidth = 1;
  for (let offset = -h; offset < w; offset += 6) {
    context.beginPath(); context.moveTo(x + offset, y + h); context.lineTo(x + offset + h, y); context.stroke();
  }
  context.restore();
}

// ---- the drawing: setup ---------------------------------------------------------------------------------------------

function drawStage() {
  if (!state.showSetup && state.field) return drawField();
  drawSetup();
}

function drawSetup() {
  const config = state.run ? state.run.config : state.config;
  const derived = state.run ? state.run.derived : state.derived;
  const { context: c, width, height } = setupCanvas($("#drawing"));
  $("#tooltip").hidden = true;
  if (!derived || !config.length) return;
  const ink = css("--ink"), ink2 = css("--ink-2"), muted = css("--muted");
  const usesPml = config.boundary_method === "PML";
  const pml = usesPml ? config.pml_thickness : 0;
  const pmlLeft = config.coupling === "Taper" ? pml : 0;
  const profile = humpProfile(config);
  const humpMax = config.shape === "hump" ? config.hump_height : 0;

  // Two panels: the side view (z horizontal, y vertical) and the cross-section (x horizontal, y vertical), same scale.
  const margin = { left: 70, right: 20, top: 30, bottom: 40 }, gap = 56;
  const totalZ = config.length + pml + pmlLeft, totalY = config.domain_height + 2 * pml + humpMax, totalX = config.domain_width + 2 * pml;
  let scale = Math.min((width - margin.left - margin.right - gap) / (totalZ + totalX), (height - margin.top - margin.bottom) / totalY);
  // Long waveguides: compress z so that the cross-section stays readable.
  let scaleZ = scale;
  const minScale = (height - margin.top - margin.bottom) / totalY * 0.9;
  if (scale < minScale * 0.6) {
    scale = minScale;
    scaleZ = (width - margin.left - margin.right - gap - totalX * scale) / totalZ;
  }
  const side = { x0: margin.left + pmlLeft * scaleZ, yMid: margin.top + (pml + humpMax + config.domain_height / 2) * scale };
  const Z = (z) => side.x0 + z * scaleZ;
  const Y = (y) => side.yMid - y * scale;
  const sectionX0 = margin.left + totalZ * scaleZ + gap + (pml + config.domain_width / 2) * scale;
  const X = (x) => sectionX0 + x * scale;
  const samples = 120;

  // side view: cladding (follows the shape of the waveguide axis), absorber, core
  const band = (lower, upper, fill) => {
    c.beginPath();
    for (let i = 0; i <= samples; i++) { const z = config.length * i / samples; c.lineTo(Z(z), Y(upper + profile(z))); }
    for (let i = samples; i >= 0; i--) { const z = config.length * i / samples; c.lineTo(Z(z), Y(lower + profile(z))); }
    c.closePath(); c.fillStyle = fill; c.fill();
  };
  band(-config.domain_height / 2, config.domain_height / 2, css("--cladding"));
  if (usesPml) {
    const absorber = css("--absorber");
    c.save();
    // top and bottom layers follow the shape; clip to their bands, then hatch
    for (const sign of [1, -1]) {
      c.save(); c.beginPath();
      for (let i = 0; i <= samples; i++) { const z = config.length * i / samples; c.lineTo(Z(z), Y(sign * (config.domain_height / 2 + pml) + profile(z))); }
      for (let i = samples; i >= 0; i--) { const z = config.length * i / samples; c.lineTo(Z(z), Y(sign * config.domain_height / 2 + profile(z))); }
      c.closePath(); c.clip();
      hatch(c, Z(0), Y(config.domain_height / 2 + pml + humpMax), config.length * scaleZ, (config.domain_height + 2 * pml + humpMax) * scale, absorber);
      c.restore();
    }
    hatch(c, Z(config.length), Y(config.domain_height / 2 + pml), pml * scaleZ, (config.domain_height + 2 * pml) * scale, absorber);
    if (pmlLeft > 0) hatch(c, Z(-pmlLeft), Y(config.domain_height / 2 + pml), pmlLeft * scaleZ, (config.domain_height + 2 * pml) * scale, absorber);
    c.restore();
  }
  band(-config.core_height / 2, config.core_height / 2, css("--glass-soft"));
  c.strokeStyle = css("--glass"); c.lineWidth = 1.5;
  for (const edge of [config.core_height / 2, -config.core_height / 2]) {
    c.beginPath();
    for (let i = 0; i <= samples; i++) { const z = config.length * i / samples; c.lineTo(Z(z), Y(edge + profile(z))); }
    c.stroke();
  }

  // mesh lines (only if they are at least 4 px apart) and process boundaries
  const [nx, ny, nz] = derived.cells;
  c.lineWidth = 1; c.strokeStyle = css("--grid"); c.globalAlpha = 0.55;
  if (config.length * scaleZ / nz >= 4) {
    for (let i = 1; i < nz; i++) { const z = config.length * i / nz; c.beginPath(); c.moveTo(Z(z) + 0.5, Y(config.domain_height / 2 + profile(z))); c.lineTo(Z(z) + 0.5, Y(-config.domain_height / 2 + profile(z))); c.stroke(); }
  }
  if (config.domain_height * scale / ny >= 4) {
    for (let j = 1; j < ny; j++) {
      const y = -config.domain_height / 2 + config.domain_height * j / ny;
      c.beginPath(); for (let i = 0; i <= samples; i++) { const z = config.length * i / samples; c.lineTo(Z(z), Y(y + profile(z))); } c.stroke();
    }
  }
  c.globalAlpha = 1;
  c.strokeStyle = ink2; c.setLineDash([5, 4]); c.lineWidth = 1.2;
  for (let i = 1; i < config.processes_z; i++) {
    const z = config.length * i / config.processes_z;
    c.beginPath(); c.moveTo(Z(z), Y(config.domain_height / 2 + profile(z)) - 6); c.lineTo(Z(z), Y(-config.domain_height / 2 + profile(z)) + 6); c.stroke();
  }
  for (let j = 1; j < config.processes_y; j++) {
    const y = -config.domain_height / 2 + config.domain_height * j / config.processes_y;
    c.beginPath(); for (let i = 0; i <= samples; i++) { const z = config.length * i / samples; c.lineTo(Z(z), Y(y + profile(z))); } c.stroke();
  }
  c.setLineDash([]);

  // signal input
  c.strokeStyle = ink; c.lineWidth = 2.5;
  if (config.coupling === "Dirichlet") {
    c.beginPath(); c.moveTo(Z(0), Y(config.domain_height / 2)); c.lineTo(Z(0), Y(-config.domain_height / 2)); c.stroke();
  } else {
    c.fillStyle = ink; c.globalAlpha = 0.12;
    c.fillRect(Z(0), Y(config.domain_height / 2), Math.min(1, config.length) * scaleZ, config.domain_height * scale);
    c.globalAlpha = 1;
  }
  // arrow of the incoming mode
  const arrowY = Y(0), arrowX = Z(0) - (pmlLeft > 0 ? 0 : 6);
  c.fillStyle = ink; c.strokeStyle = ink; c.lineWidth = 1.5;
  c.beginPath(); c.moveTo(arrowX - 30, arrowY); c.lineTo(arrowX - 8, arrowY); c.stroke();
  c.beginPath(); c.moveTo(arrowX - 2, arrowY); c.lineTo(arrowX - 10, arrowY - 4); c.lineTo(arrowX - 10, arrowY + 4); c.closePath(); c.fill();

  // cross-section
  if (usesPml) hatch(c, X(-config.domain_width / 2 - pml), Y(config.domain_height / 2 + pml), (config.domain_width + 2 * pml) * scale, (config.domain_height + 2 * pml) * scale, css("--absorber"));
  c.fillStyle = css("--cladding"); c.fillRect(X(-config.domain_width / 2), Y(config.domain_height / 2), config.domain_width * scale, config.domain_height * scale);
  c.lineWidth = 1; c.strokeStyle = css("--grid");
  if (config.domain_width * scale / nx >= 4) for (let i = 1; i < nx; i++) { const x = X(-config.domain_width / 2 + config.domain_width * i / nx); c.beginPath(); c.moveTo(x + 0.5, Y(config.domain_height / 2)); c.lineTo(x + 0.5, Y(-config.domain_height / 2)); c.stroke(); }
  if (config.domain_height * scale / ny >= 4) for (let j = 1; j < ny; j++) { const y = Y(-config.domain_height / 2 + config.domain_height * j / ny); c.beginPath(); c.moveTo(X(-config.domain_width / 2), y + 0.5); c.lineTo(X(config.domain_width / 2), y + 0.5); c.stroke(); }
  c.fillStyle = css("--glass-soft"); c.strokeStyle = css("--glass"); c.lineWidth = 1.5;
  c.fillRect(X(-config.core_width / 2), Y(config.core_height / 2), config.core_width * scale, config.core_height * scale);
  c.strokeRect(X(-config.core_width / 2), Y(config.core_height / 2), config.core_width * scale, config.core_height * scale);
  c.strokeStyle = ink2; c.setLineDash([5, 4]); c.lineWidth = 1.2;
  for (let i = 1; i < config.processes_x; i++) { const x = X(-config.domain_width / 2 + config.domain_width * i / config.processes_x); c.beginPath(); c.moveTo(x, Y(config.domain_height / 2) - 6); c.lineTo(x, Y(-config.domain_height / 2) + 6); c.stroke(); }
  for (let j = 1; j < config.processes_y; j++) { const y = Y(-config.domain_height / 2 + config.domain_height * j / config.processes_y); c.beginPath(); c.moveTo(X(-config.domain_width / 2) - 6, y); c.lineTo(X(config.domain_width / 2) + 6, y); c.stroke(); }
  c.setLineDash([]);

  // labels and dimensions
  c.fillStyle = ink2; c.textBaseline = "alphabetic"; c.textAlign = "left";
  c.fillText("Side view" + (scaleZ !== scale ? " (length compressed)" : ""), margin.left, 16);
  c.fillText("Cross-section", X(-config.domain_width / 2 - pml), 16);
  c.fillStyle = muted; c.textAlign = "center";
  const baseY = height - 14;
  c.fillText(`${config.length} µm, ${nz} cells`, (Z(0) + Z(config.length)) / 2, baseY);
  c.fillText(`${config.domain_width} µm, ${nx} cells`, X(0), baseY);
  c.save(); c.translate(16, Y(0)); c.rotate(-Math.PI / 2); c.fillText(`${config.domain_height} µm, ${ny} cells`, 0, 0); c.restore();

  const legend = [
    `<span class="key"><span class="swatch" style="background:${css("--glass-soft")};border-color:${css("--glass")}"></span>Core</span>`,
    `<span class="key"><span class="swatch" style="background:${css("--cladding")}"></span>Cladding</span>`,
  ];
  if (usesPml) legend.push(`<span class="key"><span class="swatch" style="background:repeating-linear-gradient(135deg, ${css("--absorber")} 0 1px, transparent 1px 4px)"></span>Absorbing layer, ${config.pml_layers} cells</span>`);
  else legend.push(`<span class="key">Infinite elements on the outer faces (not drawn)</span>`);
  if (derived.processes > 1) legend.push(`<span class="key"><span class="swatch dashed"></span>Process boundary</span>`);
  legend.push(config.coupling === "Dirichlet" ? `<span class="key"><span class="swatch line"></span>Mode prescribed here</span>` : `<span class="key"><span class="swatch" style="background:${ink};opacity:0.2"></span>Taper region</span>`);
  $("#drawing-legend").innerHTML = legend.join("");
}

// ---- the drawing: computed field ----------------------------------------------------------------------------------

function mix(a, b, t) { return a.map((value, i) => Math.round(value + (b[i] - value) * t)); }
function parseColor(hex) { return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)); }

function fieldValue(field, quantity, j, i) {
  const f = field.field;
  if (quantity === "abs") {
    let sum = 0;
    for (const name of ["re_x", "im_x", "re_y", "im_y", "re_z", "im_z"]) { const v = f[name][j][i]; if (v == null) return null; sum += v * v; }
    return Math.sqrt(sum);
  }
  return f[quantity][j][i];
}

let fieldLayout = null;
function drawField() {
  const field = state.field, quantity = $("#quantity").value;
  const usePhysical = $("#physical").checked;
  const { context: c, width, height } = setupCanvas($("#drawing"));
  const u = field.u, v = field.v, nu = u.length, nv = v.length;
  // Horizontal axis: z for the side and top view, x for the cross-section. Vertical: y (or x in the top view).
  const horizontalIsSecond = field.plane !== "z";
  const hAxis = horizontalIsSecond ? v : u, vAxis = horizontalIsSecond ? u : v;
  const cell = (axis, k) => {
    const lower = k === 0 ? axis[0] - (axis[1] - axis[0]) / 2 : (axis[k - 1] + axis[k]) / 2;
    const upper = k === axis.length - 1 ? axis[k] + (axis[k] - axis[k - 1]) / 2 : (axis[k] + axis[k + 1]) / 2;
    return [lower, upper];
  };
  let shiftMax = 0, shiftMin = 0;
  const shift = (j, i) => {
    if (!usePhysical || field.plane !== "x") return 0;      // the shape displaces y, visible in the side view
    return field.physical_u[j][i] - u[i];
  };
  if (usePhysical && field.plane === "x") for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) { const s = shift(j, i); if (s > shiftMax) shiftMax = s; if (s < shiftMin) shiftMin = s; }
  const h0 = cell(hAxis, 0)[0], h1 = cell(hAxis, hAxis.length - 1)[1];
  const v0 = cell(vAxis, 0)[0] + shiftMin, v1 = cell(vAxis, vAxis.length - 1)[1] + shiftMax;
  const margin = { left: 52, right: 20, top: 26, bottom: 40 };
  let scaleH = (width - margin.left - margin.right) / (h1 - h0), scaleV = (height - margin.top - margin.bottom) / (v1 - v0);
  // To scale if possible. A very long waveguide would become a thin line, so its height is stretched then.
  const available = height - margin.top - margin.bottom;
  if (scaleH > scaleV) scaleH = scaleV;
  else scaleV = Math.max(scaleH, Math.min(scaleV, 140 / (v1 - v0)));
  const stretch = scaleV / scaleH;
  const offsetH = margin.left + (width - margin.left - margin.right - (h1 - h0) * scaleH) / 2;
  const offsetV = margin.top + (height - margin.top - margin.bottom - (v1 - v0) * scaleV) / 2;
  const H = (value) => offsetH + (value - h0) * scaleH;
  const V = (value) => offsetV + (v1 - value) * scaleV;

  let maximum = 0;
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) { const value = fieldValue(field, quantity, j, i); if (value != null && Math.abs(value) > maximum) maximum = Math.abs(value); }
  const cold = parseColor(css("--diverging-cold")), mid = parseColor(css("--diverging-mid")), warm = parseColor(css("--diverging-warm"));
  const isMagnitude = quantity === "abs";
  const color = (value) => {
    const t = maximum > 0 ? value / maximum : 0;
    const rgb = isMagnitude ? mix(mid, cold, Math.min(1, Math.abs(t))) : t < 0 ? mix(mid, cold, Math.min(1, -t)) : mix(mid, warm, Math.min(1, t));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  };
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const value = fieldValue(field, quantity, j, i);
      if (value == null) continue;
      const [ha, hb] = cell(hAxis, horizontalIsSecond ? j : i), [va, vb] = cell(vAxis, horizontalIsSecond ? i : j);
      const s = shift(j, i);
      c.fillStyle = color(value);
      c.fillRect(Math.floor(H(ha)), Math.floor(V(vb + s)), Math.ceil((hb - ha) * scaleH) + 1, Math.ceil((vb - va) * scaleV) + 1);
    }
  }
  // outline of the core: draw the edges between core and cladding cells
  c.strokeStyle = css("--ink"); c.lineWidth = 1; c.globalAlpha = 0.75;
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    if (field.core[j][i] !== 1) continue;
    const [ha, hb] = cell(hAxis, horizontalIsSecond ? j : i), [va, vb] = cell(vAxis, horizontalIsSecond ? i : j);
    const s = shift(j, i);
    const outside = (jj, ii) => jj < 0 || ii < 0 || jj >= nv || ii >= nu || field.core[jj][ii] !== 1;
    const edges = horizontalIsSecond
      ? [[outside(j, i - 1), ha, va, hb, va], [outside(j, i + 1), ha, vb, hb, vb], [outside(j - 1, i), ha, va, ha, vb], [outside(j + 1, i), hb, va, hb, vb]]
      : [[outside(j - 1, i), ha, va, hb, va], [outside(j + 1, i), ha, vb, hb, vb], [outside(j, i - 1), ha, va, ha, vb], [outside(j, i + 1), hb, va, hb, vb]];
    for (const [draw, a1, b1, a2, b2] of edges) if (draw) { c.beginPath(); c.moveTo(H(a1), V(b1 + s)); c.lineTo(H(a2), V(b2 + s)); c.stroke(); }
  }
  c.globalAlpha = 1;
  // axes
  const names = { x: "x", y: "y", z: "z" };
  const hName = horizontalIsSecond ? field.axes[1] : field.axes[0], vName = horizontalIsSecond ? field.axes[0] : field.axes[1];
  c.fillStyle = css("--ink-2"); c.textAlign = "center"; c.textBaseline = "alphabetic";
  const ticks = (a, b) => { const span = b - a, step = Math.pow(10, Math.floor(Math.log10(span / 4))); const nice = [1, 2, 5, 10].map((m) => m * step).find((s) => span / s <= 7); const list = []; for (let t = Math.ceil(a / nice) * nice; t <= b + 1e-9; t += nice) list.push(Math.round(t * 1e6) / 1e6); return list; };
  for (const t of ticks(h0, h1)) c.fillText(String(t), H(t), V(v0) + 16);
  c.fillText(`${names[hName]} in µm`, (H(h0) + H(h1)) / 2, height - 6);
  c.textAlign = "right"; c.textBaseline = "middle";
  for (const t of ticks(v0, v1)) c.fillText(String(t), H(h0) - 6, V(t));
  c.save(); c.translate(12, (V(v0) + V(v1)) / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillText(`${names[vName]} in µm`, 0, 0); c.restore();

  fieldLayout = { H, V, h0, h1, v0, v1, hAxis, vAxis, horizontalIsSecond, shift, quantity, maximum, scaleH, scaleV };
  const label = $("#quantity").selectedOptions[0].textContent;
  const gradient = isMagnitude ? `linear-gradient(90deg, ${css("--diverging-mid")}, ${css("--diverging-cold")})` : `linear-gradient(90deg, ${css("--diverging-cold")}, ${css("--diverging-mid")}, ${css("--diverging-warm")})`;
  const low = isMagnitude ? "0" : "−" + maximum.toPrecision(3);
  $("#drawing-legend").innerHTML = `<span class="key">${label}</span><span class="key">${low} <span class="colorbar" style="background:${gradient}"></span> ${maximum.toPrecision(3)}</span>` +
    `<span class="key"><span class="swatch line" style="border-top-width:1px"></span>Core</span>` +
    (stretch > 1.05 ? `<span class="key">Height stretched ${stretch.toFixed(1)} times</span>` : "") +
    `<span class="key">Values at cell centers, mode amplitude normalized to 1</span>`;
}

function onFieldHover(event) {
  const tooltip = $("#tooltip");
  if (state.showSetup || !state.field || !fieldLayout) { tooltip.hidden = true; return; }
  const rect = $("#drawing").getBoundingClientRect();
  const px = event.clientX - rect.left, py = event.clientY - rect.top;
  const L = fieldLayout, field = state.field;
  const hValue = L.h0 + (px - L.H(L.h0)) / L.scaleH;
  const nearest = (axis, value) => { let best = 0; for (let k = 1; k < axis.length; k++) if (Math.abs(axis[k] - value) < Math.abs(axis[best] - value)) best = k; return best; };
  const hIndex = nearest(L.hAxis, hValue);
  // the vertical position depends on the displacement of the column
  let best = null;
  for (let k = 0; k < L.vAxis.length; k++) {
    const j = L.horizontalIsSecond ? hIndex : k, i = L.horizontalIsSecond ? k : hIndex;
    const y = L.V(L.vAxis[k] + L.shift(j, i));
    if (best === null || Math.abs(y - py) < best.distance) best = { distance: Math.abs(y - py), j, i };
  }
  if (hValue < L.h0 || hValue > L.h1 || !best || best.distance > 30) { tooltip.hidden = true; return; }
  const value = fieldValue(field, L.quantity, best.j, best.i);
  if (value == null) { tooltip.hidden = true; return; }
  const u = field.physical_u[best.j][best.i], v = field.physical_v[best.j][best.i];
  tooltip.innerHTML = `${$("#quantity").selectedOptions[0].textContent} = ${value.toPrecision(4)}<br>${field.axes[0]} = ${u.toFixed(2)}, ${field.axes[1]} = ${v.toFixed(2)} µm`;
  tooltip.hidden = false;
  tooltip.style.left = Math.min(px + 14, rect.width - 190) + "px";
  tooltip.style.top = Math.max(py - 44, 4) + "px";
}

// ---- convergence chart --------------------------------------------------------------------------------------------

const levelNames = ["Whole domain, sweep along z", "Blocks in z, sweep along y", "Blocks in y, sweep along x"];
let chartLayout = null;

function chartSeries(run) {
  // One entry per solver level: all residual histories of that level.
  if (run.histories && run.histories.length) return run.histories;
  if (run.progress) return run.progress.levels.map((level) => level.histories);
  return [];
}

function drawChart() {
  const run = state.run;
  const { context: c, width, height } = setupCanvas($("#chart"));
  $("#chart-tooltip").hidden = true;
  chartLayout = null;
  if (!run) return;
  const levels = chartSeries(run);
  const all = levels.flat().filter((history) => history.length > 0);
  const legend = $("#chart-legend");
  if (all.length === 0) {
    c.fillStyle = css("--ink-2"); c.textAlign = "center";
    c.fillText(run.derived.sweeping_level === 0 ? "One process: solved directly, there is no iteration." : "The residuals appear here once the iteration starts.", width / 2, height / 2);
    legend.innerHTML = "";
    return;
  }
  const margin = { left: 58, right: 16, top: 14, bottom: 38 };
  const maxSteps = Math.max(2, ...all.map((history) => history.length));
  const positive = all.flat().filter((value) => value > 0);
  let low = Math.floor(Math.log10(Math.min(...positive))), high = Math.ceil(Math.log10(Math.max(...positive)));
  if (high - low < 2) low = high - 2;
  const X = (step) => margin.left + (step - 1) / (maxSteps - 1) * (width - margin.left - margin.right);
  const Y = (value) => margin.top + (high - Math.log10(Math.max(value, 1e-300))) / (high - low) * (height - margin.top - margin.bottom);
  // grid and axes (recessive)
  c.lineWidth = 1; c.strokeStyle = css("--grid"); c.fillStyle = css("--ink-2"); c.textAlign = "right"; c.textBaseline = "middle";
  const decadeStep = Math.max(1, Math.ceil((high - low) / 7));
  for (let exponent = low; exponent <= high; exponent += decadeStep) {
    const y = Math.round(Y(Math.pow(10, exponent))) + 0.5;
    c.beginPath(); c.moveTo(margin.left, y); c.lineTo(width - margin.right, y); c.stroke();
    c.fillText("1e" + exponent, margin.left - 8, y);
  }
  c.textAlign = "center"; c.textBaseline = "alphabetic";
  const stepTick = Math.max(1, Math.ceil(maxSteps / 10));
  for (let step = 1; step <= maxSteps; step += stepTick) c.fillText(String(step), X(step), height - margin.bottom + 16);
  c.fillText("GMRES step", (margin.left + width - margin.right) / 2, height - 6);
  c.save(); c.translate(14, (margin.top + height - margin.bottom) / 2); c.rotate(-Math.PI / 2); c.fillText("Residual", 0, 0); c.restore();
  // lines: the inner levels first (thin), the outermost level on top
  const colors = [css("--series-1"), css("--series-2"), css("--series-3")];
  for (let level = levels.length - 1; level >= 0; level--) {
    const histories = levels[level];
    c.strokeStyle = colors[level]; c.fillStyle = colors[level];
    histories.forEach((history) => {
      if (history.length === 0) return;
      c.lineWidth = level === 0 ? 2 : 1; c.globalAlpha = level === 0 ? 1 : Math.max(0.25, Math.min(0.8, 6 / histories.length));
      c.beginPath(); history.forEach((value, k) => { const x = X(k + 1), y = Y(value); if (k === 0) c.moveTo(x, y); else c.lineTo(x, y); }); c.stroke();
      if (level === 0) history.forEach((value, k) => { c.beginPath(); c.arc(X(k + 1), Y(value), 3.5, 0, 2 * Math.PI); c.fill(); });
    });
  }
  c.globalAlpha = 1;
  chartLayout = { X, Y, levels, maxSteps, margin, width, height };
  legend.innerHTML = levels.map((histories, level) =>
    `<span class="key"><span class="swatch" style="background:${colors[level]}"></span>${levelNames[level] || "Level " + (level + 1)}: ${histories.length} solve${histories.length === 1 ? "" : "s"}${run.status === "running" || !run.histories ? " shown" : " on process 0"}</span>`).join("");
}

function onChartHover(event) {
  const tooltip = $("#chart-tooltip");
  if (!chartLayout) { tooltip.hidden = true; return; }
  const rect = $("#chart").getBoundingClientRect();
  const px = event.clientX - rect.left, py = event.clientY - rect.top;
  const L = chartLayout;
  let best = null;
  L.levels.forEach((histories, level) => histories.forEach((history, index) => history.forEach((value, k) => {
    const distance = Math.hypot(L.X(k + 1) - px, L.Y(value) - py);
    if (best === null || distance < best.distance) best = { distance, level, index, step: k + 1, value };
  })));
  if (!best || best.distance > 24) { tooltip.hidden = true; return; }
  tooltip.innerHTML = `${levelNames[best.level] || "Level " + (best.level + 1)}<br>Step ${best.step}: residual ${best.value.toExponential(2)}`;
  tooltip.hidden = false;
  tooltip.style.left = Math.min(px + 14, rect.width - 230) + "px";
  tooltip.style.top = Math.max(py - 46, 4) + "px";
}

// ---- runs -----------------------------------------------------------------------------------------------------------

async function refreshRunList(selectId) {
  const runs = await api("/api/runs");
  const select = $("#run-select");
  const current = selectId ?? select.value;
  select.innerHTML = '<option value="">New configuration</option>';
  for (const run of runs) {
    const option = document.createElement("option");
    const when = new Date(run.started * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    const shape = { straight: "straight", hump: "hump " + run.config.hump_height + " µm", predefined: "shape " + run.config.predefined_case }[run.config.shape];
    const outcome = run.status === "finished" && run.signal_strength != null ? "signal " + run.signal_strength.toFixed(3) : run.status;
    option.value = run.id;
    option.textContent = `${when}: ${run.config.length} µm ${shape}, ${run.derived.processes} proc., ${outcome}`;
    select.appendChild(option);
  }
  select.value = [...select.options].some((option) => option.value === current) ? current : "";
}

function stopPolling() { clearInterval(state.poll); state.poll = null; }

async function loadRun(id) {
  stopPolling();
  history.replaceState(null, "", id ? "?run=" + id : location.pathname);
  if (!id) { state.run = null; state.field = null; state.showSetup = true; showRun(null); return refreshEstimate(); }
  const run = await api(`/api/runs/${id}`);
  state.run = run; state.field = null; state.showSetup = true;
  state.config = run.config;
  writeForm(run.config);
  showRun(run);
  if (run.status === "running") state.poll = setInterval(pollRun, 1000);
  if (run.status === "finished") await loadField();
}

async function pollRun() {
  if (!state.run) return stopPolling();
  try {
    const run = await api(`/api/runs/${state.run.id}`);
    state.run = run;
    showRun(run);
    if (run.status !== "running") {
      stopPolling();
      await refreshRunList(run.id);
      if (run.status === "finished") await loadField();
    }
  } catch (error) { stopPolling(); }
}

async function loadField() {
  const run = state.run;
  if (!run || !run.planes || run.planes.length === 0) { $("#view-controls").hidden = true; return; }
  let plane = (document.querySelector('[name="plane"]:checked') || {}).value || "x";
  if (!run.planes.includes(plane)) plane = run.planes[0];
  try {
    state.field = await api(`/api/runs/${run.id}/field?plane=${plane}`);
    state.showSetup = false;
    $("#view-controls").hidden = false;
    $("#physical").parentElement.hidden = !(run.config.shape !== "straight" && plane === "x");
    $("#show-setup").textContent = "Show setup drawing";
    drawStage();
  } catch (error) { state.field = null; state.showSetup = true; drawStage(); }
}

const numberRows = [
  ["signal_strength_abs", "Signal at the output", (v) => v.toFixed(4), true],
  ["solver_iterations", "GMRES steps, outer solver", (v) => String(Math.round(v))],
  ["n_dofs", "Unknowns", (v) => Math.round(v).toLocaleString()],
  ["n_local_solves", "Subdomain solves", (v) => Math.round(v).toLocaleString()],
  ["local_factorization_time_max", "Factorization of a subdomain", (v) => v.toFixed(1) + " s"],
  ["local_time_per_solve", "One subdomain solve", (v) => (v * 1000).toFixed(0) + " ms"],
  ["local_factor_memory_mb_max", "Memory of one subdomain solver", (v) => v >= 1000 ? (v / 1000).toFixed(1) + " GB" : Math.round(v) + " MB"],
  ["compression", "Factor entries stored", null],
  ["error_l2", "Difference to the straight mode, L2", (v) => v.toPrecision(3)],
  ["error_l2_relative", "Relative error, L2", (v) => v.toPrecision(3)],
];

function showRun(run) {
  const status = $("#run-status");
  $("#delete-run").hidden = !run || run.status === "running";
  $("#cancel").hidden = !(run && run.status === "running");
  $("#run").textContent = run ? "Run again" : "Run solver";
  $("#results").hidden = !run;
  if (!run) {
    status.className = "run-status"; status.textContent = "";
    $("#view-controls").hidden = true;
    renderEstimate(); renderHints(); drawStage();
    return;
  }
  const words = { running: (run.progress ? run.progress.phase : "Starting") + ", " + formatDuration(run.elapsed), finished: "Finished in " + formatDuration(run.elapsed),
                  failed: "Failed after " + formatDuration(run.elapsed), cancelled: "Stopped", interrupted: "Interrupted (the frontend was restarted)" };
  status.className = "run-status " + (run.status === "cancelled" || run.status === "interrupted" ? "failed" : run.status);
  status.innerHTML = `<span class="dot"></span>${words[run.status]}`;
  if (run.status !== "finished") $("#view-controls").hidden = true;
  const problems = [...run.derived.warnings];
  if (run.status === "failed" && run.error) problems.unshift(run.error);
  showProblems(problems, false);
  if (run.status === "failed") $("#warnings").firstChild && ($("#warnings").firstChild.className = "error");
  $("#run").disabled = run.status === "running" || !state.info.solver_found;
  renderEstimate(); renderHints();
  if (state.showSetup || !state.field) drawSetup();

  // numbers
  const results = run.results || {};
  const body = $("#numbers-table tbody");
  body.innerHTML = "";
  if (results.local_factor_entries_full_rank > 0) results.compression = results.local_factor_entries_stored / results.local_factor_entries_full_rank;
  for (const [key, label, format, headline] of numberRows) {
    if (results[key] == null) continue;
    if (key === "compression" && results.compression > 0.999) continue;
    const row = document.createElement("tr");
    if (headline) row.className = "headline";
    const text = key === "compression" ? Math.round(results.compression * 100) + " % of the exact factors" : format(results[key]);
    row.innerHTML = `<td>${label}</td><td>${text}</td>`;
    body.appendChild(row);
  }
  if (body.children.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${run.status === "running" ? "The numbers appear when the solver has finished." : "This run produced no result."}</td><td></td>`;
    body.appendChild(row);
  }
  const level = run.derived.sweeping_level;
  $("#convergence-note").textContent = level === 0 ? "" : run.status === "running"
    ? "Residuals of the GMRES solvers, updated while the solver runs."
    : level === 1 ? "Residual of GMRES for the whole domain. Each step is one sweep over the subdomains."
    : "Residuals of the nested GMRES solvers. The thin lines are the solves inside each sweep of the level above.";
  drawChart();
  api(`/api/runs/${run.id}/log`).then((data) => { $("#log").textContent = data.log; }).catch(() => {});
}

async function startRun() {
  $("#run").disabled = true;
  try {
    const config = state.run ? state.run.config : readForm();
    const run = await api("/api/runs", { method: "POST", body: JSON.stringify(config) });
    await refreshRunList(run.id);
    await loadRun(run.id);
  } catch (error) {
    showProblems([error.message], true);
    $("#run").disabled = false;
  }
}

// ---- start ----------------------------------------------------------------------------------------------------------

async function start() {
  const theme = new URLSearchParams(location.search).get("theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  state.info = await api("/api/info");
  state.config = { ...state.info.defaults };
  const slider = form.querySelector('[name="cells_per_wavelength"]');
  writeForm(state.config);
  slider.value = state.config.cells_per_wavelength;
  form.addEventListener("input", onFormChange);
  $("#run").addEventListener("click", startRun);
  $("#cancel").addEventListener("click", async () => { if (state.run) { await api(`/api/runs/${state.run.id}/cancel`, { method: "POST" }); pollRun(); } });
  $("#run-select").addEventListener("change", (event) => loadRun(event.target.value));
  $("#delete-run").addEventListener("click", async () => {
    if (!state.run) return;
    await api(`/api/runs/${state.run.id}`, { method: "DELETE" });
    await refreshRunList("");
    await loadRun("");
  });
  document.querySelectorAll('[name="plane"]').forEach((input) => input.addEventListener("change", loadField));
  $("#quantity").addEventListener("change", drawStage);
  $("#physical").addEventListener("change", drawStage);
  $("#show-setup").addEventListener("click", () => {
    state.showSetup = !state.showSetup;
    $("#show-setup").textContent = state.showSetup ? "Show computed field" : "Show setup drawing";
    drawStage();
  });
  $("#drawing").addEventListener("mousemove", onFieldHover);
  $("#drawing").addEventListener("mouseleave", () => { $("#tooltip").hidden = true; });
  $("#chart").addEventListener("mousemove", onChartHover);
  $("#chart").addEventListener("mouseleave", () => { $("#chart-tooltip").hidden = true; });
  window.addEventListener("resize", () => { drawStage(); drawChart(); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawStage(); drawChart(); });
  // A run can be opened directly with ?run=<id>, so that results can be bookmarked and shared.
  const requested = new URLSearchParams(location.search).get("run") || "";
  await refreshRunList(requested);
  if ($("#run-select").value) await loadRun($("#run-select").value);
  else await refreshEstimate();
}

start().catch((error) => { document.body.insertAdjacentHTML("beforeend", `<p style="padding:1.5rem">The frontend could not start: ${error.message}</p>`); });
