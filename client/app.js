// ─── Configuration ────────────────────────────────────────────────────────────
// Automatically points back to whatever server served this page.
// Works for localhost (Live Server) and any device on the LAN.
const SERVER_URL = `${window.location.protocol}//${window.location.hostname}:5000`;

// Brush settings
const BRUSH_RADIUS = 18;   // px — thicker feels more natural on touch screens
const BRUSH_COLOR  = "#ffffff";

// ─── Elements ─────────────────────────────────────────────────────────────────
const canvas      = document.getElementById("canvas");
const ctx         = canvas.getContext("2d");
const btnClear    = document.getElementById("btn-clear");
const btnPredict  = document.getElementById("btn-predict");
const autoPredict = document.getElementById("auto-predict");
const bigDigit    = document.getElementById("big-digit");
const confLabel   = document.getElementById("confidence-label");
const barsEl      = document.getElementById("bars");

// ─── Build bar rows (once) ────────────────────────────────────────────────────
const barFills = [];
const barPcts  = [];

for (let i = 0; i < 10; i++) {
  const row   = document.createElement("div");  row.className = "bar-row";
  const label = document.createElement("span"); label.className = "bar-label"; label.textContent = i;
  const track = document.createElement("div");  track.className = "bar-track";
  const fill  = document.createElement("div");  fill.className = "bar-fill";
  const pct   = document.createElement("span"); pct.className = "bar-pct";

  track.appendChild(fill);
  row.append(label, track, pct);
  barsEl.appendChild(row);
  barFills.push(fill);
  barPcts.push(pct);
}

// ─── Canvas drawing ───────────────────────────────────────────────────────────
let isDrawing = false;
let hasStrokes = false;

ctx.fillStyle = "#000";
ctx.fillRect(0, 0, canvas.width, canvas.height);

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * scaleX,
    y: (src.clientY - rect.top)  * scaleY,
  };
}

function drawDot(x, y) {
  ctx.beginPath();
  ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = BRUSH_COLOR;
  ctx.fill();
}

function onStart(e) {
  e.preventDefault();
  isDrawing = true;
  hasStrokes = true;
  const { x, y } = getPos(e);
  drawDot(x, y);          // fill the tap dot first
  ctx.beginPath();        // start a clean path for the drag line
  ctx.moveTo(x, y);       // anchor it at the same point
}

function onMove(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.strokeStyle = BRUSH_COLOR;
  ctx.lineWidth   = BRUSH_RADIUS * 2;
  ctx.lineCap     = "round";
  ctx.lineJoin    = "round";
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function onEnd(e) {
  if (!isDrawing) return;
  isDrawing = false;
  if (autoPredict.checked && hasStrokes) sendImage();
}

// Pointer events work for both mouse and touch/stylus
canvas.addEventListener("pointerdown", onStart);
canvas.addEventListener("pointermove", onMove);
canvas.addEventListener("pointerup",   onEnd);
canvas.addEventListener("pointerleave", onEnd);

// ─── Clear ────────────────────────────────────────────────────────────────────
function clearCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  hasStrokes = false;
  bigDigit.textContent = "—";
  bigDigit.className = "big-digit";
  confLabel.textContent = "";
  barFills.forEach(f => { f.style.width = "0%"; f.className = "bar-fill"; });
  barPcts.forEach(p => { p.textContent = ""; });
  clearActivations();
}

btnClear.addEventListener("click", clearCanvas);

// ─── Predict ──────────────────────────────────────────────────────────────────
btnPredict.addEventListener("click", () => { if (hasStrokes) sendImage(); });

async function sendImage() {
  bigDigit.className = "big-digit pending";
  bigDigit.textContent = "…";
  confLabel.textContent = "";

  // Export canvas as PNG blob
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
  const form = new FormData();
  form.append("image", blob, "digit.png");

  try {
    const res = await fetch(`${SERVER_URL}/predict`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    showResult(data);
  } catch (err) {
    bigDigit.textContent = "!";
    confLabel.textContent = "Could not reach server";
    console.error(err);
  }
}

function showResult(data) {
  const { digit, confidence, scores } = data;

  bigDigit.className = "big-digit";
  bigDigit.textContent = digit;
  confLabel.textContent = `${(confidence * 100).toFixed(1)}% confident`;

  scores.forEach((score, i) => {
    const pct = (score * 100).toFixed(1);
    barFills[i].style.width = `${pct}%`;
    barFills[i].className = "bar-fill" + (i === digit ? " top" : "");
    barPcts[i].textContent = `${pct}%`;
  });

  if (data.activations) renderActivations(data.activations);
}

// ─── Activation visualiser ────────────────────────────────────────────────────
// This renderer consumes the stable activation data contract returned by
// /predict. The same contract will be used by the future UE5 bridge.
//
// Data shape per layer:
//   { name, type, shape, maps: [[row,...],...]  }  ← conv / pool / input
//   { name, type, shape, values: [...]          }  ← fc / output

const LAYER_DELAY_MS = 500;   // animation delay between layers
const MAP_SIZE       = 28;    // px per feature-map tile

const actSection = document.getElementById("act-section");
const actLayers  = document.getElementById("act-layers");

function clearActivations() {
  actSection.hidden = true;
  actLayers.innerHTML = "";
}

async function renderActivations(layers) {
  actLayers.innerHTML = "";
  actSection.hidden = false;

  for (const layer of layers) {
    const block = document.createElement("div");
    block.className = "act-layer-block act-hidden";
    block.dataset.name = layer.name;

    const label = document.createElement("div");
    label.className = "act-layer-label";
    label.textContent = `${layer.name}  [${layer.shape.join(" × ")}]`;
    block.appendChild(label);

    if (layer.maps) {
      // ── Spatial feature maps (input / conv / pool) ──────────────────────
      const grid = document.createElement("div");
      grid.className = "act-map-grid";

      for (let m = 0; m < layer.maps.length; m++) {
        const mapData = layer.maps[m];
        const rows = mapData.length;
        const cols = mapData[0].length;

        const c = document.createElement("canvas");
        c.width  = cols;
        c.height = rows;
        c.className = "act-map-canvas";
        c.title = `${layer.name} map ${m}`;

        const imgData = c.getContext("2d").createImageData(cols, rows);
        for (let r = 0; r < rows; r++) {
          for (let col = 0; col < cols; col++) {
            const val = Math.round(mapData[r][col] * 255);
            const idx = (r * cols + col) * 4;
            imgData.data[idx]     = val;
            imgData.data[idx + 1] = val;
            imgData.data[idx + 2] = val;
            imgData.data[idx + 3] = 255;
          }
        }
        c.getContext("2d").putImageData(imgData, 0, 0);
        grid.appendChild(c);
      }
      block.appendChild(grid);

    } else if (layer.values) {
      // ── Flat FC / output layer ───────────────────────────────────────────
      const bars = document.createElement("div");
      bars.className = "act-fc-bars";

      layer.values.forEach((v, i) => {
        const row = document.createElement("div");
        row.className = "act-fc-row";

        const lbl = document.createElement("span");
        lbl.className = "act-fc-label";
        lbl.textContent = layer.name === "output" ? i : i;

        const track = document.createElement("div");
        track.className = "act-fc-track";
        const fill = document.createElement("div");
        fill.className = "act-fc-fill";
        fill.style.width = `${(v * 100).toFixed(1)}%`;
        track.appendChild(fill);

        row.append(lbl, track);
        bars.appendChild(row);
      });
      block.appendChild(bars);
    }

    actLayers.appendChild(block);

    // Staggered reveal — each layer fades in after LAYER_DELAY_MS
    await new Promise(r => setTimeout(r, LAYER_DELAY_MS));
    block.classList.remove("act-hidden");
    block.classList.add("act-visible");
  }
}
