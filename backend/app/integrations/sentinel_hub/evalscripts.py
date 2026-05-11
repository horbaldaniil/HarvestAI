"""Sentinel Hub evalscripts.

Two kinds of scripts:
- INDICES_EVALSCRIPT — emits FLOAT32 outputs per index for Statistical API,
  plus a `dataMask` band that lets the API report cloud-free pixel fraction.
- HEATMAP_EVALSCRIPTS — per-index colour-mapped PNG for visual overlay
  via Process API.

Sentinel-2 L2A bands used:
  B02 = Blue, B03 = Green, B04 = Red, B08 = NIR

Cloud handling:
  Done by the request's `dataFilter.maxCloudCoverage = 30` (scene-level
  filter; SH discards scenes with >30% cloud cover before our script ever
  sees them) and `dataMask` (per-pixel data-availability flag). Adding a
  SCL-based fine-grained cloud mask required two input groups with
  different `units` (REFLECTANCE for B0x, DN for SCL) — that path proved
  brittle in our environment (rejected by SH or zeroing out everything),
  so we keep the canonical single-input approach. A SCL refinement is
  documented as a master's-thesis extension.

NDVI = (NIR - Red) / (NIR + Red)
EVI  = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
NDWI = (Green - NIR) / (Green + NIR)
SAVI = (NIR - Red) * 1.5 / (NIR + Red + 0.5)
"""

INDICES_EVALSCRIPT: str = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "dataMask"]
    }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "evi",  bands: 1, sampleType: "FLOAT32" },
      { id: "ndwi", bands: 1, sampleType: "FLOAT32" },
      { id: "savi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

// Clamp helper — keeps values inside what our NUMERIC(5,3) column can store
// (|x| < 100). EVI's formula has an unstable denominator on water/shadow
// pixels and can spike into the billions; without clamping those poison
// the bucket aggregate and overflow Postgres.
function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

function evaluatePixel(s) {
  if (s.dataMask !== 1) {
    return {
      ndvi: [NaN], evi: [NaN], ndwi: [NaN], savi: [NaN],
      dataMask: [0]
    };
  }
  const nir = s.B08;
  const red = s.B04;
  const blue = s.B02;
  const green = s.B03;

  const denomR = nir + red;
  const denomW = green + nir;
  const eviDenom = nir + 6.0 * red - 7.5 * blue + 1.0;

  // Guard the denominators. EVI's denominator can become very small or
  // negative on water/dark pixels — return NaN there so the pixel doesn't
  // skew the aggregate.
  const ndvi = denomR > 0 ? (nir - red) / denomR : NaN;
  const evi  = Math.abs(eviDenom) > 0.05 ? clamp(2.5 * (nir - red) / eviDenom, -2, 2) : NaN;
  const ndwi = denomW > 0 ? (green - nir) / denomW : NaN;
  const savi = clamp((nir - red) * 1.5 / (nir + red + 0.5), -2, 2);

  return {
    ndvi: [clamp(ndvi, -1, 1)],
    evi:  [evi],
    ndwi: [clamp(ndwi, -1, 1)],
    savi: [savi],
    dataMask: [1]
  };
}
"""


def _heatmap_evalscript(formula: str) -> str:
    """Build a heatmap RGB evalscript for one index.

    `formula` is a JS expression in scope where s = sample. Output is RGBA
    suitable for L.ImageOverlay (transparent where masked).
    """
    return f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: ["B02", "B03", "B04", "B08", "dataMask"]
    }}],
    output: {{ bands: 4, sampleType: "UINT8" }}
  }};
}}

function ramp(v) {{
  v = Math.max(0, Math.min(1, v));
  const stops = [
    [0.0, [165,  15,  21]],
    [0.2, [222,  45,  38]],
    [0.4, [251, 154,  41]],
    [0.6, [255, 255, 102]],
    [0.8, [173, 221, 142]],
    [1.0, [ 49, 163,  84]]
  ];
  for (let i = 0; i < stops.length - 1; i++) {{
    const [a, ca] = stops[i];
    const [b, cb] = stops[i + 1];
    if (v <= b) {{
      const t = (v - a) / (b - a);
      return [
        Math.round(ca[0] + t * (cb[0] - ca[0])),
        Math.round(ca[1] + t * (cb[1] - ca[1])),
        Math.round(ca[2] + t * (cb[2] - ca[2]))
      ];
    }}
  }}
  return stops[stops.length - 1][1];
}}

function evaluatePixel(s) {{
  if (s.dataMask !== 1) {{
    return [0, 0, 0, 0];
  }}
  const nir = s.B08, red = s.B04, blue = s.B02, green = s.B03;
  const raw = {formula};
  // Normalise to [0..1] for the ramp: NDVI/NDWI in [-1,1], EVI/SAVI usually [-1,1].
  const v = (raw + 1.0) / 2.0;
  const c = ramp(v);
  return [c[0], c[1], c[2], 220];
}}
"""


HEATMAP_EVALSCRIPTS: dict[str, str] = {
    "ndvi": _heatmap_evalscript("(nir - red) / (nir + red + 1e-9)"),
    "evi": _heatmap_evalscript(
        "2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)"
    ),
    "ndwi": _heatmap_evalscript("(green - nir) / (green + nir + 1e-9)"),
    "savi": _heatmap_evalscript("(nir - red) * 1.5 / (nir + red + 0.5)"),
}


SUPPORTED_INDICES: tuple[str, ...] = ("ndvi", "evi", "ndwi", "savi")
