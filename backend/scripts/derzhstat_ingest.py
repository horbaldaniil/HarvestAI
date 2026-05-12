"""Ingest Держстат monthly statistical bulletin "Обсяги виробництва
продукції сільського господарства" (filename `ovuzpsg_MMYY.xls`) into the
canonical `per_oblast_yield` section of `derzhstat_yield_2017_2023.yaml`
(keeps the filename — file now covers 2017-2025 effective).

## Source format

Files come from Держстат: https://www.ukrstat.gov.ua/
→ Статистична інформація → Сільське, лісове та рибне господарство
→ Експрес-випуски → "Обсяги виробництва продукції сільського
господарства".

Filenames encode `MMYY`:
  ovuzpsg_1118.xls = November 2018 (final-year data)
  ovuzpsg_0925.xls = September 2025 (partial-year — only early-harvest crops)

Each XLS has per-crop sheets named e.g. "6 пшен" (wheat), "12 кукур" (corn).
**Sheet POSITION shifts between months**: July files have fewer crops →
wheat is in sheet "3 пшен"; November files have full set → "6 пшен".
This script matches by **stable Ukrainian abbreviation** regex, NOT by
sheet position.

Cumulative data: each subsequent month within a year contains the
previous month's data plus new crops harvested since then. The script
auto-picks the **latest month per year** for ingest (i.e. November or
December gives final-year yields; September would be partial).

Each crop sheet has the structure:
  row 0: title
  row 1-2: column headers
  row 3:   Україна (national totals)
  rows 4-27: 24 oblasts
    col 0: blank
    col 1: oblast name (Ukrainian)
    col 4: **YIELD ц/га** (all-categories, what we extract)
    col 11: oblast name (English)

Yields are in **центнерів з гектара (ц/га)** — divide by 10 → t/ha.

## Manual download steps

1. https://www.ukrstat.gov.ua/operativ/operativ_YYYY/sg/svm/ — different
   subfolder per year. Look for `ovuzpsg_*.xls` links.
2. Latest-month per year:
   - 2018-2020: November file (no December published)
   - 2021-2023: December file (full-year final)
   - 2025: October file (partial — corner-cases flagged `is_partial_year`)
3. Put files in `backend/data/raw/derzhstat_raw/` and run
   `uv run python scripts/derzhstat_ingest.py`.

If no files are present the script no-ops gracefully —
`build_yield_csv.py` falls back to weather-conditioned synthesis.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402
from app.data_reference.oblast_names import iso_from_any_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("derzhstat_ingest")

INPUT_DIR = ROOT / "data" / "raw" / "derzhstat_raw"
YAML_PATH = ROOT / "data" / "raw" / "derzhstat_yield_2017_2023.yaml"


# Keyword regexes — match Ukrainian crop abbreviations stable across
# Держстат bulletins regardless of sheet position prefix. Ordering
# matters: more-specific regexes go BEFORE more-general ones, so e.g.
# "кукур корм" (corn silage) wins before "кукур" (corn grain).
CROP_KEYWORD_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # ─── Specific cases first (qualifying word) ─────
    (re.compile(r"\bкукур\s+корм\b"), "corn_silage"),
    (re.compile(r"\bбуряк\s+цукр\b"), "sugar_beet"),
    # ─── General — `пшен`, `ячм`, `жито`, `ріпак` are aliased to combined
    # forms. We skip pure-винторий/ярий variants (e.g. "пшенОЗ", "ячмЯР")
    # because they double-count when "пшен" already covers all wheat.
    (re.compile(r"\bпшен\b"), "wheat"),
    (re.compile(r"\bкукур\b"), "corn"),
    (re.compile(r"\bячм\b"), "barley"),
    (re.compile(r"\bжито\b"), "rye"),                   # combined rye sheet, if present
    (re.compile(r"\bжитоОЗ\b"), "rye"),                 # winter rye = dominant in UA
    (re.compile(r"\bовес\b"), "oats"),
    (re.compile(r"\bгречка\b"), "buckwheat"),
    (re.compile(r"\bгорох\b"), "peas"),
    (re.compile(r"\bсоя\b"), "soybean"),
    (re.compile(r"\bріпак\b"), "rapeseed"),
    # Some years bulletin spells "соняшн" (with `н`), others "соняш" without.
    # No trailing word-boundary so we match both.
    (re.compile(r"\bсоняш"), "sunflower"),
    (re.compile(r"\bкарт\b"), "potato"),
)

# Crops harvested mostly AFTER October — these get `is_partial_year=True`
# when ingested from a pre-November bulletin. Downstream evaluator may
# exclude or de-weight them for the affected (crop, year).
LATE_HARVEST_CROPS: frozenset[str] = frozenset({
    "sugar_beet",       # Sep-Nov harvest
    "potato",           # Sep-Oct main, some Nov
    "corn",             # Sep-Nov for grain (silage Aug-Sep)
})

# Column index for "all categories" yield in each crop sheet (col 4).
YIELD_COL_INDEX = 4
OBLAST_NAME_COL_INDEX = 1

# Oblast rows: 4 through 27 inclusive (Україна in row 3, then 24 oblasts).
OBLAST_ROW_RANGE = (4, 28)


def _match_crop(sheet_name: str) -> str | None:
    """Return crop slug or None for a Держстат sheet name.

    Examples (works for both July-bulletin and November-bulletin sheets):
      "6 пшен"      → wheat
      "3 пшен"      → wheat
      "12 кукур"    → corn
      "50 кукур корм" → corn_silage  (matched first by more-specific regex)
      "43 буряк цукр" → sugar_beet
      "20 житоОЗ"   → rye           (winter rye is the ONLY commercial rye in UA)
      "5 пшенОЗ"    → None          (winter-wheat-only — skipped, "пшен" handles combined)
    """
    if not sheet_name:
        return None
    # Reject pure-winter / pure-spring sheets for crops where BOTH varieties
    # exist (we use the combined sheet instead). Rye is excluded from this
    # rejection because in Ukraine spring rye is negligible — winter rye
    # ("житоОЗ") IS the crop.
    if re.search(r"\b(пшен|ячм|ріпак)(ОЗ|ЯР)\b", sheet_name):
        return None
    for pat, crop in CROP_KEYWORD_PATTERNS:
        if pat.search(sheet_name):
            return crop
    return None


def parse_month_year(stem: str) -> tuple[int, int] | None:
    """Parse `ovuzpsg_MMYY` filename stem → (month, year) or None.

    Examples:
      "ovuzpsg_1118" → (11, 2018)
      "ovuzpsg_1221" → (12, 2021)
      "ovuzpsg_0725" → (07, 2025)
    """
    m = re.search(r"ovuzpsg_(\d{2})(\d{2})", stem.lower())
    if not m:
        return None
    month, yy = int(m.group(1)), int(m.group(2))
    # Naïve century rollover: 17-30 → 20YY; 50-99 → 19YY (historical, none expected).
    year = 2000 + yy if yy < 50 else 1900 + yy
    return month, year


def pick_latest_per_year(files: list[Path]) -> dict[int, Path]:
    """Group files by year, return only the rightmost month per year.

    Дerzhstat monthly bulletins are CUMULATIVE — November contains
    everything from earlier months plus more crops harvested since.
    So the November file alone is sufficient (December is +5% only).
    Caller iterates the returned dict.
    """
    by_year: dict[int, list[tuple[int, Path]]] = {}
    for f in files:
        parsed = parse_month_year(f.stem)
        if parsed is None:
            log.warning("Could not parse month/year from %s — skipping.", f.name)
            continue
        month, year = parsed
        by_year.setdefault(year, []).append((month, f))
    return {year: max(items, key=lambda x: x[0])[1] for year, items in by_year.items()}


def parse_xls(path: Path, partial_year: bool = False) -> dict[str, dict[str, float]]:
    """Parse one Держstat bulletin → `{crop_slug: {iso: yield_tha}}`.

    Walks every sheet, applies `_match_crop()` to identify which crop's
    table it is, extracts per-oblast yields from column 4 (ц/га).
    Converts to t/ha (÷ 10). Skips footnote rows.

    Args:
        path: XLS file path.
        partial_year: when True, the file is from a pre-November bulletin
            (e.g. October 2025) and may not contain final yields for
            late-harvest crops. Caller logs a warning but proceeds; the
            `is_partial_year` flag is set on emitted records separately.

    Returns empty dict on parse failure.
    """
    try:
        import xlrd
    except ImportError:
        log.error("xlrd required: uv pip install xlrd")
        return {}

    try:
        wb = xlrd.open_workbook(str(path))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not open %s: %s", path.name, exc)
        return {}

    out: dict[str, dict[str, float]] = {}

    for sheet_name in wb.sheet_names():
        crop_slug = _match_crop(sheet_name)
        if crop_slug is None:
            continue
        if crop_slug in out:
            # First match wins per crop — skip duplicates (e.g. when the
            # bulletin has both "пшен" combined AND specific "пшенОЗ").
            continue
        if partial_year and crop_slug in LATE_HARVEST_CROPS:
            log.info("  %s — flagged is_partial_year (late-harvest, pre-Nov bulletin)",
                     crop_slug)

        sh = wb.sheet_by_name(sheet_name)
        crop_yields: dict[str, float] = {}
        for r in range(*OBLAST_ROW_RANGE):
            if r >= sh.nrows:
                break
            ob_name = sh.cell_value(r, OBLAST_NAME_COL_INDEX)
            if not ob_name or not str(ob_name).strip():
                continue
            iso = iso_from_any_name(str(ob_name))
            if not iso:
                continue
            cell = sh.cell_value(r, YIELD_COL_INDEX)
            try:
                centners = float(cell)
            except (TypeError, ValueError):
                continue
            if centners <= 0:
                continue
            crop_yields[iso] = round(centners / 10.0, 2)
        if crop_yields:
            out[crop_slug] = crop_yields
            log.info("  %s (sheet %r) — %d oblasts",
                     crop_slug, sheet_name, len(crop_yields))

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                        help="folder with ovuzpsg_*.xls files")
    parser.add_argument("--yaml", type=Path, default=YAML_PATH,
                        help="canonical yield YAML to update")
    parser.add_argument("--extra-files", nargs="*", type=Path, default=[],
                        help="additional XLS files outside input-dir to ingest")
    parser.add_argument("--partial-ok", action="store_true", default=True,
                        help="accept pre-November bulletins for partial-year ingest "
                             "(default true; set --no-partial-ok to require Nov+)")
    parser.add_argument("--no-partial-ok", dest="partial_ok", action="store_false")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse + report stats; do NOT update YAML")
    args = parser.parse_args()

    files: list[Path] = []
    if args.input_dir.exists():
        files.extend(sorted(args.input_dir.glob("ovuzpsg_*.xls")))
        files.extend(sorted(args.input_dir.glob("ovuzpsg_*.xlsx")))
    for f in args.extra_files:
        if f.is_dir():
            files.extend(sorted(f.glob("ovuzpsg_*.xls")))
            files.extend(sorted(f.glob("ovuzpsg_*.xlsx")))
        elif f.exists():
            files.append(f)
        else:
            log.warning("Extra path %s does not exist; skipping.", f)

    if not files:
        log.info("No Держстат files found. Skipping ingest — build_yield_csv.py "
                 "will use weather-conditioned synthesis.")
        return 0

    log.info("Found %d Держстат file(s).", len(files))

    # Pick latest month per year (Nov/Dec dominant; Oct for partial 2025).
    latest = pick_latest_per_year(files)
    log.info("Latest-month-per-year selected:")
    for year, fp in sorted(latest.items()):
        parsed = parse_month_year(fp.stem)
        month_str = f"{parsed[0]:02d}" if parsed else "??"
        partial_flag = month_str < "11"
        partial_marker = " (PARTIAL — pre-November)" if partial_flag else ""
        log.info("  %d → %s [month %s]%s", year, fp.name, month_str, partial_marker)

    # per_oblast_yield: {crop: {year: {iso: yield_tha}}}
    per_oblast: dict[str, dict[int, dict[str, float]]] = {c: {} for c in ALL_CROPS}
    # partial_year_flag: {crop: {year: bool}} — emitted as part of YAML metadata.
    partial_flag: dict[str, dict[int, bool]] = {c: {} for c in ALL_CROPS}
    n_rows = 0
    for year, xls in sorted(latest.items()):
        month, _ = parse_month_year(xls.stem) or (12, year)
        is_partial = month < 11   # before November is considered partial
        if is_partial and not args.partial_ok:
            log.warning("Skipping %s (month=%d) — partial-year bulletins disabled "
                        "(--no-partial-ok).", xls.name, month)
            continue
        log.info("Parsing %s (year=%d, month=%d)%s …",
                 xls.name, year, month, "  [partial-year]" if is_partial else "")
        per_crop = parse_xls(xls, partial_year=is_partial)
        for crop, iso_to_yield in per_crop.items():
            if crop not in per_oblast:
                continue
            per_oblast[crop].setdefault(year, {}).update(iso_to_yield)
            n_rows += len(iso_to_yield)
            # Late-harvest crops in a pre-November bulletin → partial flag.
            if is_partial and crop in LATE_HARVEST_CROPS:
                partial_flag[crop][year] = True

    log.info("Total per-(crop, year, oblast) rows extracted: %d", n_rows)

    if args.dry_run:
        log.info("Dry-run; not writing YAML.")
        return 0
    if n_rows == 0:
        log.warning("No usable rows extracted — leaving YAML unchanged.")
        return 0

    try:
        import yaml
    except ImportError:
        log.error("pyyaml is required: uv pip install pyyaml")
        return 1

    existing = yaml.safe_load(args.yaml.read_text(encoding="utf-8")) or {}
    populated = {c: years for c, years in per_oblast.items() if years}
    existing["per_oblast_yield"] = populated
    # Emit partial-year annotations separately so downstream code can
    # filter or de-weight those rows. Empty crop blocks are dropped.
    partial_annot = {c: years for c, years in partial_flag.items() if years}
    if partial_annot:
        existing["partial_year_flag"] = partial_annot
    elif "partial_year_flag" in existing:
        del existing["partial_year_flag"]

    args.yaml.write_text(
        yaml.dump(existing, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Updated %s — per_oblast_yield section has %d crops, %d rows total.",
             args.yaml, len(populated), n_rows)
    if partial_annot:
        log.info("Partial-year flags: %s", {c: sorted(yrs.keys()) for c, yrs in partial_annot.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
