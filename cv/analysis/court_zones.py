"""
cv/analysis/court_zones.py — Tennis court zone definitions and classifier.

═══════════════════════════════════════════════════════════════════════════════
COORDINATE SYSTEM
═══════════════════════════════════════════════════════════════════════════════

All zone boundaries are defined in **normalised court coordinates** (0.0–1.0)
where the origin (0, 0) is the top-left corner of the full doubles court
(far baseline, left doubles sideline) and (1, 1) is the bottom-right corner
(near baseline, right doubles sideline).

This matches the output of the homography transform built in AnalyticsPipeline.
Values are derived from the CourtReference pixel grid used by CourtDetector:
  - Raw x range: 286 → 1379  (total = 1093 px)
  - Raw y range: 561 → 2935  (total = 2374 px)

Key normalised boundaries:
  x=0.000   Left doubles sideline
  x=0.125   Left singles sideline
  x=0.500   Center service line
  x=0.875   Right singles sideline
  x=1.000   Right doubles sideline

  y=0.000   Far baseline  (top of court, far side)
  y=0.231   Far service line
  y=0.500   Net
  y=0.769   Near service line
  y=1.000   Near baseline (bottom of court, near side)

═══════════════════════════════════════════════════════════════════════════════
ZONE LAYOUT
═══════════════════════════════════════════════════════════════════════════════

The court is split into four horizontal bands:

  FAR BASELINE AREA  (y: 0.000 → 0.231)
  ┌─────┬───────┬───────┬───────┬───────┬─────┐
  │ AA  │   A   │   B   │   C   │   D   │ DD  │
  └─────┴───────┴───────┴───────┴───────┴─────┘
    dbl   ¼ sgl  ¼ sgl   ¼ sgl   ¼ sgl   dbl

  FAR SERVICE BOXES  (y: 0.231 → 0.500)
  ┌──────────────────────┬──────────────────────┐
  │  Wide │ Body │  Tee  │  Tee  │ Body │ Wide  │
  └──────────────────────┴──────────────────────┘
      ⅓ left svc box         ⅓ right svc box

  NEAR SERVICE BOXES  (y: 0.500 → 0.769)
  ┌──────────────────────┬──────────────────────┐
  │  Wide │ Body │  Tee  │  Tee  │ Body │ Wide  │
  └──────────────────────┴──────────────────────┘

  NEAR BASELINE AREA  (y: 0.769 → 1.000)
  ┌─────┬───────┬───────┬───────┬───────┬─────┐
  │ AA  │   A   │   B   │   C   │   D   │ DD  │
  └─────┴───────┴───────┴───────┴───────┴─────┘

Service box conventions:
  - Wide = zone nearest the singles sideline (outside third of service box)
  - Body = middle third of service box
  - Tee  = zone nearest the center service line (inside third of service box)
  - The same names apply on both sides: Wide is always outer, Tee is always inner.

Zone name format: {side}_{area}  e.g. "far_A", "near_service_left_tee"

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Court boundary constants (normalised 0–1) ─────────────────────────────────

# Derived from CourtReference pixel grid: x [286, 1379], y [561, 2935]
# x_norm = (raw_x - 286) / 1093
# y_norm = (raw_y - 561) / 2374

X_DOUBLES_LEFT   = 0.000   # Left doubles sideline
X_SINGLES_LEFT   = 0.125   # Left singles sideline  (137 / 1093)
X_CENTER         = 0.500   # Center service line     (546 / 1093)
X_SINGLES_RIGHT  = 0.875   # Right singles sideline  (956 / 1093)
X_DOUBLES_RIGHT  = 1.000   # Right doubles sideline

Y_FAR_BASELINE   = 0.000   # Far (top) baseline
Y_FAR_SERVICE    = 0.231   # Far service line        (549 / 2374)
Y_NET            = 0.500   # Net                     (1187 / 2374)
Y_NEAR_SERVICE   = 0.769   # Near service line       (1825 / 2374)
Y_NEAR_BASELINE  = 1.000   # Near (bottom) baseline

# Singles court interior width and its quarters
_SINGLES_WIDTH   = X_SINGLES_RIGHT - X_SINGLES_LEFT   # ≈ 0.750
_BASELINE_QTR    = _SINGLES_WIDTH / 4                  # ≈ 0.1875 per A/B/C/D zone

# Each service box spans half the singles interior width
_SVC_BOX_HALF    = _SINGLES_WIDTH / 2                  # ≈ 0.375
_SVC_THIRD       = _SVC_BOX_HALF / 3                   # ≈ 0.125 per Wide/Body/Tee


# ── Pre-computed x boundaries for service box zones ──────────────────────────

# Left service box  (singles left → center)
X_SVC_LEFT_WIDE_END  = X_SINGLES_LEFT + _SVC_THIRD          # outer edge → Wide end
X_SVC_LEFT_BODY_END  = X_SINGLES_LEFT + 2 * _SVC_THIRD      # Wide end → Body end
# X_CENTER is Tee end (left box's inner boundary)

# Right service box  (center → singles right)
X_SVC_RIGHT_TEE_END  = X_CENTER + _SVC_THIRD                 # center → Tee end
X_SVC_RIGHT_BODY_END = X_CENTER + 2 * _SVC_THIRD             # Tee end → Body end
# X_SINGLES_RIGHT is Wide end (right box's outer boundary)

# Pre-computed x boundaries for A/B/C/D baseline zones
X_A_END = X_SINGLES_LEFT + _BASELINE_QTR      # A ends here
X_B_END = X_SINGLES_LEFT + 2 * _BASELINE_QTR  # B ends here (≈ X_CENTER)
X_C_END = X_SINGLES_LEFT + 3 * _BASELINE_QTR  # C ends here
# X_SINGLES_RIGHT is D's outer boundary


# ── Zone dataclass ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CourtZone:
    """
    A rectangular region of the court in normalised (0-1) coordinates.

    Attributes:
        name:       Short identifier, e.g. "far_service_left_wide"
        label:      Human-readable label, e.g. "Far Left Wide"
        side:       Which half of the court — "far" or "near"
        area:       Broad area type — "baseline" or "service"
        sub_zone:   Fine-grained label — "AA","A","B","C","D","DD","wide","body","tee"
        x1, y1:     Top-left corner (normalised)
        x2, y2:     Bottom-right corner (normalised)
    """
    name:     str
    label:    str
    side:     str   # "far" | "near"
    area:     str   # "baseline" | "service"
    sub_zone: str   # "AA"|"A"|"B"|"C"|"D"|"DD"|"wide"|"body"|"tee"
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        """Return True if the normalised point (x, y) falls inside this zone."""
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


# ── Zone registry ──────────────────────────────────────────────────────────────

def _make_baseline_zones(side: str, y1: float, y2: float) -> list[CourtZone]:
    """Build the 6 AA/A/B/C/D/DD zones for one baseline strip."""
    prefix = f"{side}_baseline"
    return [
        CourtZone(f"{prefix}_AA", f"{'Far' if side=='far' else 'Near'} AA (Left Alley)",
                  side, "baseline", "AA",
                  X_DOUBLES_LEFT, y1, X_SINGLES_LEFT, y2),
        CourtZone(f"{prefix}_A",  f"{'Far' if side=='far' else 'Near'} A",
                  side, "baseline", "A",
                  X_SINGLES_LEFT, y1, X_A_END, y2),
        CourtZone(f"{prefix}_B",  f"{'Far' if side=='far' else 'Near'} B",
                  side, "baseline", "B",
                  X_A_END, y1, X_B_END, y2),
        CourtZone(f"{prefix}_C",  f"{'Far' if side=='far' else 'Near'} C",
                  side, "baseline", "C",
                  X_B_END, y1, X_C_END, y2),
        CourtZone(f"{prefix}_D",  f"{'Far' if side=='far' else 'Near'} D",
                  side, "baseline", "D",
                  X_C_END, y1, X_SINGLES_RIGHT, y2),
        CourtZone(f"{prefix}_DD", f"{'Far' if side=='far' else 'Near'} DD (Right Alley)",
                  side, "baseline", "DD",
                  X_SINGLES_RIGHT, y1, X_DOUBLES_RIGHT, y2),
    ]


def _make_service_zones(side: str, y1: float, y2: float) -> list[CourtZone]:
    """
    Build the 6 service box zones for one half of the court.

    Left service box (x: singles_left → center):
      Wide → Body → Tee  (left to right)

    Right service box (x: center → singles_right):
      Tee  → Body → Wide  (left to right — mirror of left)
    """
    prefix = f"{side}_service"
    tag = "Far" if side == "far" else "Near"
    return [
        # Left service box
        CourtZone(f"{prefix}_left_wide",  f"{tag} Left Wide",
                  side, "service", "wide",
                  X_SINGLES_LEFT,       y1, X_SVC_LEFT_WIDE_END, y2),
        CourtZone(f"{prefix}_left_body",  f"{tag} Left Body",
                  side, "service", "body",
                  X_SVC_LEFT_WIDE_END,  y1, X_SVC_LEFT_BODY_END, y2),
        CourtZone(f"{prefix}_left_tee",   f"{tag} Left Tee",
                  side, "service", "tee",
                  X_SVC_LEFT_BODY_END,  y1, X_CENTER,             y2),

        # Right service box (mirror: Tee is on the left side of this box)
        CourtZone(f"{prefix}_right_tee",  f"{tag} Right Tee",
                  side, "service", "tee",
                  X_CENTER,             y1, X_SVC_RIGHT_TEE_END,  y2),
        CourtZone(f"{prefix}_right_body", f"{tag} Right Body",
                  side, "service", "body",
                  X_SVC_RIGHT_TEE_END,  y1, X_SVC_RIGHT_BODY_END, y2),
        CourtZone(f"{prefix}_right_wide", f"{tag} Right Wide",
                  side, "service", "wide",
                  X_SVC_RIGHT_BODY_END, y1, X_SINGLES_RIGHT,      y2),
    ]


# Master ordered list of all 24 zones (top-to-bottom, left-to-right within each row)
ALL_ZONES: list[CourtZone] = [
    *_make_baseline_zones("far",  Y_FAR_BASELINE, Y_FAR_SERVICE),    # 6 zones
    *_make_service_zones ("far",  Y_FAR_SERVICE,  Y_NET),             # 6 zones
    *_make_service_zones ("near", Y_NET,           Y_NEAR_SERVICE),   # 6 zones
    *_make_baseline_zones("near", Y_NEAR_SERVICE,  Y_NEAR_BASELINE),  # 6 zones
]

# Fast lookup by name
ZONE_BY_NAME: dict[str, CourtZone] = {z.name: z for z in ALL_ZONES}


# ── Public classifier ─────────────────────────────────────────────────────────

def classify(court_x: float, court_y: float) -> Optional[CourtZone]:
    """
    Return the CourtZone that contains the normalised point (court_x, court_y).

    Args:
        court_x: Normalised x coordinate (0 = left doubles sideline, 1 = right)
        court_y: Normalised y coordinate (0 = far baseline, 1 = near baseline)

    Returns:
        The matching CourtZone, or None if the point is outside the court
        (e.g. out of bounds, or exactly on a boundary that rounds to 1.0+).

    Example:
        >>> zone = classify(0.55, 0.60)
        >>> zone.name
        'near_service_right_tee'
        >>> zone.label
        'Near Right Tee'
    """
    for zone in ALL_ZONES:
        if zone.contains(court_x, court_y):
            return zone
    return None


def classify_name(court_x: float, court_y: float) -> Optional[str]:
    """Convenience wrapper — returns just the zone name string."""
    zone = classify(court_x, court_y)
    return zone.name if zone else None


def get_zone(name: str) -> Optional[CourtZone]:
    """Look up a zone by its name string."""
    return ZONE_BY_NAME.get(name)


def zones_for_side(side: str) -> list[CourtZone]:
    """Return all zones for 'far' or 'near' side."""
    return [z for z in ALL_ZONES if z.side == side]


def zones_for_area(area: str) -> list[CourtZone]:
    """Return all zones of a given area type: 'baseline' or 'service'."""
    return [z for z in ALL_ZONES if z.area == area]


# ── Debug / inspection ────────────────────────────────────────────────────────

def print_zone_table() -> None:
    """Print a summary table of all zones with their boundaries."""
    header = f"{'Name':<35} {'Label':<25} {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6}"
    print(header)
    print("-" * len(header))
    for z in ALL_ZONES:
        print(f"{z.name:<35} {z.label:<25} {z.x1:>6.3f} {z.y1:>6.3f} {z.x2:>6.3f} {z.y2:>6.3f}")


if __name__ == "__main__":
    print_zone_table()
    print(f"\nTotal zones: {len(ALL_ZONES)}")
    # Quick sanity check
    test_cases = [
        (0.05, 0.10, "far_baseline_AA"),
        (0.20, 0.10, "far_baseline_A"),
        (0.40, 0.10, "far_baseline_B"),
        (0.60, 0.10, "far_baseline_C"),
        (0.80, 0.10, "far_baseline_D"),
        (0.95, 0.10, "far_baseline_DD"),
        (0.13, 0.35, "far_service_left_wide"),
        (0.30, 0.35, "far_service_left_body"),
        (0.45, 0.35, "far_service_left_tee"),
        (0.55, 0.35, "far_service_right_tee"),
        (0.70, 0.35, "far_service_right_body"),
        (0.85, 0.35, "far_service_right_wide"),
        (0.40, 0.85, "near_baseline_B"),
    ]
    print("\nSanity checks:")
    all_ok = True
    for cx, cy, expected in test_cases:
        result = classify_name(cx, cy)
        ok = "✓" if result == expected else "✗"
        if result != expected:
            all_ok = False
        print(f"  {ok}  classify({cx:.2f}, {cy:.2f}) = {result!r:40s}  (expected {expected!r})")
    if all_ok:
        print("\n✓ All zone checks passed.")
    else:
        print("\n✗ Some zone checks FAILED — review boundary constants.")
