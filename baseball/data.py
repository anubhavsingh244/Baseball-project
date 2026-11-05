"""CSV loading and per-player stat computation."""
import csv
from pathlib import Path

from . import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATTING_FIELDS = ("G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "HBP", "SF", "SO")
PITCHING_FIELDS = ("W", "L", "G", "GS", "H", "ER", "HR", "BB", "SO")


def _load(path, int_fields, float_fields=()):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in int_fields:
                row[k] = int(row[k])
            for k in float_fields:
                row[k] = float(row[k])
            row["year"] = int(row["year"])
            rows.append(row)
    return rows


def load_batting(path=DATA_DIR / "batting.csv"):
    rows = _load(path, BATTING_FIELDS)
    for r in rows:
        r["AVG"] = stats.batting_average(r["H"], r["AB"])
        r["OBP"] = stats.on_base_percentage(r["H"], r["BB"], r["HBP"], r["AB"], r["SF"])
        r["SLG"] = stats.slugging(r["H"], r["2B"], r["3B"], r["HR"], r["AB"])
        r["OPS"] = r["OBP"] + r["SLG"]
        r["ISO"] = r["SLG"] - r["AVG"]
    return rows


def load_pitching(path=DATA_DIR / "pitching.csv"):
    rows = _load(path, PITCHING_FIELDS, ("IP",))
    for r in rows:
        ip = stats.innings_to_float(r["IP"])
        r["ERA"] = stats.era(r["ER"], ip)
        r["WHIP"] = stats.whip(r["H"], r["BB"], ip)
        r["K/9"] = stats.k_per_9(r["SO"], ip)
        r["BB/9"] = stats.bb_per_9(r["BB"], ip)
        r["FIP"] = stats.fip(r["HR"], r["BB"], r["SO"], ip)
    return rows


def career_batting(rows):
    """Aggregate counting stats across seasons per player and recompute rates."""
    totals = {}
    for r in rows:
        t = totals.setdefault(r["player"], {k: 0 for k in BATTING_FIELDS})
        for k in BATTING_FIELDS:
            t[k] += r[k]
    out = []
    for player, t in totals.items():
        obp = stats.on_base_percentage(t["H"], t["BB"], t["HBP"], t["AB"], t["SF"])
        slg = stats.slugging(t["H"], t["2B"], t["3B"], t["HR"], t["AB"])
        out.append({"player": player, **t, "AVG": stats.batting_average(t["H"], t["AB"]),
                    "OBP": obp, "SLG": slg, "OPS": obp + slg})
    return out
