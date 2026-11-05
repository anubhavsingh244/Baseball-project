"""Command line interface: python -m baseball.cli {batting,pitching,career}"""
import argparse

from . import data


def _table(rows, cols, fmt):
    widths = [max(len(c), *(len(fmt[c](r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(fmt[c](r[c]).ljust(w) for c, w in zip(cols, widths)))


def _rate(x):
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"


def main(argv=None):
    p = argparse.ArgumentParser(prog="baseball", description=__doc__)
    p.add_argument("report", choices=["batting", "pitching", "career"])
    p.add_argument("--year", type=int, help="filter to one season")
    p.add_argument("--sort", help="stat to sort by (default: OPS / ERA)")
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args(argv)

    plain = {"player": str, "team": str, "year": str}
    if a.report == "pitching":
        rows = data.load_pitching()
        cols = ["player", "team", "year", "IP", "ERA", "WHIP", "K/9", "BB/9", "FIP"]
        fmt = {**plain, "IP": lambda x: f"{x:.1f}", "ERA": lambda x: f"{x:.2f}",
               "WHIP": lambda x: f"{x:.2f}", "K/9": lambda x: f"{x:.1f}",
               "BB/9": lambda x: f"{x:.1f}", "FIP": lambda x: f"{x:.2f}"}
        key, reverse = a.sort or "ERA", False
    else:
        rows = data.load_batting()
        if a.report == "career":
            rows = data.career_batting(rows)
            cols = ["player", "G", "AB", "H", "HR", "AVG", "OBP", "SLG", "OPS"]
        else:
            cols = ["player", "team", "year", "AB", "HR", "AVG", "OBP", "SLG", "OPS"]
        fmt = {**plain, **{c: str for c in ("G", "AB", "H", "HR")},
               **{c: _rate for c in ("AVG", "OBP", "SLG", "OPS")}}
        key, reverse = a.sort or "OPS", True
    if a.year:
        rows = [r for r in rows if r.get("year") == a.year]
    rows = sorted(rows, key=lambda r: r[key], reverse=reverse)[: a.top]
    if not rows:
        print("No rows match.")
        return
    _table(rows, cols, fmt)


if __name__ == "__main__":
    main()
