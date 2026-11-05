# Baseball Project

A small, dependency-free Python toolkit for baseball statistics analysis.

## Features
- Batting: AVG, OBP, SLG, OPS, ISO
- Pitching: ERA, WHIP, K/9, BB/9, FIP (handles `195.1` innings notation)
- Career aggregation across seasons
- CLI reports over CSV data in `data/` (sample data is fictional)

## Usage
```bash
python -m baseball.cli batting --year 2024
python -m baseball.cli pitching --sort FIP
python -m baseball.cli career --top 5
```

## Tests
```bash
python -m unittest discover -s tests
```

## Using real data
Replace `data/batting.csv` / `data/pitching.csv` with your own files using the
same column headers (e.g. exported from Lahman or Baseball Reference).
