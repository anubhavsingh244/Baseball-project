# Baseball-project

A small machine learning project that predicts MLB team wins and publishes the results as a static site of interactive Plotly charts.

## What it does

- Downloads `Teams.csv` from the [Baseball Databank](https://github.com/chadwickbureau/baseballdatabank) (CC BY-SA 3.0).
- Builds per-game features (runs, hits, home runs, walks, strikeouts, errors, ERA) for seasons since 1900.
- Trains on seasons before 2005 and tests on 2005 onward.
- Compares a Pythagorean-expectation baseline, linear regression and a random forest.
- Writes `site/index.html` with four charts: actual vs predicted wins, model error, feature importance, and league scoring trends.

## Run locally

```bash
pip install -r requirements.txt
python src/pipeline.py
```

Then open `site/index.html` in a browser. The data is cached in `data/Teams.csv`.

## Hosting

The workflow in `.github/workflows/pages.yml` builds the site on every pull request and deploys it to GitHub Pages when changes reach `main`.

One-time setup: in the repo, go to Settings > Pages and set Source to "GitHub Actions".
