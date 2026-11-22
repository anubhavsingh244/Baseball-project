# Baseball-project

A small machine learning project that models MLB team wins and publishes the results as a static site of interactive Plotly charts.

## What it does

- Downloads `Teams.csv` from the [Baseball Databank](https://github.com/chadwickbureau/baseballdatabank) (CC BY-SA 3.0), using a copy in the `xorq-labs/baseballdatabank` mirror pinned to a fixed commit, so the downloaded dataset is reproducible.
- Builds per-game features for seasons since 1900: runs scored and allowed, hits, home runs, walks and strikeouts (batting and allowed), and errors. ERA is used as is, since it is already a rate.
- Trains on seasons before 2005 and tests on 2005 onward.
- Compares a Pythagorean-expectation baseline, linear regression and a random forest.
- Writes `site/index.html` with four charts: actual vs estimated wins, model error, feature importance, and league scoring trends.

## What the model does and does not do

The features come from the same season as the target. The model therefore explains how a season's stats map to its win total. It does not forecast future wins. Runs scored and allowed already determine most of a team's record, so the Pythagorean baseline is a strong reference point.

The first chart shows the model with the lowest test error. That model is chosen on the test seasons, so its error is slightly optimistic.

## Run locally

```bash
pip install -r requirements.txt
python src/pipeline.py
```

Then open `site/index.html` in a browser. The data is cached in `data/Teams.csv`. Delete that file to force a fresh download, for example after the pinned commit changes.

## Hosting

The workflow in `.github/workflows/pages.yml` builds the site on every pull request and deploys it to GitHub Pages when changes reach `main`.

One-time setup: in the repo, go to Settings > Pages and set Source to "GitHub Actions".
