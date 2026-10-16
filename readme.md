# F1 Race Outcome Predictor
**Live demo:** https://f1-predictor-wshjdayvdgxpoaltj7bxum.streamlit.app/

Predicts Formula 1 finishing order from qualifying position, recent form, team strength, and track history, then looks at **where and why predictions fail**.

## The question
Qualifying position is already a strong predictor of the result. Can a model built on form and track history beat it, and what breaks both when a race gets chaotic?

## Data
- Race results from the [FastF1](https://docs.fastf1.dev/) library, 2023-2025 (70 races, about 1,400 driver results)
- Trained on 2023-2024, tested on all 24 races of 2025, which the model never saw
- Features use only information available before each race, with no leakage

## Features
Grid position, driver form (average finish over the last 5 races), team form, driver's average finish at that circuit, and team DNF rate.

## Results (2025 test season)

| Predictor | Spearman | Avg. position error |
|---|---|---|
| Grid-position baseline | 0.651 | 3.34 |
| LightGBM model | 0.656 | 3.38 |

With retirements removed, the model scores 0.779 against the baseline's 0.755. It reads pace slightly better than qualifying order, but the edge is small.

## Where it fails

| DNFs in race | Races | Model | Grid baseline |
|---|---|---|---|
| 0-1 | 7 | 0.784 | 0.811 |
| 2-3 | 9 | 0.626 | 0.609 |
| 4+ | 8 | 0.577 | 0.559 |

**Main finding:** retirements, not misjudged pace, cause most of the error. Going from calm to chaotic races costs about 0.2 of Spearman, ten times more than the model's edge over the baseline.

The number of DNFs is only a rough measure of chaos. Miami had 4 DNFs and was predicted well, while Zandvoort had 3 and was predicted badly. Who retires, and when, matters more than how many.

### Worst races
<!-- Fill in one line per race after looking up what happened -->
- Zandvoort:
- Las Vegas:
- Shanghai:
- Silverstone:
- Spielberg:

## Limitations
- Only two seasons of training data
- Weather features could not be loaded (the data source failed), so rain is not modeled
- Only 24 test races, so small differences between models are noise
- DNF count is a crude stand-in for chaos
## 2026: what happens when the rules reset?

2026 brought new regulations, 11 teams (including Audi and Cadillac), and a different pecking order. I pulled the 17 races run so far and made team features restart each season, while driver features carry over.

Three predictors were compared on all 17 races:
- **Grid baseline:** finish where you qualified
- **Frozen model:** trained on 2023-2025 only
- **Walk-forward model:** retrained before each 2026 race on everything earlier

| Races | Grid | Frozen | Walk-forward |
|---|---|---|---|
| All 17 | 0.647 | 0.658 | 0.664 |
| 1-8 | 0.638 | 0.632 | 0.628 |
| 9-17 | 0.654 | 0.681 | 0.695 |

*Average Spearman correlation.*

Paired comparison across the 17 races:

| Comparison | Mean gap | Better in |
|---|---|---|
| Walk-forward vs grid | +0.017 (± 0.020) | 9 of 17 |
| Frozen vs grid | +0.011 (± 0.018) | 7 of 17 |
| Walk-forward vs frozen | +0.006 (± 0.009) | 8 of 17 |

**Finding:** across 17 races of the 2026 rule reset, a model using form, team strength, and track history was statistically tied with predicting that drivers finish where they qualified. Retraining on 2026 data as the season went on made no reliable difference. A likely reason is that qualifying position already reflects car pace and driver form (a hypothesis, not tested here).

### Live test: United States Grand Prix
The prediction is made after qualifying, pushed to GitHub before the race, and scored afterward.
- Prediction: *(add after qualifying)*
- Result: *(add after the race)*

## Run it yourself
```
pip install fastf1 pandas numpy scikit-learn lightgbm matplotlib streamlit pyarrow scipy
python get_data.py
python clean.py
python features.py
python train.py
python failure.py
streamlit run app.py
```