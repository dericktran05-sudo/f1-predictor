# F1 Race Outcome Predictor

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