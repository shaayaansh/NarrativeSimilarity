# NarrativeSimilarity

This repository analyzes perceived narrative similarity ratings using event alignments and semantic similarity between aligned events.

## Data and Target

- Data lives under `data/`.
- The prediction target is participant-provided `event_rating`, converted to numeric `event_rating_num` (1-4).
- For modeling at pair level, the target is aggregated as `event_rating_num_mean` per `PairID`.

## Model 1: Alignment-Only

Pair-level model:

`event_rating_num_mean ≈ alpha * alignment_similarity`

Where:
- `alignment_similarity = 1 - D_alignment`
- `D_alignment` includes:
  - `cost_skip`: unmatched events across both stories
  - `cost_reorder`: event-level inversion count from many-to-many aligned pairs
  - normalization to keep distance in `[0, 1]`

### Results

- Pairs evaluated: `98`
- `alpha_hat = 3.3954`
- `RMSE = 1.4759`
- `MAE = 1.3612`
- `R^2 = -7.6702`
- `Pearson r = 0.5649`

### Interpretation

- Moderate positive correlation (`r≈0.56`) means alignment captures directional signal.
- Large errors and strongly negative `R^2` indicate poor absolute prediction/calibration.
- Alignment structure alone is not sufficient for accurate rating prediction.

## Model 2: Alignment + Semantic Similarity

Joint pair-level model:

`event_rating_num_mean ≈ alpha * alignment_similarity + beta * semantic_similarity`

Where semantic similarity is computed from SentenceTransformer embeddings over aligned event pairs.

### Results

- `alpha_hat_joint = 0.4211`
- `beta_hat_joint = 3.1387`
- `RMSE = 1.3999`
- `MAE = 1.2248`
- `R^2 = -6.8006`

### Interpretation

- `beta` is much larger than `alpha`, so semantic similarity contributes most of the predictive signal in this linear setup.
- RMSE/MAE improved versus alignment-only.
- `R^2` remains negative, so the model still underperforms a mean baseline in absolute prediction.

## Notebook

Main workflow is in:

- `src/modeling.ipynb`

