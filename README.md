# NarrativeSimilarity

This repository analyzes perceived narrative similarity ratings using event alignments and semantic similarity between aligned events.

## Data and Target

- Data lives under `data/`.
- The prediction target is participant-provided `event_rating`, converted to numeric `event_rating_num` (1-4).
- For modeling at pair level, the target is aggregated as `event_rating_num_mean` per `PairID`.

## Alignment Distance (`D_alignment`)

The structural distance is computed from the full alignment dictionary at event level:

- `cost_skip`: unmatched events across both stories.
- `cost_reorder`: inversion count over induced B-order from many-to-many aligned event pairs.
- Normalization keeps `D_alignment` in `[0, 1]`.

Then:

- `alignment_similarity = 1 - D_alignment`

## Model 1: Alignment-Only (With Intercept)

Pair-level model:

`event_rating_num_mean ≈ c + alpha * alignment_similarity`

### Results

- Pairs evaluated: `98`
- `c_hat = 1.6057`
- `alpha_hat = 1.0122`
- `RMSE = 0.4136`
- `MAE = 0.3367`
- `R^2 = 0.3192`
- `Pearson r = 0.5649`

### Interpretation

- Adding an intercept materially improved calibration compared to no-intercept fitting.
- The model now explains about 31.9% of pair-level variance.
- Alignment structure alone provides meaningful signal but leaves room for improvement.

## Semantic Distance (`D_semantic`)

Semantic distance is computed over aligned event pairs:

`D_semantic(S_a, S_b) = (1 / |A|) * sum_{(i,j) in A} d(e_ai, e_bj)`

where `d` is normalized cosine distance from SentenceTransformer embeddings.

Then:

- `semantic_similarity = 1 - D_semantic`

## Model 2: Alignment + Semantic Similarity (With Intercept)

Joint pair-level model:

`event_rating_num_mean ≈ c + alpha * alignment_similarity + beta * semantic_similarity`

### Results

- `c_hat_joint = 1.5672`
- `alpha_hat_joint = 0.1376`
- `beta_hat_joint = 0.9832`
- `RMSE = 0.3883`
- `MAE = 0.3208`
- `R^2 = 0.3999`
- `Pearson r = 0.6324`

### Interpretation

- Joint model improves over alignment-only model (`RMSE` down, `R^2` up, `r` up).
- `beta_hat_joint > alpha_hat_joint`, suggesting semantic similarity contributes more than alignment similarity in this linear specification.
- Combining structure + semantics gives the best current pair-level explanation of average event ratings.

## Notebook

Main workflow is in:

- `src/modeling.ipynb`

