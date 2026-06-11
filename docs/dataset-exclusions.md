# Dataset exclusions

Documents deliberately kept out of FUNXD-SYNTH releases, with rationale.

The machine-readable source of truth is `EXCLUSIONS` in
`src/aff/synth/build_dataset.py` — each release recipe drops these
`doc_id`s at manifest-build time (`build_manifest(exclude_doc_ids=...)`),
and the count lands in `manifest.json`'s `build_stats.excluded_dropped`.
This file is the longer-form log.

| doc_id | source | reason | date |
| --- | --- | --- | --- |
| `fr_train_70` | xfund_fr | Mislabeled annotations — the answer bboxes don't line up with the rendered content, so both the blanked output and the labels are wrong. Not a generator bug; the upstream annotation is bad. | 2026-06-11 |

## Adding an exclusion

1. Add the `doc_id` + a one-line reason to `EXCLUSIONS` in
   `src/aff/synth/build_dataset.py`.
2. Add a row here with the source and date.
3. Re-run the affected release build; confirm `build_stats.excluded_dropped`
   incremented and the doc is gone from `manifest.json`.
