# Roadmap

Forward-looking tracks beyond the FUNXD-SYNTH v0-beta release. Items
are listed by **dependency**, not strict priority — issues that
unblock others come first. Each track is a GitHub issue (or `obs-`
note where unfiled).

This file lives on `main`; it intentionally describes both shipped
state and planned work in a single forward-looking document.

---

## Active tracks

### Image-fallback algorithm

- **#8 — Checkmark / checkbox detection** *(feature, not a fix)*

  When an answer field is a checkbox, the only real answer is the
  **tick mark inside the box**. The current pipeline erases the
  entire annotated bbox — which on XFUND golden-set fixtures
  contains the *label* of the ticked checkbox (e.g. *"Erstbestellung
  (mit Bild)"*). That's semantically wrong: the question is the
  label, the answer is the tick state. Goal: detect the box, erase
  only the tick mask, preserve the label + box outline. Canonical
  fixture: `fr_train_39` ("comment connaissez-vous FACT?").

  *Unblocks*: touch-up suppression on checkbox rows, and is a soft
  prereq for #10 (meta-output).

- **#7 — Dot-vs-glyph discriminator**

  Current focus before touch-up can ship enabled-by-default. FUNSD
  forms build fill-in baselines from rows of repeated typewriter
  characters (`ffff`/`oooo`/periods); the connected-component dot
  detector reads them as dotted lines (same ~7–8 px size as XFUND's
  real bold dots, so a size cap can't separate them). Candidate
  signals: duty-cycle (gap ratio), fill-ratio.

  *Unblocks*: touch-up ON by default in the release recipe.

- **#3 — Median-fill answer-location ghosts**

  On multi-coloured backgrounds the sampled paper-colour median sits
  between dominant colours, leaving a faint readable rectangle at
  the answer location (anti-cheating risk). Also reports bbox-extent
  mismatches that erase labels. Per-pixel paper sampling is the
  likely fix.

- **#9 — Dashed-line completion + fillable-region dot synthesis**

  Dashes are a distinct primitive from dotted lines. Speculative
  experiment: synthesising dots at fillable answer locations that
  have no surviving dots to bracket post-erase. Distinct from
  `obs-13` below.

- **`obs-13` — Pre-erase dotted-line detection** *(unfiled)*

  For dotted lines that fall **fully inside** an answer bbox, there
  are no surviving dots to bracket the gap after erasure. The
  clone-stamp touch-up has nothing to anchor on. Fix: detect dotted
  clusters on the **clean** image first, so the line can be rebuilt
  even when no part of it survives. Distinct from #7; does not fix
  the FUNSD typewriter-glyph FPs on its own.

### Release tooling

- **#10 — Meta-output uncertainty flags**

  Emit a per-run `flags.jsonl` next to `manifest.jsonl` with five
  orthogonal signals (pixel-coverage anomaly, bbox-area outlier,
  paper-colour variance, dotted-cluster overlap, touch-up rejections)
  so the release build surfaces its own uncertainty instead of
  needing a 596-doc visual review.

  *Soft prereq*: #8 must land first or every checkbox row fires
  every signal.

  Explicitly **not** a single confidence score — the signals capture
  orthogonal failure modes and shouldn't be conflated.

---

## Dependent work (downstream of the above)

These items are deliberately **not** filed as standalone issues
because they only become well-scoped once their prereq lands.

- **Touch-up suppression on checkbox rows.** Once #8 lands, extend
  `touch_up.py` to skip dotted-line detection inside rows the
  checkbox detector flagged. Today the touch-up extends dotted
  lines across `fr_train_39` because the answer bboxes there
  contain checkbox labels, and the touch-up reads the resulting
  row as a dotted-line cluster. **This is a side benefit of #8, not
  the reason #8 exists** — but it's the natural follow-on and
  should be tracked so it isn't forgotten.

- **Touch-up enabled-by-default in the release recipe.** Flip
  `funxd-synth-v0-beta.touch_up_dotted_lines = True` once #7's
  discriminator suppresses FUNSD FPs. Until then, touch-up is
  opt-in via `--touch-up-dotted-lines`.

---

## Sequencing graph

```
    #8 (checkbox detection) ─┬─► touch-up suppression on checkbox rows
                             └─► soft prereq for #10 (meta-output)

    #7 (dot-vs-glyph)        ───► touch-up ON by default in release

    #3, #9, obs-13: independent tracks (no prereqs, no dependents)
```

---

## Decisions and rationale

- **Touch-up stays opt-in / OFF in v0-beta release.** FUNSD's
  typewriter fill-character baselines fire the dot detector at the
  same size as XFUND's real bold dots; size cap can't separate
  them. Re-enabling by default needs #7. Until then, touch-up is
  available via `--touch-up-dotted-lines` for QA / xfund-style
  forms.

- **Checkmark detection is a feature, not a touch-up fix.** Its
  value is independent of dotted lines: semantically, the bbox
  annotation is over-broad for a checkbox answer regardless of
  what touch-up does. The touch-up benefit is a side effect, not
  the goal — that's why "touch-up suppression on checkbox rows"
  is listed as **dependent work**, not as part of #8 itself.

- **No single "confidence score" in meta-output.** Five orthogonal
  signals stay orthogonal in `flags.jsonl`. The reviewer reads the
  flag name and decides. See #10.

- **VRDU registration corpus deferred.** The whole subset turned
  out to be OCR'd scans rather than born-digital PDFs; routing
  correctly requires a classifier refinement (full-page-image-
  XObject detection → new `ocrd_pdf` category) that is queued but
  not landed. Not on the v0-beta critical path.

---

## How this document is maintained

- **One source of truth.** GitHub issues hold full scope, history,
  and comments; this file is the index that captures dependencies
  and sequencing.
- Add a track here when you file the issue. Remove it when the
  issue closes.
- Dependencies / sequencing graph is the load-bearing part — keep
  it accurate. The narrative around each issue can stay terse.
