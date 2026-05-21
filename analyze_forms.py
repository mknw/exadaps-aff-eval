"""
analyze_forms.py — Systematic comparison of blank forms vs originals.

Metrics:
  1. Dataset overview (field counts, answer counts per source)
  2. Pixel-level diff (% pixels changed, mean brightness shift)
  3. Redaction quality (whiteness of answer regions in blank forms)
  4. Structural preservation (page count match, file size ratios)
  5. Visual side-by-side comparisons
"""

import json
import os
import glob
import random
import sys
from collections import defaultdict
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

random.seed(42)

FIELDS_DIR = "data/consolidated/fields"
BLANK_DIR = "data/generated/blank_forms"

SOURCE_PREFIXES = [
    "vrdu_ad_buy", "vrdu_registration",
    "xfund_de", "xfund_fr",
    "synthetic_supplier", "synthetic_invoice",
    "synthetic_compliance", "synthetic_patient",
    "funsd",
]


def source_subdir(stem):
    for prefix in SOURCE_PREFIXES:
        if stem.startswith(prefix + "_") or stem == prefix:
            return prefix
    return stem.split("_", 1)[0]


def load_all_docs():
    """Load all non-rvlcdip field JSONs, grouped by source."""
    by_source = defaultdict(list)
    for jf in sorted(glob.glob(f"{FIELDS_DIR}/*.json")):
        name = Path(jf).stem
        if name.startswith("rvlcdip_"):
            continue
        doc = json.load(open(jf, encoding="utf-8"))
        by_source[doc["source"]].append((jf, doc))
    return by_source


def dataset_overview(by_source):
    print("=" * 80)
    print("1. DATASET OVERVIEW")
    print("=" * 80)
    header = f"{'Source':25s} {'Total':>6s} {'w/Ans':>6s} {'AvgFld':>7s} {'AvgAns':>7s} {'AvgQ':>6s} {'AvgHdr':>7s} {'Pages':>6s}"
    print(header)
    print("-" * 80)

    for src in sorted(by_source):
        docs = by_source[src]
        total = len(docs)
        with_ans = sum(
            1 for _, d in docs
            if any(f["role"] == "answer" for f in d.get("fields", []))
        )
        field_counts = [len(d.get("fields", [])) for _, d in docs]
        ans_counts = [
            sum(1 for f in d.get("fields", []) if f["role"] == "answer")
            for _, d in docs
        ]
        q_counts = [
            sum(1 for f in d.get("fields", []) if f["role"] == "question")
            for _, d in docs
        ]
        h_counts = [
            sum(1 for f in d.get("fields", []) if f["role"] == "header")
            for _, d in docs
        ]
        page_counts = [d.get("page_count", 1) for _, d in docs]

        print(
            f"{src:25s} {total:6d} {with_ans:6d} "
            f"{np.mean(field_counts):7.1f} {np.mean(ans_counts):7.1f} "
            f"{np.mean(q_counts):6.1f} {np.mean(h_counts):7.1f} "
            f"{np.mean(page_counts):6.1f}"
        )
    print()


def pixel_analysis(by_source, n_samples=20):
    print("=" * 80)
    print("2. PIXEL-LEVEL ANALYSIS (original vs blank, page 0)")
    print("=" * 80)
    header = (
        f"{'Source':25s} {'N':>3s} {'%Changed':>9s} {'StdDev':>7s} "
        f"{'OrigBright':>11s} {'BlankBright':>12s} {'SizeRatio':>10s}"
    )
    print(header)
    print("-" * 80)

    all_results = {}

    for src in sorted(by_source):
        docs = by_source[src]
        sample = random.sample(docs, min(n_samples, len(docs)))

        pct_changes = []
        orig_brights = []
        blank_brights = []
        size_ratios = []

        for jf, doc in sample:
            stem = Path(jf).stem
            subdir = source_subdir(stem)
            blank_pdf = Path(BLANK_DIR) / subdir / f"{stem}.pdf"
            if not blank_pdf.exists():
                continue

            img_rel = doc.get("image_path", "").replace("\\", "/")
            orig_path = Path(img_rel)
            if not orig_path.exists():
                continue

            try:
                orig = np.array(Image.open(orig_path).convert("RGB"))

                pdf_doc = fitz.open(str(blank_pdf))
                pix = pdf_doc[0].get_pixmap(dpi=150)
                blank = np.array(
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                )
                pdf_doc.close()

                # Resize blank to match original dimensions
                if orig.shape[:2] != blank.shape[:2]:
                    blank_pil = Image.fromarray(blank).resize(
                        (orig.shape[1], orig.shape[0]), Image.LANCZOS
                    )
                    blank = np.array(blank_pil)

                diff = np.abs(orig.astype(float) - blank.astype(float))
                changed = diff.mean(axis=2) > 10
                pct = changed.mean() * 100

                pct_changes.append(pct)
                orig_brights.append(orig.mean())
                blank_brights.append(blank.mean())
                size_ratios.append(blank_pdf.stat().st_size / orig_path.stat().st_size)
            except Exception as e:
                continue

        if pct_changes:
            r = {
                "pct_mean": np.mean(pct_changes),
                "pct_std": np.std(pct_changes),
                "orig_bright": np.mean(orig_brights),
                "blank_bright": np.mean(blank_brights),
                "size_ratio": np.mean(size_ratios),
                "n": len(pct_changes),
            }
            all_results[src] = r
            print(
                f"{src:25s} {r['n']:3d} {r['pct_mean']:8.2f}% {r['pct_std']:6.2f} "
                f"{r['orig_bright']:11.1f} {r['blank_bright']:12.1f} "
                f"{r['size_ratio']:10.2f}x"
            )

    print()
    return all_results


def redaction_quality(by_source, n_samples=15):
    print("=" * 80)
    print("3. REDACTION QUALITY (answer bbox regions in blank forms)")
    print("=" * 80)
    header = (
        f"{'Source':25s} {'N_fields':>9s} {'AvgWhite':>9s} {'MinWhite':>9s} "
        f"{'>=98%':>6s} {'>=95%':>6s} {'<90%':>6s}"
    )
    print(header)
    print("-" * 80)

    for src in sorted(by_source):
        docs = by_source[src]
        sample = random.sample(docs, min(n_samples, len(docs)))
        is_synthetic = src.startswith("synthetic_")

        whiteness_scores = []

        for jf, doc in sample:
            stem = Path(jf).stem
            subdir = source_subdir(stem)
            blank_pdf = Path(BLANK_DIR) / subdir / f"{stem}.pdf"
            if not blank_pdf.exists():
                continue

            try:
                pdf_doc = fitz.open(str(blank_pdf))
                pix = pdf_doc[0].get_pixmap(dpi=150)
                blank = np.array(
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                )
                pdf_doc.close()
                h, w = blank.shape[:2]

                answers = [
                    f for f in doc.get("fields", [])
                    if f["role"] == "answer" and f.get("page", 0) == 0
                ]

                for fld in answers:
                    bbox = fld.get("bbox_norm", [])
                    if len(bbox) != 4:
                        continue
                    x0, y0, x1, y1 = bbox
                    if is_synthetic:
                        # Synthetic bboxes use PDF coords — but for blanked
                        # forms rendered via AcroForm clearing, the regions
                        # should be visually empty. Check rendered position.
                        y0, y1 = 1.0 - y1, 1.0 - y0
                    if (x1 - x0) < 0.001 or (y1 - y0) < 0.001:
                        continue
                    px0 = max(0, int(x0 * w))
                    py0 = max(0, int(y0 * h))
                    px1 = min(w, int(x1 * w))
                    py1 = min(h, int(y1 * h))
                    if px1 <= px0 or py1 <= py0:
                        continue
                    region = blank[py0:py1, px0:px1]
                    if region.size == 0:
                        continue
                    whiteness = region.mean() / 255.0
                    whiteness_scores.append(whiteness)
            except Exception:
                continue

        if whiteness_scores:
            avg = np.mean(whiteness_scores)
            mn = np.min(whiteness_scores)
            ge98 = sum(1 for w in whiteness_scores if w >= 0.98) / len(whiteness_scores) * 100
            ge95 = sum(1 for w in whiteness_scores if w >= 0.95) / len(whiteness_scores) * 100
            lt90 = sum(1 for w in whiteness_scores if w < 0.90) / len(whiteness_scores) * 100
            print(
                f"{src:25s} {len(whiteness_scores):9d} {avg:9.3f} {mn:9.3f} "
                f"{ge98:5.1f}% {ge95:5.1f}% {lt90:5.1f}%"
            )

    print()


def structural_analysis(by_source, n_samples=20):
    print("=" * 80)
    print("4. STRUCTURAL PRESERVATION")
    print("=" * 80)
    header = (
        f"{'Source':25s} {'PagesOK':>8s} {'AvgOrigKB':>10s} {'AvgBlankKB':>11s} "
        f"{'Strategy':>12s}"
    )
    print(header)
    print("-" * 80)

    for src in sorted(by_source):
        docs = by_source[src]
        sample = random.sample(docs, min(n_samples, len(docs)))

        pages_ok = 0
        total = 0
        orig_sizes = []
        blank_sizes = []

        for jf, doc in sample:
            stem = Path(jf).stem
            subdir = source_subdir(stem)
            blank_pdf = Path(BLANK_DIR) / subdir / f"{stem}.pdf"
            if not blank_pdf.exists():
                continue

            try:
                pdf_doc = fitz.open(str(blank_pdf))
                blank_pages = pdf_doc.page_count
                pdf_doc.close()

                expected = doc.get("page_count", 1)
                total += 1
                if blank_pages == expected:
                    pages_ok += 1

                blank_sizes.append(blank_pdf.stat().st_size / 1024)

                img_rel = doc.get("image_path", "").replace("\\", "/")
                orig_path = Path(img_rel)
                if orig_path.exists():
                    orig_sizes.append(orig_path.stat().st_size / 1024)
            except Exception:
                continue

        strategy = "AcroForm" if src.startswith("synthetic_") else "ImageRedact"
        if total:
            pct = pages_ok / total * 100
            avg_orig = np.mean(orig_sizes) if orig_sizes else 0
            avg_blank = np.mean(blank_sizes) if blank_sizes else 0
            print(
                f"{src:25s} {pct:7.0f}% {avg_orig:10.0f} {avg_blank:11.0f} "
                f"{strategy:>12s}"
            )

    print()


def generate_side_by_side(by_source):
    print("=" * 80)
    print("5. GENERATING SIDE-BY-SIDE COMPARISONS")
    print("=" * 80)

    out_dir = Path("test_output/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(by_source):
        docs = by_source[src]
        # Pick first doc with answers
        chosen = None
        for jf, doc in docs:
            if any(f["role"] == "answer" for f in doc.get("fields", [])):
                chosen = (jf, doc)
                break
        if not chosen:
            continue

        jf, doc = chosen
        stem = Path(jf).stem
        subdir = source_subdir(stem)
        blank_pdf = Path(BLANK_DIR) / subdir / f"{stem}.pdf"
        img_rel = doc.get("image_path", "").replace("\\", "/")
        orig_path = Path(img_rel)

        if not blank_pdf.exists() or not orig_path.exists():
            print(f"  {src}: SKIP (files missing)")
            continue

        try:
            orig = Image.open(orig_path).convert("RGB")

            pdf_doc = fitz.open(str(blank_pdf))
            pix = pdf_doc[0].get_pixmap(dpi=150)
            blank = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_doc.close()

            # Resize to same height
            target_h = 1000
            orig_w = int(orig.width * target_h / orig.height)
            blank_w = int(blank.width * target_h / blank.height)
            orig_r = orig.resize((orig_w, target_h), Image.LANCZOS)
            blank_r = blank.resize((blank_w, target_h), Image.LANCZOS)

            # Create side-by-side with label bar
            gap = 20
            total_w = orig_w + gap + blank_w
            canvas = Image.new("RGB", (total_w, target_h + 30), (255, 255, 255))
            canvas.paste(orig_r, (0, 30))
            canvas.paste(blank_r, (orig_w + gap, 30))

            # Add labels
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except OSError:
                font = ImageFont.load_default()
            draw.text((10, 5), f"ORIGINAL - {src}", fill=(0, 0, 0), font=font)
            draw.text((orig_w + gap + 10, 5), "BLANK FORM", fill=(0, 100, 200), font=font)

            out_path = out_dir / f"compare_{src}.png"
            canvas.save(str(out_path))
            print(f"  {src}: saved {out_path}")

            orig.close()
            blank.close()
        except Exception as e:
            print(f"  {src}: ERROR - {e}")

    print()


if __name__ == "__main__":
    print("Loading field metadata...")
    by_source = load_all_docs()
    print(f"Loaded {sum(len(v) for v in by_source.values())} docs across {len(by_source)} sources\n")

    dataset_overview(by_source)
    pixel_results = pixel_analysis(by_source)
    redaction_quality(by_source)
    structural_analysis(by_source)
    generate_side_by_side(by_source)

    print("Analysis complete.")
