# mm per pixel from dimension annotations

Commands (the dataset root holds `train-00`, `train-01`, `test-00`):

    python -m fpyolo.scale --src <FloorPlanCAD> --n 40 --seed 0 --out reports/scale.json --qa-dir reports/qa/scale
    python -m fpyolo.scale --src <FloorPlanCAD> --n 40 --seed 0 --out reports/scale_anchor.json --mode anchor

Full numbers: `reports/scale.json` (centre rule, main) and `reports/scale_anchor.json` (literal rule).

## Method

- A dimension text is a `<text>` whose stripped content is digits only (at least 2 digits); its value is in mm.
- It is matched to a straight two-point segment on the **same layer** (`inkscape:label` of the enclosing layer group), parallel to
  the text direction (deviation <= 2 degrees, modulo 180), whose midpoint is <= 3 viewBox units from a reference point; the nearest
  midpoint wins. All `transform`s (own and ancestors') are applied.
- **Centre rule (main):** the reference point is the text centre = start anchor + half the estimated width along the text direction,
  width = n_chars x 0.6 x font-size (font-size 2 in the real files). Segments shorter than 1 viewBox unit (ticks, arrow heads) are ignored.
- **Literal rule (comparison):** the reference point is the start anchor itself and there is no minimum length (the rule as first written in the brief).
- mm/px = `value_mm / (segment_length_viewbox * img_w / vb_w)`, with `img_w` from the PNG (1000 or 999).
- Per tile: median of its estimates, and consistency = share of estimates within +-10% of that median. A tile is **consistent** if consistency >= 0.5.
- Overall median, IQR and IQR/median are computed over the medians of **consistent tiles only**. Conclusion: "constant" if IQR/median <= 0.10,
  "varies" otherwise, "inconclusive" if fewer than 20 consistent tiles.
- Sampling: all 15,663 tiles shuffled with seed 0 and scanned in that order; the first 40 with >= 3 estimates are used. If fewer than 20 of them
  were consistent, scanning would continue (same order) until 20 consistent tiles or 400 scanned tiles.

## Result, centre rule (main)

| quantity | value |
|---|---|
| tiles scanned to find 40 with >= 3 estimates | 90 (no extra scanning needed) |
| tiles with >= 3 estimates (sampled) | 40 |
| consistent tiles | 33 of 40 |
| overall median of consistent-tile medians | 10.000 mm/px |
| Q1 / Q3 | 10.000 / 10.004 |
| IQR | 0.0040 |
| IQR / median | 0.0004 (threshold 0.10) |
| **conclusion** | **constant** |

Of all 407 individual estimates in the 40 tiles, 74% lie within 5% of 10.0 mm/px.

### Distribution of consistent-tile medians (33 tiles)

| median mm/px | tiles |
|---|---|
| < 9.5 | 4 |
| 9.5 to 10.5 | 29 |
| > 10.5 | 0 |

Sorted: 9.05, 9.17, 9.20, 9.38, 9.99, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.01, 10.01, 10.01, 10.01, 10.01, 10.02, 10.02, 10.02.

There are no clusters at 5, 20 or 25. 29 of the 33 medians lie within 10.0 +- 0.02 (19 of them are exactly 10.0); the 4 others (9.05, 9.17, 9.20, 9.38) are 6 to 9.5% below 10.
The 7 inconsistent tiles (below) have medians between 2.9 and 10.8 and too few agreeing estimates to be trusted; none shows a different consistent scale.
Consistent tiles by archive: test-00 16, train-01 14, train-00 3 (train-00 has few tiles with >= 3 numeric dimension texts matched).

## Result, literal anchor rule (comparison)

The literal rule on the same first 40 tiles (the numbers from the first version of this report, no consistency filter): overall median
22.54 mm/px, IQR 35.33, IQR/median 1.57, per-tile medians from 2.9 to 10,000 mm/px, only 6 of the 40 tiles consistent.
Run through the same pipeline (`reports/scale_anchor.json`), only 20 consistent tiles were found after scanning 254 tiles (79 tiles with >= 3 estimates);
over these 20: median 12.53 mm/px, IQR 29.48, IQR/median 2.35, conclusion "varies".
Consistent-tile medians under the literal rule: 2.86, 4.69, 10.0, 10.0, 10.0, 10.0, 10.03, 10.23, 10.71, 11.92, 13.13, 20.0, 33.33, 35.0, 37.5, 45.44, 50.0, 55.0, 7250.0, 10000.0.
The literal rule is dominated by wrong matches (short extension ticks next to the start anchor, long chain lines), including tile medians of 7,250 and 10,000 mm/px;
it is not evidence about the scale.

## Conclusion

With text-centre matching, 33 of 40 sampled tiles give internally consistent estimates, and their medians agree with each other very tightly
(median 10.00 mm/px, IQR/median 0.0004, all medians between 9.05 and 10.02). The measurement therefore supports a **constant scale of about
10 mm/px (1 viewBox unit = 100 mm = 10 px for 1000 px tiles)** across all three archives. Limits: the sample is 40 tiles drawn from the tiles that
carry at least three matched numeric dimension texts, so drawings with few or no dimension annotations are not represented (in particular train-00 is thin: 3 consistent tiles);
the 7 inconsistent tiles were excluded rather than explained, and the consistency filter by construction cannot prove that every tile has the same scale. The earlier
"varies" verdict under the literal rule was an artifact of matching noise.

## Visual inspection

Overlays of the first 10 selected tiles (centre rule); red circle = text centre, blue line = matched segment:

- `qa/scale/train-01_0033-0019.png`
- `qa/scale/test-00_0785-0057.png`
- `qa/scale/test-00_0273-0010.png`
- `qa/scale/train-01_0082-0115.png`
- `qa/scale/train-01_1251-0012.png`
- `qa/scale/train-01_0926-0000.png`
- `qa/scale/train-01_1133-0108.png`
- `qa/scale/test-00_0815-0061.png`
- `qa/scale/test-00_0200-0004.png`
- `qa/scale/train-01_1114-0070.png`

## Per-tile results, centre rule (sorted by median)

| tile | estimates | median mm/px | consistency | consistent |
|---|---|---|---|---|
| test-00_0454-0017 | 5 | 2.86 | 0.20 | no |
| train-01_0267-0079 | 7 | 3.33 | 0.14 | no |
| train-01_1133-0108 | 7 | 4.25 | 0.14 | no |
| train-01_0590-0026 | 4 | 6.96 | 0.00 | no |
| test-00_0924-0008 | 5 | 7.14 | 0.20 | no |
| test-00_0273-0010 | 5 | 8.28 | 0.20 | no |
| test-00_1108-0003 | 20 | 9.05 | 0.55 | yes |
| test-00_1211-0006 | 12 | 9.17 | 0.67 | yes |
| train-01_0848-0005 | 15 | 9.20 | 0.53 | yes |
| train-01_0472-0011 | 8 | 9.38 | 0.75 | yes |
| train-01_1114-0070 | 12 | 9.99 | 1.00 | yes |
| test-00_0785-0057 | 3 | 10.00 | 1.00 | yes |
| train-01_0082-0115 | 14 | 10.00 | 1.00 | yes |
| train-01_1251-0012 | 26 | 10.00 | 0.92 | yes |
| train-01_0926-0000 | 13 | 10.00 | 0.77 | yes |
| test-00_0815-0061 | 3 | 10.00 | 0.67 | yes |
| test-00_0200-0004 | 12 | 10.00 | 1.00 | yes |
| test-00_1206-0038 | 6 | 10.00 | 0.83 | yes |
| test-00_0077-0018 | 32 | 10.00 | 0.94 | yes |
| train-01_0035-0054 | 16 | 10.00 | 1.00 | yes |
| test-00_0829-0137 | 12 | 10.00 | 1.00 | yes |
| train-01_0785-0068 | 3 | 10.00 | 1.00 | yes |
| test-00_1008-0191 | 10 | 10.00 | 0.90 | yes |
| train-00_0414-0003 | 22 | 10.00 | 0.73 | yes |
| test-00_1150-0018 | 3 | 10.00 | 1.00 | yes |
| train-00_0087-0010 | 4 | 10.00 | 1.00 | yes |
| train-01_0018-0037 | 3 | 10.00 | 0.67 | yes |
| train-01_0916-0042 | 12 | 10.00 | 0.92 | yes |
| test-00_0881-0016 | 9 | 10.00 | 0.89 | yes |
| train-01_1010-0046 | 10 | 10.00 | 0.90 | yes |
| train-01_0506-0020 | 7 | 10.00 | 0.86 | yes |
| train-01_0816-0014 | 10 | 10.01 | 0.90 | yes |
| test-00_0691-0011 | 8 | 10.01 | 0.88 | yes |
| train-00_0115-0016 | 24 | 10.01 | 0.96 | yes |
| test-00_0209-0004 | 3 | 10.01 | 1.00 | yes |
| test-00_0301-0012 | 5 | 10.01 | 0.80 | yes |
| test-00_0801-0010 | 10 | 10.02 | 1.00 | yes |
| train-01_0372-0018 | 8 | 10.02 | 1.00 | yes |
| test-00_0622-0000 | 14 | 10.02 | 0.86 | yes |
| train-01_0033-0019 | 5 | 10.81 | 0.20 | no |
