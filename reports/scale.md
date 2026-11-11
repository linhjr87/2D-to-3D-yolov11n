# mm per pixel from dimension annotations

Measured with `python -m fpyolo.scale --src <FloorPlanCAD> --n 40 --seed 0 --out reports/scale.json --qa-dir reports/qa/scale`.
Full numbers: `reports/scale.json`.

## Method

- A dimension text is a `<text>` whose stripped content is digits only (at least 2 digits); its value is in mm.
- It is matched to a straight two-point segment on the **same layer** (`inkscape:label` of the enclosing layer group),
  parallel to the text direction (deviation <= 2 degrees, modulo 180), whose midpoint is <= 3 viewBox units from the text
  anchor. If several segments qualify, the one with the nearest midpoint is used. All `transform`s (own and ancestors') are applied.
- mm/px = `value_mm / (segment_length_viewbox * img_w / vb_w)`, with `img_w` from the PNG (1000 or 999).
- Tile selection: all 15,663 tiles (the three archives) shuffled with seed 0; the first 40 with >= 3 estimates are used
  (119 tiles scanned to find them). Per tile: the median of its estimates. Across tiles: median and IQR (numpy linear percentiles).

## Result

| quantity | value |
|---|---|
| tiles used | 40 |
| overall median of per-tile medians | 22.54 mm/px |
| Q1 / Q3 | 11.33 / 46.65 |
| IQR | 35.33 |
| IQR / median | 1.57 |
| threshold for "constant" | <= 0.10 |

**Conclusion: varies.** IQR/median is about 1.6, far above 0.10; the per-tile medians span
2.9 to 10,000 mm/px (10th to 90th percentile: 9.8 to 3,200). Do not assume one mm/px for the whole dataset.

## Caveats (read before quoting the number)

- The estimates are noisy. Per-tile spreads are large too: within one tile the estimates often differ by a factor of 2 to 10.
  The matching rule is applied exactly as specified and not tuned, so some matches are wrong. Inspected overlays show that the
  nearest parallel same-layer segment is often a short extension tick or the long overall chain line, not the dimension segment
  the text labels. The text anchor is the text start (baseline left), not its centre.
- Overlays show the raw matched pairs: red circle = text anchor, blue line = matched segment.
- Values above ~200 mm/px (6 of 40 tile medians) are not physically plausible for these drawings and come from wrong matches
  (or from drawings whose text is in other units); they inflate the IQR. A pure-noise explanation is not proven, though:
  only 16% of all 236 estimates lie within 5% of 10.0 mm/px (the most common single value, in 22 of 40 tiles), and the other
  tiles have clusters at other values (e.g. 5, 20, 25, 30, 50 mm/px), which is consistent with drawings at different scales.
- Conclusion for the final report: with this rule and this data, mm/px is **not demonstrably constant**; a tighter matching
  rule would be needed to say more.

## Visual inspection

Overlays of the first 10 selected tiles:

- `qa/scale/train-01_0033-0019.png`
- `qa/scale/test-00_0273-0010.png`
- `qa/scale/train-01_0082-0115.png`
- `qa/scale/train-01_1251-0012.png`
- `qa/scale/train-01_0926-0000.png`
- `qa/scale/train-00_0108-0052.png`
- `qa/scale/test-00_0200-0004.png`
- `qa/scale/test-00_0077-0018.png`
- `qa/scale/train-01_0035-0054.png`
- `qa/scale/train-01_0267-0079.png`

## Per-tile medians (sorted)

| tile | estimates | median mm/px |
|---|---|---|
| train-01_0082-0115 | 8 | 2.86 |
| test-00_0454-0017 | 4 | 3.64 |
| train-01_0379-0023 | 6 | 5.75 |
| test-00_0273-0010 | 5 | 8.28 |
| train-01_0033-0019 | 5 | 10.00 |
| test-00_0077-0018 | 13 | 10.00 |
| train-00_0414-0003 | 14 | 10.00 |
| test-00_0431-0026 | 3 | 10.00 |
| train-01_0506-0020 | 7 | 10.00 |
| test-00_0429-0011 | 3 | 10.23 |
| test-00_0987-0037 | 5 | 11.69 |
| test-00_0924-0008 | 3 | 12.51 |
| train-01_0267-0079 | 3 | 13.13 |
| train-00_0369-0040 | 3 | 13.19 |
| train-00_0244-0003 | 5 | 13.57 |
| train-01_1269-0304 | 4 | 15.17 |
| train-00_0196-0018 | 6 | 16.83 |
| train-01_0035-0054 | 5 | 17.63 |
| train-01_0916-0042 | 7 | 20.00 |
| train-01_1251-0012 | 7 | 21.07 |
| train-00_0115-0016 | 14 | 24.00 |
| test-00_0829-0137 | 3 | 25.00 |
| test-00_0203-0082 | 3 | 26.67 |
| test-00_1008-0191 | 4 | 26.67 |
| train-01_0848-0005 | 8 | 30.00 |
| train-00_0321-0016 | 3 | 32.50 |
| train-00_0514-0011 | 3 | 33.33 |
| train-00_0317-0008 | 14 | 36.25 |
| test-00_0844-0179 | 4 | 37.50 |
| train-01_1159-0013 | 5 | 46.00 |
| train-00_0108-0052 | 3 | 48.60 |
| train-01_0926-0000 | 7 | 50.00 |
| train-01_0472-0011 | 6 | 50.00 |
| test-00_1213-0057 | 7 | 70.00 |
| test-00_0622-0000 | 7 | 1750.00 |
| train-01_0816-0014 | 5 | 3000.00 |
| test-00_0200-0004 | 9 | 5000.00 |
| test-00_0881-0016 | 6 | 6000.00 |
| test-00_0691-0011 | 6 | 6000.00 |
| test-00_0801-0010 | 3 | 10000.00 |
