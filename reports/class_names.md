# Thing class names (FloorPlanCAD `semantic-id` 3–32)

Names in `configs/classes_v0.yaml` are the dataset authors' own `coco_vis` labels
(SymPoint vocabulary) in snake_case. **All 30 are confirmed; none is uncertain.**

## Method

1. `python -m fpyolo.qa contact-sheet --src <FloorPlanCAD> --out reports/qa/classes --per-class 12 --seed 0`
   indexes every SVG of `train-00`, `train-01` and `test-00` (15,663 tiles), then
   draws up to 12 objects per `semantic-id`, one per tile, preferring distinct
   source drawings and objects fully inside the tile. Each pair in
   `reports/qa/classes/sid_XX.png` is *[ink + analytic box | coco_vis + the same
   box converted to coco_vis pixels]*; `sid_XX.txt` lists the tiles, boxes and
   detected coco_vis frames.
2. coco_vis pixel = frame origin + image pixel × scale. The frame (matplotlib
   axes) is detected per image as the outermost rows/columns holding at least
   half the top dark-pixel count. On 606 random coco_vis images it was always
   (x0, y0, x1, y1) = (255, 240, 1795, 1780) for 1000×1000 tiles, i.e. origin
   (255, 240) and **1.540 coco_vis px per image px**; (254, 240, 1796, 1780) for
   1000×999 and (256, 240, 1794, 1780) for 999×1000 tiles (equal-aspect axes).
   Verified visually: on every pair the converted (blue) box lies on the
   outline of the authors' coloured polygon.
3. coco_vis prints each label at its object's box top-left corner. Because
   neighbouring labels overlap, the label attributed to an object is the one
   anchored at the blue box's top-left. Labels were read on the contact sheets
   (sids 3–8, 21) and, for all 30 sids, on native-resolution coco_vis crops
   around that corner (an uncommitted reading aid; downscaled sheets make
   some labels small).
4. Every viewed instance of a class showed the same label, so no class needed
   the 3-extra-instance tie-break. Classes 7 and 8 have only 7 and 8 objects in
   the whole dataset (5 and 8 distinct tiles), all of which were viewed.

## Evidence

"Objects" counts every (tile, instance-id) in all 15,663 tiles. "Common layers"
are the top-3 inkscape layer labels (`<g inkscape:label>`) of the objects'
elements, counted once per object, over all tiles.

| sid | schema_id | name | group | coco_vis label (agreeing/viewed) | objects | coco_vis tiles viewed | common layers (objects) |
|---|---|---|---|---|---|---|---|
| 3 | 1 | `single_door` | door | "single door" on 12/12 | 23159 | train-01/0308-0007, test-00/1060-0013, train-00/0529-0002, test-00/1314-0022, train-00/0260-0002, train-00/0271-0014, train-01/0745-0017, train-00/0348-0024, train-00/0236-0012, train-00/0013-0004, train-01/0433-0048, test-00/0782-0040 | WINDOW (11258); DOOR_FIRE (1116); A-DOOR (649) |
| 4 | 2 | `double_door` | door | "double door" on 12/12 | 12372 | train-01/0797-2262, train-00/0026-0031, test-00/0355-0029, train-01/0591-0594, train-01/0769-0005, train-01/0026-0043, test-00/0280-0002, test-00/0300-0022, train-00/0372-0066, train-01/0890-0010, train-01/0846-0006, train-00/0141-0022 | WINDOW (5080); DOOR_FIRE (2911); A-WIN (944) |
| 5 | 3 | `sliding_door` | door | "sliding door" on 12/12 | 5536 | train-00/0166-0008, train-00/0014-0025, train-00/0514-0003, train-01/0805-0006, train-01/0074-0026, train-01/0415-0009, test-00/0195-0017, train-00/0209-0001, train-01/1291-0012, train-00/0153-0004, train-00/0494-0009, test-00/0919-0007 | WINDOW (3568); DOOR_FIRE (478); A-DOOR (157) |
| 6 | 4 | `folding_door` | door | "folding door" on 12/12 | 83 | train-01/0684-0016, test-00/0654-0001, test-00/0129-0017, test-00/0461-0054, test-00/0719-0037, test-00/0227-0002, train-01/1172-0055, train-00/0283-0014, test-00/0091-0013, test-00/0360-0020, train-01/0101-0055, test-00/0302-0002 | WINDOW (59); WALL (18); WALL-MOVE (1) |
| 7 | 5 | `revolving_door` | door | "revolving door" on 5/5 | 7 | train-00/0155-0004, train-00/0364-0028, test-00/0650-0105, train-01/0090-0092, train-00/0018-0029 | AED_180233_Hotel_L1-L5_Plan$0$214 CURTAIN WALL (1); 家具-省院 (1); H-L1$0$XFINE-铝板 (1) |
| 8 | 6 | `rolling_door` | door | "rolling door" on 8/8 | 8 | train-01/0090-0113, train-01/1262-0693, test-00/0043-0044, train-01/1123-0008, train-01/1039-0308, train-00/0123-0013, train-00/0340-0612, test-00/0652-0093 | WINDOW (3); DOOR_FIRE (2); x消防-防火卷帘 (2) |
| 9 | 7 | `window` | window | "window" on 12/12 | 16459 | train-01/0724-0028, train-01/1192-0004, train-01/0533-0021, train-01/0819-0006, train-01/0757-0058, test-00/1314-0036, test-00/0136-0003, test-00/0357-0007, train-01/0980-0015, test-00/1152-0029, test-00/0800-4437, train-01/0057-0018 | WINDOW (10380); window (920); A-GLAZ-MULL (495) |
| 10 | 8 | `bay_window` | window | "bay window" on 12/12 | 485 | train-01/1298-0003, train-00/0252-0003, train-01/0391-0017, test-00/0864-0026, test-00/1167-0001, train-01/0214-0011, train-00/0328-0005, test-00/0253-0027, test-00/0749-0021, train-01/1030-0004, train-00/0243-0003, train-01/0735-0010 | WINDOW (380); S38-10$0$A-WINDOW (17); 建-GLAZ (14) |
| 11 | 9 | `blind_window` | window | "blind window" on 12/12 | 1040 | train-00/0283-0007, train-00/0300-0015, test-00/0090-0173, train-01/0717-0002, test-00/0109-0011, train-00/0215-0004, test-00/0437-0041, test-00/0958-0042, train-00/0306-0051, test-00/0970-0010, train-00/0018-0009, train-01/1274-0019 | WINDOW (696); A-WINDOW (69); A-WIND (54) |
| 12 | 10 | `opening_symbol` | window | "opening symbol" on 12/12 | 998 | train-00/0466-0002, train-00/0455-0002, train-00/0571-0008, train-00/0486-0004, test-00/0987-0032, train-00/0463-0013, train-01/0097-0359, train-01/0110-0332, train-00/0465-0011, test-00/0175-0017, train-00/0220-0003, test-00/0595-0017 | 门窗开启方向 (172); WINDOW (96); A-WINDOW (62) |
| 13 | 11 | `sofa` | furniture | "sofa" on 12/12 | 1427 | test-00/0097-0413, train-01/0014-0051, train-01/0099-0021, train-01/1152-0008, train-01/0893-0035, test-00/0452-0107, train-00/0188-0014, train-00/0509-0005, test-00/0051-0009, train-01/0717-0032, train-01/0949-0016, test-00/0050-0032 | A-FURN (123); JZ-家具 (122); S38-10$0$A-FURNITURE (92) |
| 14 | 12 | `bed` | furniture | "bed" on 12/12 | 3099 | train-01/1290-0021, train-01/1172-0062, train-01/0008-0052, train-01/1323-0010, train-00/0494-0003, test-00/0018-0011, train-00/0537-0004, train-00/0442-0003, test-00/0864-0018, train-00/0429-0009, train-00/0439-0022, train-00/0486-0013 | A-FURN(家具) (943); 洁具 (205); A-Furniture (152) |
| 15 | 13 | `chair` | furniture | "chair" on 12/12 | 9148 | train-00/0006-0003, test-00/0036-0087, train-01/0925-0017, train-01/1022-0004, test-00/0132-0014, train-00/0153-0007, train-01/0063-0062, test-00/0019-0012, train-01/0035-0059, test-00/0030-0018, train-00/0199-0016, test-00/0800-4403 | 墙体$0$墙体(新)$0$$0$A-FURN (1267); 室内家具 (780); 00家具 (762) |
| 16 | 14 | `table` | furniture | "table" on 12/12 | 4633 | train-01/0078-0014, test-00/0800-4410, train-01/0035-0035, test-00/0060-0059, train-01/1291-0007, test-00/0132-0014, train-00/0011-0011, train-01/0063-0062, test-00/0089-0091, test-00/0595-0012, train-01/0092-0043, train-00/0247-0003 | 墙体$0$墙体(新)$0$$0$A-FURN (872); AR_FUR (622); 室内家具 (476) |
| 17 | 15 | `tv_cabinet` | furniture | "tv cabinet" on 12/12 | 1241 | train-00/0456-0005, test-00/0957-0057, train-00/0462-0009, train-00/0495-0007, test-00/1288-0010, train-00/0330-0002, test-00/0856-0021, train-00/0428-0003, train-00/0426-0005, train-00/0004-0002, test-00/0759-0007, train-00/0002-0005 | 1-家具线 (188); hx-138$0$家具 (117); A-FURN (112) |
| 18 | 16 | `wardrobe` | furniture | "wardrobe" on 12/12 | 1737 | train-00/0509-0005, train-01/1160-0080, train-00/0491-0004, train-01/1326-0007, test-00/0001-0055, train-00/0510-0011, train-01/0008-0042, test-00/1106-0013, train-01/1099-0029, train-01/0595-0022, train-01/1175-0027, test-00/1158-0004 | 建-家具 (212); JZ-家具 (149); hx-90md1$0$C03平面-活动家具-细节 (149) |
| 19 | 17 | `cabinet` | furniture | "cabinet" on 12/12 | 6600 | train-00/0147-0007, train-00/0194-0012, train-00/0189-0016, test-00/1134-0028, train-00/0441-0007, train-01/0913-0005, test-00/1040-0046, train-00/0512-0003, test-00/0049-0246, train-00/0463-0009, test-00/0957-0035, train-00/0325-0004 | 1-家具线 (708); A-FURN(家具) (382); 建-家具 (347) |
| 20 | 18 | `refrigerator` | sanitary-kitchen | "refrigerator" on 12/12 | 1276 | train-01/0341-0002, test-00/1035-0013, train-00/0463-0007, test-00/0342-0016, train-00/0188-0016, test-00/0875-0035, train-00/0503-0008, test-00/0687-0012, train-01/0311-0002, test-00/0128-0103, test-00/0228-0022, test-00/0806-0061 | LVTRY (321); 厨卫 (115); 卫生器具 (83) |
| 21 | 19 | `airconditioner` | appliance-hvac | "airconditioner" on 12/12 | 2553 | train-01/0463-0015, test-00/0012-0001, test-00/0734-0105, train-00/0476-0008, train-01/0748-0007, train-00/0327-0004, test-00/0077-0059, test-00/1062-0059, test-00/1265-0009, train-00/0021-0015, train-01/0440-0052, test-00/0414-0019 | 空调 (382); 空调机 (337); AC (208) |
| 22 | 20 | `gas_stove` | sanitary-kitchen | "gas stove" on 12/12 | 1523 | train-01/1277-0003, test-00/0448-0018, train-00/0199-0013, train-01/0700-0004, test-00/0180-0008, train-01/1050-0003, train-01/0649-0019, train-00/0136-0019, test-00/0489-0004, test-00/0717-0017, test-00/0416-0002, train-00/0316-0420 | LVTRY (290); 厨卫 (281); A-构件-洁具 (130) |
| 23 | 21 | `sink` | sanitary-kitchen | "sink" on 12/12 | 8393 | test-00/0218-0018, test-00/0523-0105, test-00/0431-0026, train-01/0635-0022, train-00/0311-0449, test-00/0885-0010, train-01/0024-0036, train-01/1280-0036, train-00/0238-0007, train-01/0890-0017, train-00/0546-0001, test-00/0084-0013 | LVTRY (1053); 厨卫 (552); MS (431) |
| 24 | 22 | `bath` | sanitary-kitchen | "bath" on 12/12 | 1941 | train-00/0324-0006, train-01/0608-0007, train-00/0308-0449, train-00/0492-0010, train-01/1288-0007, train-00/0476-0008, train-01/0919-0027, train-01/0545-0000, train-01/0957-0059, test-00/0593-0005, train-00/0239-0048, test-00/0466-0032 | 厨卫 (195); LVTRY (184); MS (137) |
| 25 | 23 | `bath_tub` | sanitary-kitchen | "bath tub" on 12/12 | 766 | test-00/0365-0022, test-00/0561-0003, train-01/0942-0007, train-00/0261-0003, test-00/1101-0023, test-00/0337-0007, train-01/0893-0035, train-01/0533-0012, train-01/0402-0017, test-00/1031-0007, train-01/0364-0006, train-01/1197-0035 | LVTRY (231); FUR (100); A-FIXT (51) |
| 26 | 24 | `washing_machine` | sanitary-kitchen | "washing machine" on 12/12 | 1319 | test-00/0232-0013, train-01/0398-0007, train-01/0431-0011, train-01/0533-0012, train-00/0495-0011, train-01/0673-0017, test-00/0925-0022, test-00/0517-0008, test-00/0193-0003, test-00/0364-0026, train-00/0326-0003, test-00/1106-0019 | 厨卫 (303); LVTRY (165); 5fix (131) |
| 27 | 25 | `squat_toilet` | sanitary-kitchen | "squat toilet" on 12/12 | 2496 | test-00/1253-0015, train-00/0438-0007, train-00/0091-0009, test-00/0556-0009, test-00/1044-0078, train-00/0536-0019, train-00/0396-0095, train-00/0376-0015, train-01/0551-0098, train-01/0349-1017, train-00/0522-0012, train-01/0059-0013 | LVTRY (669); 洁具 (253); FL121POP_A10-D-5-1(除示范区)平面$0$AE-EQPM-TOLT (166) |
| 28 | 26 | `urinal` | sanitary-kitchen | "urinal" on 12/12 | 1793 | test-00/1088-0018, train-00/0390-0080, test-00/1037-0008, test-00/1315-0014, test-00/0815-0066, train-01/1041-0015, train-01/1182-0015, train-01/1321-0066, train-00/0011-0043, train-00/0392-0051, train-01/0067-0073, test-00/1039-0388 | MS (508); LVTRY (243); 1-FG$0$家具 (116) |
| 29 | 27 | `toilet` | sanitary-kitchen | "toilet" on 12/12 | 5035 | train-00/0526-0007, test-00/0400-0131, test-00/0456-0008, train-01/1316-0073, test-00/0825-0015, train-00/0549-0000, test-00/0860-0052, train-00/0037-0033, train-00/0191-0015, train-01/0164-0022, test-00/0326-0061, test-00/1319-0029 | 1-FG$0$家具 (449); LVTRY (415); 厨卫 (257) |
| 30 | 28 | `stairs` | vertical-circulation | "stairs" on 12/12 | 5014 | test-00/0944-0013, train-00/0328-0005, train-00/0351-0013, test-00/0134-0145, train-01/0433-0048, test-00/0957-0023, train-01/1309-0015, train-01/0409-0022, test-00/0948-0018, train-00/0179-0013, test-00/0553-0017, test-00/0526-0110 | STAIR (2366); 5s-line (387); A-STAIR (368) |
| 31 | 29 | `elevator` | vertical-circulation | "elevator" on 12/12 | 3718 | test-00/0632-0007, train-00/0158-0006, train-00/0237-0017, train-01/1296-0006, test-00/0049-0232, test-00/0119-0052, test-00/0373-0009, train-01/1098-0035, test-00/0314-0006, train-01/1289-0010, train-01/0786-0036, test-00/0624-0025 | EVTR (863); 建-楼梯 (760); STAIR (637) |
| 32 | 30 | `escalator` | vertical-circulation | "escalator" on 12/12 | 360 | test-00/0554-0196, train-01/0346-0644, train-01/0493-0330, train-00/0290-0026, train-00/0371-0041, test-00/0407-0051, test-00/0400-0116, train-01/0349-0775, train-01/0851-0676, train-01/1039-0680, train-00/0378-0021, test-00/0273-0018 | STAIR (142); A-STAIR (72); Stair【楼梯、电梯】 (56) |

## Notes

- **sid 21 = airconditioner is confirmed** (12/12; layers 空调 / 空调机 / AC).
- The order is SymPoint's thing order shifted by 2 (SymPoint 1 = single door →
  sid 3), except that sids 20–23 hold refrigerator, airconditioner, gas stove,
  sink where a plain shift would give gas stove, sink, refrigerator,
  airconditioner. Every name above was read, not inferred from the order.
- "bath" (sid 24) is drawn as shower heads / shower trays; "bath tub" (sid 25)
  is the tub. The authors' names are kept as-is.
- Groups follow spec §6.2: `appliance-hvac` is reserved for airconditioner, so
  refrigerator, gas stove and washing machine stay in `sanitary-kitchen`;
  opening symbol (layer 门窗开启方向, "door/window opening direction") stays in
  `window`. No group needed changing.
