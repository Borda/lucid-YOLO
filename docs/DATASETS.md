# Provisioning the datasets

Datasets are never committed to this repository and never downloaded by test code (AGENTS.md sec. 3). Every unit gate runs offline against synthetic fixtures (A26/D12b); only the WPs marked `[DATA]` need a real tree, and a tier acceptance run needs one that passes `lucid-data check`.

This file says where each dataset comes from, what has to be on disk before a run starts, and — since WP-099d — how the two dataset formats lay out on disk and what their annotations actually contain. None of it is restated from memory: the provisioning sections follow what `lucid_yolo.data.check` validates, and the **Dataset formats** reference below follows `lucid_yolo.data.layout`, `.coco`, `.yolo` and `.dota` — those modules are the specification, and every claim below cites the one it came from.

## COCO 2017 (R12)

Automated, because the archives are served from a stable public host (`images.cocodataset.org`) at fixed URLs:

```bash
lucid-data download --data_root /data/coco                       # val + annotations, ~1 GB
lucid-data download --data_root /data/coco --splits '[train,val]'  # adds the 18 GB train archive
lucid-data check    --data_root /data/coco
```

Transfers stream to a `.part` file and are renamed on completion, so a killed run resumes rather than leaving a truncated archive, and an already-extracted split is skipped without touching the network. The official archives publish no authoritative SHA-256 digests, so none are hard-coded; the computed digest of every archive is printed, and an expected one may be enforced per archive with `--sha256`.

## DOTA-v1.0 (R18)

**Manual, and it stays manual.** The distribution has no equivalent of `images.cocodataset.org`: the dataset page offers Google Drive and Baidu Drive folders, which are interactive pages rather than archive URLs. A folder link cannot be resolved to a stable file URL without rendering JavaScript, the Drive large-file path adds a confirmation interstitial, and Baidu Drive requires an account. A downloader written against any of that is a scraper of someone else's UI, which breaks silently and off-repository. So `lucid-data download` covers COCO only, and DOTA provisioning is an operator step.

### Where

Start at the dataset page of the official DOTA site, maintained by the authors of R18:

- <https://captain-whu.github.io/DOTA/dataset.html>

Under **Data Download**, take the **DOTA-v1.0** section — the page also carries DOTA-v1.5 and DOTA-v2.0 sections, and those are different label sets. Each version section offers the same three items, on either host:

| Item | Contains | Needed here |
| -- | -- | -- |
| Training set | images plus their `labelTxt` annotations | yes |
| Validation set | images plus their `labelTxt` annotations | yes |
| Testing images | images only | no |

The testing images are not needed and not downloaded: R18 states that ground truth is released for the training and validation sets *"but not for the testing set. For testing, we are currently building an evaluation server."* The tier is scored on val (WP-063), so nothing here depends on the withheld annotations.

Two cautions from the page as it stands:

- The v1.5 section links the **same** Drive folders as the v1.0 section. Whichever link is followed, take the DOTA-v1.0 label archive, not a v1.5 one: v1.5 adds a sixteenth category (`container crane`), and `lucid_yolo.data.dota` reads the fifteen of R18 in their published order, where the index *is* the class id. A v1.5 label file fails the read with the offending file and line named rather than being silently dropped.
- The Drive folder listings are rendered client-side, so the archive filenames inside them are not quoted here — they have not been read, and a filename copied from memory is the kind of detail that sends an operator to the wrong file. Take what the folder offers for the version and item named above.

### What has to be on disk

Unpack so that each split directory holds an `images/` and a `labelTxt/` directory, with one label file per image sharing its stem:

```text
<dota_root>/
    train/
        images/P0000.png ...
        labelTxt/P0000.txt ...
    val/
        images/P0003.png ...
        labelTxt/P0003.txt ...
```

The archives do not necessarily unpack into this shape; moving directories after extraction is expected. `lucid-data check --data_root <dota_root> --dataset dota` is what decides whether the result is usable: it verifies both directories exist per split, that images and label files pair one-to-one in both directions, and that every object line parses, naming the file and line of the first that does not.

A complete train-plus-val provisioning from the official distribution reports:

```text
  PASS train — 1411 images, 98990 instances, 15 classes
  PASS val — 458 images, 28853 instances, 15 classes
  NOTE totals across ['train', 'val']: 1869 images, 127843 instances, 15 classes
```

Those are measured, not published — R18 gives whole-dataset totals only, and 2,806 minus 1,869 is exactly the 937 images of the withheld testing third. Compare against them, and pass them to make the comparison a gate:

```bash
lucid-data check --data_root <dota_root> --dataset dota \
  --expected_images 1869 --expected_instances 127843
```

**What it does not check by default, and why.** The totals across the checked splits are reported, not required. The published 2,806 images / 188,282 instances / 15 classes describe DOTA-v1.0 *whole*, and R18 splits it as "half of the original images as the training set, 1/6 as validation set, and 1/3 as the testing set" while releasing ground truth for the first two only — so an annotated root holds about two thirds of that and can never sum to it. Requiring those totals by default is what the check did until WP-097, which made it fail on every correct download. The line to read is now `NOTE totals across ...`; once the counts for a given provisioning are known they can be asserted with `--expected_images`, `--expected_instances` and `--expected_classes`, and are then enforced exactly as before.

### Then build the tiles

The tier trains on 1024 px tiles, never on the original tree — DOTA images run to several thousand pixels a side in PNG, which has no random-access region decode, so cropping per sample would decode a whole image to yield one window (WP-094):

```bash
lucid-data build-tiles --root <dota_root> --out <tiles_root> --splits train,val --overlap 512 --workers 8
```

Tiles are PNG by default, matching the lossless source. `--suffix .jpg --quality 92` trades that for decode speed, and the trade is worth weighing: the loader decodes every tile once per epoch and mosaic pulls four source tiles per training sample, so a 64-image step at 1024 px is around 250 decodes and PNG is where a run starves before the GPU does. Annotations are unaffected by the choice.

The tiles are a build artifact and are never committed. The output is a COCO container with quadrilateral rings, in the layout `obb_smoke.yaml` names, so training and evaluation read it through the same dataset class as COCO.

Overlap is a real choice, not a default to accept unread. Pixels are amplified by `(patch / (patch - overlap))²`: 512 px of overlap is R18's own crop stride and costs 4.0x, while the 200 px of A21 costs about 1.55x. Larger overlap means fewer objects severed by a tile edge and a longer epoch.

## Dataset formats: the layouts they publish, and what one annotation contains

This is a reader's map, not a restatement: every claim below cites the module that enforces it, and none of it substitutes for reading that module when something is unclear. It answers two questions the provisioning sections above do not — how a reader finds its images and annotations under a dataset root, and what one annotation actually contains, per task and per container.

### Directory conventions

`lucid_yolo.data.layout` names the directory conventions a root may satisfy, as two separate tables — `CANDIDATES` for a COCO-format root (an images directory beside an `instances_*.json`) and `YOLO_CANDIDATES` for a YOLO-format one (a `data.yaml` beside per-split `images`/`labels` directories). They stay two tables rather than becoming rows of one because a YOLO split's annotations are a *directory*, not a file: the "both halves exist" predicate that lets `CANDIDATES` fall back safely would have to become an existence check that accepts either kind of thing, and a root satisfying both conventions would then resolve to whichever table came first — handing a labels directory to the COCO reader or an `instances_*.json` to the YOLO one (`lucid_yolo/data/layout.py:28-40`).

`CANDIDATES`, tried in this order for a split (`val` below), the first row whose directory *and* file both exist winning (`lucid_yolo/data/layout.py:78-83`, `resolve_split`, `:156-163`):

| Convention | Images directory | Annotation file |
| -- | -- | -- |
| COCO 2017 | `val2017/` | `annotations/instances_val2017.json` |
| plain COCO / tiled | `val/` | `annotations/instances_val.json` |
| images-subdirectory | `images/val/` | `annotations/instances_val.json` |
| per-split export | `val/` | `val/_annotations.coco.json` |

`YOLO_CANDIDATES`, the same rule over two rows, both halves directories so both are tested with `is_dir()` (`lucid_yolo/data/layout.py:89-92`, `resolve_yolo_split`, `:203-210`):

| Convention | Images directory | Labels directory |
| -- | -- | -- |
| per-split export | `val/images/` | `val/labels/` |
| split-subdirectory | `images/val/` | `labels/val/` |

Neither table raises on a miss: when nothing matches, `resolve_split`/`resolve_yolo_split` return the first row unchanged, leaving the "no such file" to the reader that actually opens the path (`lucid_yolo/data/layout.py:20-26`). `detect_layout` — the one caller with no reader of its own to ask, `DetectionDataModule` deciding which reader a bare `data_root` calls for — is stricter: it probes both tables in full and raises if neither is satisfied (naming every path tried) or if **both** are, since across the two tables the loser is a different label space and a different image set, not another spelling of the same reader (A63; `lucid_yolo/data/layout.py:213-282`).

DOTA-v1.0's own `images/` + `labelTxt/` layout, documented above under **DOTA-v1.0 (R18)**, is neither of these tables — it is read directly by `lucid_yolo.data.dota`, not through `lucid_yolo.data.layout`, and only becomes a COCO-format root (one of the rows above) after `lucid-data build-tiles` writes it out.

A YOLO root's own `data.yaml` outranks `YOLO_CANDIDATES` for any split it names: `YoloDataConfig.images_dir` resolves the file's own entry — the literal reading against the file's directory first, then the same entry with leading `..` components dropped, since the published export writes `../train/images` for a directory that actually sits beside the yaml — and the convention table above applies only to a split the file leaves unnamed (`lucid_yolo/data/yolo.py:211-253`, `from_root`, `:308-363`).

### What one annotation contains

| Task | COCO container | YOLO container |
| -- | -- | -- |
| detect | `bbox`: pixel `[x, y, w, h]`, converted to `xyxy` on read | five normalized fields: `cls cx cy w h` |
| segment | first ring of `segmentation`: pixel `(P, 2)` polygon, `P >= 3` | not expressible — the format carries no rings |
| obb | four-point `segmentation` ring, plus four non-schema keys (A53) | nine normalized fields: `cls x1 y1 x2 y2 x3 y3 x4 y4` |

**COCO detect.** An annotation's `bbox` is `[x, y, w, h]` in pixels, top-left corner plus width and height; the reader converts it to this project's `xyxy` convention and clamps it to the image bounds (`lucid_yolo/data/coco.py:9-12`, `_xywh_to_xyxy`, `:436-452`). On a 100x50px image, `"bbox": [10, 5, 20, 10]` reads as `xyxy = [10, 5, 30, 15]` (`x2 = x + w = 30`, `y2 = y + h = 15`).

**COCO segment.** `segmentation` is a list of flat polygon rings; only the *first* ring of an ordinary (list-typed) annotation is kept, reshaped to `(P, 2)`, and only when it has at least three points — a run-length dict, an empty list, or a shorter ring makes the reader skip the annotation entirely, along with any `iscrowd=1` one (`lucid_yolo/data/coco.py:14-22`, `_parse_ring`, `:412-433`). The same object as above, `"segmentation": [[10, 5, 30, 5, 30, 15, 10, 15]]`, parses to the ring `[[10, 5], [30, 5], [30, 15], [10, 15]]` — the envelope of that ring is the same box `bbox` gives above, though the (non-oriented) reader does not derive one from the other here; it does on the oriented path below.

**COCO obb.** Under `oriented=True` the `segmentation` ring is read as a rotated box's quadrilateral rather than a free polygon: it must be exactly four points, fitted to a canonical long-edge box `(cx, cy, w, h, theta)` (`lucid_yolo/data/rotated_geom.py:4`), and `boxes` is recomputed as the **envelope of that same ring** rather than read from `bbox` — so `boxes[i]` and `rboxes[i]` describe one object by construction (`lucid_yolo/data/coco.py:24-40`, `_oriented_targets`, `:312-352`). The module's own worked example: the ring `[[3, 2], [7, 2], [7, 4], [3, 4]]` yields `boxes = [[3.0, 2.0, 7.0, 4.0]]` and `rboxes[0] = [5.0, 3.0, 4.0, 2.0, 0.0]` (`lucid_yolo/data/coco.py:334-338`).

A53 adds four keys the COCO schema has none for, written by `lucid-data build-tiles` and read back by name — never by a standard COCO reader, which sees an ordinary detection set and ignores all four:

| Key | Scope | Written at | Read by this project's reader |
| -- | -- | -- | -- |
| `difficult` | per annotation | `annotation_records` (`lucid_yolo/data/tiles.py:212`) | yes — forwarded onto `Targets.difficult`; absent reads as `False` (`lucid_yolo/data/coco.py:308`) |
| `visible_fraction` | per annotation | `annotation_records` (`lucid_yolo/data/tiles.py:213`) | no — not read by `CocoDetectionDataset` |
| `source_image` | per image | `_image_record` (`lucid_yolo/data/tiles.py:473`) | no |
| `window` | per image | `_image_record` (`lucid_yolo/data/tiles.py:474`), a `(x0, y0, x1, y1)` pixel box (`lucid_yolo/data/tiles.py:150`) | no |

**YOLO detect.** Each non-blank label line is five whitespace-separated fields, class index first, the rest `cx cy w h` normalized by the image's own width/height (`lucid_yolo/data/yolo.py:12-18`). At `width=100, height=40`, the row `1 0.5 0.5 0.5 0.25` denormalizes to `xyxy = [25.0, 15.0, 75.0, 25.0]` — `cx=50, w=50 -> x1=25, x2=75`; `cy=20, h=10 -> y1=15, y2=25` (the module's own doctest, `lucid_yolo/data/yolo.py:470-478`).

**YOLO obb.** The oriented row is exactly nine fields — class index, then four normalized `(x, y)` corners — and a ten-field row (R18's own label lines append a trailing `difficult` flag the normalized variant has no published spelling for) is rejected naming the file and line, never assumed away (A56; `lucid_yolo/data/yolo.py:73-76`, `_parse_row`, `:655-679`). At `width=10, height=10`, the row `0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3` denormalizes to the quadrilateral `[[1, 1], [5, 1], [5, 3], [1, 3]]`, giving `boxes = [[1.0, 1.0, 5.0, 3.0]]` and `rboxes[0] = [3.0, 2.0, 4.0, 2.0, 0.0]` (the module's own doctest, `lucid_yolo/data/yolo.py:887-892`).

**YOLO segment.** Not expressible. A label line carries five or nine normalized numbers and never a polygon ring — the format has no rings to rasterize a mask from (`lucid_yolo/data/yolo.py:12-18`). `DetectionDataModule` refuses `mask_targets=True` on a YOLO root rather than rasterizing empty masks and reporting a plausible detection number off a segmentation head trained on nothing: "`mask_targets=True is unavailable on a YOLO root: the format carries no per-instance polygon rings, so there is nothing to rasterise masks from`" (`lucid_yolo/ptl/datamodule.py:931-950`).

## Where a provisioned tree belongs on a hosted runtime

A hosted runtime is ephemeral and its local disk goes with it, so the two halves of a run belong in different places.

**The dataset and the tiles stay on the runtime's local disk** (`/content/...`), not on a mounted Drive. Every epoch reads every tile, and Drive is a network filesystem mounted through FUSE — putting the training set there turns each sample fetch into a round trip and starves the loader workers the epoch-recycling path exists to keep busy.

**Checkpoints and logs go the other way**, onto storage that outlives the runtime. That is `--trainer.default_root_dir`, and it belongs to the launch rather than to provisioning: docs/TRAINING.md carries it, along with the three tiers' commands.

## License terms, and what they constrain

DOTA is not permissively licensed. The dataset page states:

> All images and their associated annotations in DOTA can be used for academic purposes only, but any commercial use is prohibited.

and separately that use of the Google Earth imagery within it must respect the Google Earth terms of use. This project's dependency policy — permissive licenses only, never AGPL or a commercial license — is about code that ships inside the wheel, and DOTA is neither shipped nor vendored. What the terms do constrain is what may be derived from it: releases publish no trained weights (D14), datasets are never committed (AGENTS.md sec. 3), and the oriented tier's outputs are the reported metrics and the frozen goldens. Anyone reusing this repository for anything other than academic work should read the terms before provisioning DOTA at all.

COCO 2017 carries its own terms, published on `cocodataset.org` and not restated here — the annotations and the images they describe are covered separately, and the wording on that page is the authority. Neither dataset is redistributed by this repository.
