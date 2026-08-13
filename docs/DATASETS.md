# Provisioning the datasets

Datasets are never committed to this repository and never downloaded by test code (AGENTS.md sec. 3). Every unit gate runs offline against synthetic fixtures (A26/D12b); only the WPs marked `[DATA]` need a real tree, and a tier acceptance run needs one that passes `lucid-data check`.

This file says where each dataset comes from and what has to be on disk before a run starts. It does not restate the layout that `lucid_yolo.data.check` validates — that module is the specification, and this file follows it.

## COCO 2017 (R12)

Automated, because the archives are served from a stable public host (`images.cocodataset.org`) at fixed URLs:

```bash
lucid-data download --data_root /data/coco                       # val + annotations, ~1 GB
lucid-data download --data_root /data/coco --splits train val    # adds the 18 GB train archive
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

**Known gap in that check, as of 0.3.0.dev0.** Its totals clause compares the summed image and instance counts against the published 2,806 images / 188,282 instances, and those totals describe DOTA-v1.0 *whole* — R18 splits the images as "half of the original images as the training set, 1/6 as validation set, and 1/3 as the testing set", so a correctly provisioned train-plus-val root holds about two thirds of them and reports two count problems it cannot avoid. The per-split layout, pairing and parse results above it are the part to read; the totals line is expected to fail until the clause is corrected. `check_dota_root` already takes `expected_images` / `expected_instances` parameters for a partially provisioned root, but the CLI exposes no flag for them, so there is no workaround from the command line today.

### Then build the tiles

The tier trains on 1024 px tiles, never on the original tree — DOTA images run to several thousand pixels a side in PNG, which has no random-access region decode, so cropping per sample would decode a whole image to yield one window (WP-094):

```bash
lucid-data build-tiles --root <dota_root> --out <tiles_root> --splits train,val --overlap 512
```

The tiles are a build artifact and are never committed. The output is a COCO container with quadrilateral rings, in the layout `obb_smoke.yaml` names, so training and evaluation read it through the same dataset class as COCO.

Overlap is a real choice, not a default to accept unread. Pixels are amplified by `(patch / (patch - overlap))²`: 512 px of overlap is R18's own crop stride and costs 4.0x, while the 200 px of A21 costs about 1.55x. Larger overlap means fewer objects severed by a tile edge and a longer epoch.

## Where a provisioned tree belongs on a hosted runtime

A hosted runtime is ephemeral and its local disk goes with it, so the two halves of a run belong in different places.

**The dataset and the tiles stay on the runtime's local disk** (`/content/...`), not on a mounted Drive. Every epoch reads every tile, and Drive is a network filesystem mounted through FUSE — putting the training set there turns each sample fetch into a round trip and starves the loader workers the epoch-recycling path exists to keep busy.

**Checkpoints and logs go the other way**, onto storage that outlives the runtime. That is `--trainer.default_root_dir`, and it belongs to the launch rather than to provisioning: docs/TRAINING.md carries it, along with the three tiers' commands.

## License terms, and what they constrain

DOTA is not permissively licensed. The dataset page states:

> All images and their associated annotations in DOTA can be used for academic purposes only, but any commercial use is prohibited.

and separately that use of the Google Earth imagery within it must respect the Google Earth terms of use. This project's dependency policy — permissive licenses only, never AGPL or a commercial license — is about code that ships inside the wheel, and DOTA is neither shipped nor vendored. What the terms do constrain is what may be derived from it: releases publish no trained weights (D14), datasets are never committed (AGENTS.md sec. 3), and the oriented tier's outputs are the reported metrics and the frozen goldens. Anyone reusing this repository for anything other than academic work should read the terms before provisioning DOTA at all.

COCO 2017 carries its own terms, published on `cocodataset.org` and not restated here — the annotations and the images they describe are covered separately, and the wording on that page is the authority. Neither dataset is redistributed by this repository.
