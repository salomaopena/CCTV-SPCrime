# CCTV-SPCrime

**Dataset of surveillance images annotated according to an event-driven policy, for detecting public safety incidents.**

[![DOI](https://img.shields.io/badge/10.5281/zenodo.20801289)](https://doi.org/10.5281/zenodo.20801289) [![License](https://img.shields.io/badge/license-CC%20BY%204.0-orange)](./LICENSE) [![Version](https://img.shields.io/badge/version-1.0.0-informational)](./CHANGELOG.md)

> **Documentation:** this repository follows international data documentation standards — see the [DATASHEET.md](DATASHEET.md) (Datasheets for Datasets, Gebru et al.) and the [CITATION.cff](CITATION.cff). The dataset adheres to the **FAIR** principles (Findable, Accessible, Interoperable, Reusable).

---

## Overview

**CCTV-SPCrime** is a dataset of still images taken from surveillance scenarios, labeled for **eight types of public safety incidents** according to an **event-oriented annotation policy** — where the incident, not the individual object, is the unit of annotation. It's meant for **reproducible evaluation** of incident detection models.

- **Mode:** images (frames) 640×640 px, labels in YOLO format + event attributes.
- **Classes (8):** accident, suspicious behavior, crime, fire, intrusion, suspicious object, fall, vandalism.
- **Split:** 80% training / 10% validation / 10% testing.
- **Total:** : 4.042 images.
- **License:** CC BY 4.0.
- **Associated works:** data paper (Pena et al., 2026) and AIVIS.GCUB architecture (Pena et al., 2026) — see [Citation](#citation).

## Classes and distribution

| Class | Short description | Train | Validation | Test | Total |
|---|---|---|---|---|---|
| accident | collisions, mass falls, visible medical emergencies | `400` | `50` | `50` | `500` |
| suspicious_behavior | atypical movements, escape, pursuit | `404` | `51` | `51` | `506` |
| crime | physical violence or taking of goods (with attributes) | `455` | `57` | `57` | `569` |
| fire | fire, smoke, flames | `420` | `53` | `53` | `524` |
| intrusion | crossing perimeter/restricted areas | `426` | `50` | `50` | `532` |
| suspicious_object | abandoned objects, weapons | `400` | `50` | `50` | `500` |
| fall | person lying in an unusual position | `326` | `41` | `41` | `408` |
| vandalism | property damage | `403` | `50` | `50` | `503` |

> The numbers should be filled in after the recount (see [Provenance and licenses](#provenance-and-licenses)).

## Repository structure

``` 
CCTV-SPCrime/
├── README.md                 # this file
├── DATASHEET.md              # datasheet (Gebru et al.)
├── CITATION.cff              # citation metadata
├── LICENSE                   # set license CC-BY-4
├── data.yaml                 # YOLO setup (classes + paths)
├── images/
│   ├── train/  val/  test/   # imagens .jpg
├── labels/
│   ├── train/  val/  test/   # YOLO labels .txt (one line per object)
├── attributes/               # event attributes by image (.json)
├── annotation_guidelines/    # annotation guidelines and examples
└── provenance/               # origin and license by sample (.csv)
```

## Annotation format

Each image `images/<split>/<id>.jpg` has:

1. **Spatial label (YOLO):** `labels/<split>/<id>.txt`, one line per object: `<class_id> <x_center> <y_center> <width> <height>` (coordinates normalized [0,1]).
2. **Event attributes:** `attributes/<id>.json`, according to the four pillars of the policy:

   ```json
   {
     "image_id": "<id>",
     "incident_class": "crime",
     "pillars": {
       "object": ["person"],
       "action": ["assault"],
       "environment": ["restricted_area"],
       "normal_abnormal": "abnormal"
     },
     "attributes": { "weapon_present": false, "theft": false },
     "source": "<source>", "license": "<origin_license>"
   }
   ```

The mapping `class_id → name` is in `data.yaml`. The full guidelines (definitions, boundaries, and positive/negative examples per class) are in `annotation_guidelines/`.

## annotation_guidelines

| File | Content |
|---|---|
| [`class_guidelines.md`](annotation_guidelines/class_guidelines.md) | **Class guidelines** (image): definitions, boundaries, positive/negative examples, the 4 pillars, confusion table, and quality control (kappa + IoU). |
| [`video_annotation_policy.md`](annotation_guidelines/video_annotation_policy.md) | Annotation policy for the **video extension** (semantic-temporal analysis; three layers). |
| [`cvat_video_guide.md`](annotation_guidelines/cvat_video_guide.md) | Operational guide for **CVAT** for the video extension. |

For the main set (image, v1.0), the reference is `class_guidelines.md`. The last two documents cover the video roadmap (v2.0).

## Quick use

```python
# Training with Ultralytics YOLO (example)
from ultralytics import YOLO
model = YOLO("yolo26n.pt")           # or yolov8n.pt / yolo11n.pt
model.train(data="data.yaml", imgsz=640, epochs=100, batch=32, seed=0)
metrics = model.val(split="test")    # evaluation on the test set
```

## Provenance and licenses

The set combines **original source frames** with samples from public repositories whose licenses **allow redistribution/derivation**. The provenance is recorded per sample in `provenance/` (source, original license, date.

The license of the resulting set is CC BY 4.0, respecting the licenses of the retained sources. See [LICENSE](LICENSE) and [DATASHEET.md](DATASHEET.md).

## Ethics and privacy (GDPR)

The images depict people in public spaces. Before publication, **anonymization** (blurring faces and license plates) is applied, and no personal identifiers are retained. The dataset is intended for **research**; it should not be used to identify individuals or for high-risk automated decisions without human oversight and ethical-legal review. See the *Uses* and *Collection Process* sections of the [DATASHEET.md](DATASHEET.md).

## Versioning and roadmap

- **v1.0.0** — image version (image-level detection).
- **Roadmap:** extension for **video** with time-stamped annotations in three layers (spatial/tracking/event) and evaluation by tIoU; full replacement of restrictive license sources.

Changes are recorded in [CHANGELOG.md](./CHANGELOG.md) and follow semantic versioning.

## Citation

If you use this dataset, cite the data paper (see [CITATION.cff](CITATION.cff)):

> Pena, S. B. N.; Souza, J. R.; Nomura, S. (2026). *CCTV-SPCrime: An Event-Oriented Dataset for Public Safety Incident Detection.* `[TO FILL IN: vehicle and DOI]`

## How to contribute

Annotation corrections and bug reports are welcome via *issues* and *pull requests*. See the guidelines at `annotation_guidelines/`.

## Contact

- Salomão Pena | Universidade Federal de Uberlândia — FACOM/UFU (`salomao.pena@ufu.br`).
- Salomão Pena | Instituto Superior de Ciências da Educação da Huíla (`salomao.pena@isced-huila.edu.ao`)

---
