# Third-party notices

This repository contains or depends on the following third-party components. Each entry
states only what the repository itself records; where no licence file ships here, the
upstream project's own terms apply.

| Component | Where | Origin | Licence |
| --- | --- | --- | --- |
| G1 walking-controller kernel | `vendor/holosoma_inference/` | Derived from Amazon FAR Holosoma (revision `4ed2cebf9780f3efb59621657916006afadb87dc`) and the authors' AMO/Holosoma adapter; observation math from OpenTeleVision/AMO and Psi0 (module docstrings in `g1_control/config.py`, `g1_control/amo.py`, `g1_control/runtime.py`) | Apache-2.0, full text in `vendor/holosoma_inference/LICENSE`; Copyright Amazon.com, Inc. or its affiliates, all rights reserved (upstream NOTICE, carried in `NOTICE`) |
| AMO walking-policy weights | `assets/models/amo_jit.pt`, `adapter_jit.pt`, `adapter_norm_stats.pt` | Byte-identical to OpenTeleVision/AMO revision `34caaf943660e6f9420e35f64e86dd56fb51dd0e` (`g1_control/config.py`) | Apache-2.0, Copyright 2025 Jialong Li, Xuxin Cheng, Tianshu Huang and Xiaolong Wang (upstream LICENSE); upstream releases the weights for research use only |
| Unitree G1 robot description and meshes (29-DoF with Dex3 hands) | `assets/robot/g1_29dof_with_hand.urdf`, `assets/robot/g1_grasp/` | Unitree G1 model (`g1_29dof_with_hand_rev_1_0`); the URDF keeps joints only, the MuJoCo model and STL meshes are used for simulation | BSD-3-Clause, full text in `assets/robot/LICENSE` |
| MuJoCo | Python dependency (`pyproject.toml`), not vendored | MuJoCo physics engine | Apache-2.0 (MuJoCo) |
| Scanned kitchen | `assets/real2sim/src_kitchen/` | Captured by the authors; machine-local capture paths removed for release (`source.json`) | Not a third-party component |
| Project website assets (Three.js, HLS.js, DM Sans, provider icons) | `docs/` | As listed in `docs/LICENSES.txt` | MIT, Apache-2.0, SIL OFL 1.1 and MIT, full texts in `docs/LICENSES.txt`. Provider and institution logos are their owners' trademarks, shown for identification only. |

File hashes for every shipped asset are in `assets/manifest.json` and are checked by
`tools/verify_assets.py`. Integrity checks do not establish redistribution rights.
