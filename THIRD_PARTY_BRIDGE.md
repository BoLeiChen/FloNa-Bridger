# Attribution

`model/bridge_diffusion.py` and the path trust / shallow residual design in
`model/pointgoal_bridge.py` adapt the VGCNav implementation, branch `feat/navibridger-astar-heading-realign`
(commit `b3b2ee5` plus the working-tree adaptations present during migration).
The local `main` branch does not contain this bridge implementation.
VGCNav's bridge in turn adapts NaviBridger's VE denoising diffusion bridge.

The original MIT notices are retained below.

## VGCNav

MIT License

Copyright (c) 2025 Shiyong Meng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## NaviBridger

MIT License

Copyright (c) 2025 Hao Ren, Yiming Zeng, Zetong Bi, Zhaoliang Wan,
Junlong Huang, Hui Cheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## RGB Planner v2 source provenance (2026-09-06)

`model/vgc_rgb_condition.py` copies the four named conditioning components
from the local VGCNav working tree verbatim. Its source snapshot SHA256 is:

21b6a0e7d304981fc9a43cbc297577a11987dea83e2ed9e5e5c5803e83704de1

`depth_anything_v2/dinov2.py` and `dinov2_layers/*.py` are copied from that
repository's Depth Anything V2 DINOv2 backbone. Original Meta/DINOv2 copyright
headers are preserved; the Apache-2.0 license is included at
`depth_anything_v2/LICENSE`. No DPT head is included or used.

Depth Anything V2 ViT-S pretrained file is copied unchanged from local
`checkpoints/depth_anything_v2_vits.pth`; SHA256:

715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378

## Method audit (2026-09-07)

The copied condition components retain their computations. Comments now clarify
FloVerse's RGB-only risk fallback and zero invalid confidence with retained padded XY.
