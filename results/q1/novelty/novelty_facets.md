# Architecture novelty: facet matrix

| Facet | MHF-Net (TPAMI'22) | MoG-DCN (TIP'21) | DHIF-Net (TCI'22) | SMGU-Net (PR'25) | PSTUN (InfFus'25) | CaFormer (PR'26) | OTPNet (AAAI'25) | IR&ArF (TIP'25) | PGU-Net (arXiv'26) | HyCoNet (TGRS'20) | SSA (ICML'26) | MIMFormer (JSTARS'24) | FUSE (TIP'15) | HySure (TGRS'15) | Robust fusion CD (TGRS'18) | USRNet (CVPR'20, SR) | SC-Net (arXiv'26) | DDNM (ICLR'23) | Deep decomposition (ECCV'20) | Ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unfolded data step + learned prior | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  | ✓ |  | ~ |  | ✓ |
| exact (non-learned) observation operators |  |  |  |  |  | ~ |  |  |  | ~ |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| border rule and adjoint of the simulation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |
| closed-form exact data step |  |  |  |  |  |  |  |  |  |  |  |  | ~ |  |  | ~ |  |  |  | ✓ |
| learned filters in the operator's eigen domain |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |  |  | ✓ |
| operators estimated from the observation pair |  |  |  |  |  |  |  |  | ✓ | ~ |  |  |  | ✓ |  |  |  |  |  | ✓ |
| trained over a family of observation models |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |  |  |  | ✓ |
| operator / noise conditioning |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |  |  |  | ✓ |
| one network for any MSI response (no retraining) |  |  |  |  |  |  |  |  | ~ |  | ~ |  |  |  |  |  |  |  |  | ✓ |
| one network for several scale factors |  |  |  |  | ✓ |  |  | ✓ |  |  | ✓ |  |  |  |  | ✓ |  |  |  | ✓ |
| misregistration in the operator |  |  |  |  |  |  |  | ✓ |  |  |  |  |  |  |  |  |  |  |  | ✓ |
| prior confined to the null space (range-null) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ |
| uncertainty output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |
| down-weights corrupted / changed inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |  |  |  |  | ✓ |
| global mixing in the operator's eigenbasis |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |
| fixed-point / non-expansiveness guarantees |  |  |  |  |  |  | ~ |  |  |  |  |  |  |  |  |  |  |  |  | ✓ |

## Compared with all prior work (19 methods)

- facets no method has: 3 of 16 (19%): border rule and adjoint of the simulation; uncertainty output; global mixing in the operator's eigenbasis
- facets only partly present elsewhere: 3: closed-form exact data step; one network for any MSI response (no retraining); fixed-point / non-expansiveness guarantees
- closest methods by facet overlap (weighted Jaccard): USRNet (CVPR'20, SR) 34%; IR&ArF (TIP'25) 19%; PGU-Net (arXiv'26) 16%
- distance to the nearest method: 66%

## Compared with fusion methods only (15 methods)

- facets no method has: 7 of 16 (44%): border rule and adjoint of the simulation; learned filters in the operator's eigen domain; trained over a family of observation models; operator / noise conditioning; prior confined to the null space (range-null); uncertainty output; global mixing in the operator's eigenbasis
- facets only partly present elsewhere: 3: closed-form exact data step; one network for any MSI response (no retraining); fixed-point / non-expansiveness guarantees
- closest methods by facet overlap (weighted Jaccard): IR&ArF (TIP'25) 19%; PGU-Net (arXiv'26) 16%; Robust fusion CD (TGRS'18) 12%
- distance to the nearest method: 81%
