# Mitosis detector for PyMicroglia lifecycle events (plan, draft)

Drafted 12 November 2026 for Jamie's Late Stage Review. Not started; nothing here changes existing measurements.

## Goal

Turn the current rule-based division candidate (`pymicroglia.measure.modules.lifecycle`, rule: a new name touching an existing one whose two areas together are within 40 % of the parent's previous area) into a scored, validated mitosis call. It runs on data we already have: the saved label stacks, unclaimed-foreground stacks and total-light traces of every tracked recording.

## Where it lives

As a post-tracking **output** of the lifecycle module, not a change to the tracker. The guide already says events are classified after tracking from the original accepted Motion masks, and films never re-infer events. The detector adds scored columns and a status, and the Events page, event clips and track timeline pick it up unchanged. A later optional step can feed confirmed divisions back to the tracker to repair identities.

## What exists today (checked 12 Nov 2026)

Pooled lifecycle table for the seven RBM3 recordings (`development/mask_to_motion_handoff/analysis_outputs/rbm3_7/pooled/lifecycle_events.csv`, dated 18 Sep): 1 division candidate (RBM3_MCG_09, cell 19 into 19 + 29 at 22.1 h; areas 244 into 183 + 57), 113 death candidates, 7 absorptions, 102 unexplained appearances. No candidate has been confirmed by eye. Cry1 and CD68 recordings have not been run through lifecycle in that table.

## First look at the one candidate (12 Nov 2026)

RBM3_MCG_09 cells 19 and 29 (frame 12, 22.1 h): looked at raw frames 8 to 19 with both outlines. Cell 29 is present at frames 12, 14 and 17 only (57, 84 and 38 px) and absent between, sitting as a thin sliver on the edge of cell 19's outline, while cell 19's own area wanders 141 to 282 px. The raw frames are photon-counting noise at 1.58 h per frame. This reads as a segmentation flicker, not a division, so it is not a usable example. No other existing lifecycle table (20260721_1417, 20260810_1432, MCG_04, the other six RBM3 recordings) contains a division. The newer full-length CD68 and Cry1 recordings have no lifecycle table yet.

Consequence: stage 02 must start by running lifecycle on those recordings, which have clearer cell morphology, and the `min_frames_after_split` rule needs checking, because this candidate passed it while cell 29 vanished in between.

## Scan of every accepted tracking stack (12 Nov 2026)

Ran the lifecycle rules unchanged on the latest accepted label stack of 21 recordings (11 CD68 `95_*`, 7 RBM3, MCG_04, two LV200); table in `Admin/LSR/Supplementary videos/events/scan_events_all_recordings.csv`, script `scan_events.py` beside it. 96 division candidates in total, 67 of them in CD68. Most are fragments: of the 67, only 18 have both daughters at 40 px or more and stay together 30 or more frames. Looking at those by eye, about two look like real two-body splits (95_A2 cell 14 into 14 + 63 at 17 h is the clearest; 95_B3 cell 66 + 24 is a maybe). The rest are outlines over noise or one cell re-labelled. Absorptions mostly look like identity merges, not biology.

Code finding: `_division` computes `together = min(last[identity], last[partner]) - frame_index + 1`, which counts to each name's last appearance, not consecutive frames, so a flickering fragment (RBM3_MCG_09 cell 29) passes the 3-frame rule. Stage 03 must use consecutive presence.

These stacks are the older roughly 49 h CD68 tracking (99 frames), not the newer 476-frame recordings, so counts are a first look, not a rate.

Candidate clips and review sheets: `Admin/LSR/Supplementary videos/events/` (`clips/`, `review_*.png`; renderer `make_event_clips.py`).

## Risk to state up front

Microglia in slice may divide very rarely. A validated answer of "almost none" is a legitimate result, but it means a trained classifier may have too few true positives. Stage 02 sets the size of the gold set before anything is trained.

## Stages (each one is a /do-step file)

| NN | name | goal | needs Jamie |
|----|------|------|-------------|
| 01 | scout-existing | Compare existing trackers and division detectors before writing any | no |
| 02 | gold-set | Refresh events on current tracking, collect candidates and look-alikes, label by eye | yes, about 1 to 2 h of labelling |
| 03 | features | Per candidate: rounding, area symmetry, daughters separating, total-light conservation, persistence | no |
| 04 | detector | Transparent scored rule first, learned model only if the gold set allows; leave-one-recording-out check | no |
| 05 | integrate | New columns and status in lifecycle, Events page filter, clips, tests, METHOD_VERSION bump | no |
| 06 | report | Division counts and rate per recording and per 24 h, before vs after amyloid, only if validated | review |

Order is strict: 01 can change 03 to 05; 02 sets what 04 may claim.
