# 02 gold-set

1. Re-run the lifecycle module on the current accepted tracking for every recording (the 18 Sep tables may predate it). Record the input fingerprints.
2. Collect: every division candidate; a recall-oriented list from a relaxed rule (touching new name with no area limit, plus single tracks that split into two blobs); look-alikes (touching pairs that never separate, merges, fragments).
3. Render event clips with the existing `follow` action (`span=event`) into the run's `AI_Exports/event_videos`.
4. Jamie labels each clip in the Events page: division, not a division, or unsure. Save as one table: recording, identity, frame, label, labeller, date.
5. Count positives. If fewer than about 15 true divisions, stage 04 stays a transparent rule with reported sensitivity, not a trained model.

Done when: the label table is saved with fingerprints and its positive count is stated.
