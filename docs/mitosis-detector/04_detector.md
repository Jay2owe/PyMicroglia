# 04 detector

1. Baseline: the current rule scored against the gold set (sensitivity and precision).
2. A transparent scored rule from the stage 03 features, thresholds set from labelled positives.
3. A learned model only if stage 02 reached the positive count; otherwise skip.
4. Evaluate leave-one-recording-out so no recording scores itself. Report sensitivity, precision and how many wrong calls a reader would have to dismiss per true division.
5. Promote only if it beats the baseline on both numbers.

Done when: a results table and the chosen method are recorded. A negative result is a valid result.
