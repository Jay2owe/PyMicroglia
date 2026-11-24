# 03 features

For every gold-set event compute from saved data only, with the window length as a setting:

- parent rounding before the split (solidity, eccentricity, area change)
- area ratio of the daughters and conservation of area
- total-light conservation: daughters' summed light over the parent's light before
- distance between daughters over the frames after, and whether they separate
- both parts persisting for the minimum frames, and no re-merge
- border contact and gap frames

One table, one row per event, same keys as the label table. No image models.

Done when: the feature table is saved with the label table and a short check shows each feature is defined for every row.
