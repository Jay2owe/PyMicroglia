"""Distinct measured peaks with gap eligibility based on observed continuations."""
import pandas as pd
import continuation_only_soma_peaks as base
import distinct_core_continuation_peaks as distinct
import latent_body_tracks
import observed_core_continuation_linker

frame_hypotheses = distinct.frame_hypotheses


def link_hypotheses(hypotheses, raw, motion, params):
    original = latent_body_tracks.link_observations
    eligibility_audit = []

    def link(table, images, evidence, settings):
        return observed_core_continuation_linker.link_observations(
            table, images, evidence,
            dict(settings, preserve_measured_core_continuations=True),
            audit=eligibility_audit)

    latent_body_tracks.link_observations = link
    try:
        tracks, continuation_audit = base.link_hypotheses(hypotheses, raw, motion, params)
    finally:
        latent_body_tracks.link_observations = original
    if eligibility_audit:
        continuation_audit = pd.concat(
            [continuation_audit, pd.DataFrame(eligibility_audit)], ignore_index=True)
    return tracks, continuation_audit
