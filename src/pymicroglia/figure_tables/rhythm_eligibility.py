"""Require recorded scientific support before displaying comparable timing."""
import numpy as np
import pandas as pd


def daily_profiles(source,rows):
    if 'daily_profile_measures_enabled' in rows:
        enabled=rows.daily_profile_measures_enabled.fillna(False).astype(str).str.lower().isin(['true','1','1.0'])
        if enabled.all() and len(enabled):return
    elif source.module_params('rhythms').get('daily_profile_measures') is True:
        return
    raise ValueError('This figure requires explicitly enabled daily_profile_measures in the saved rhythm analysis; unknown rhythms are not assumed to be daily')


def common_period(rows):
    required={'best_period_hours','rhythmic','period_underdetermined'}
    if not required<=set(rows):
        raise ValueError('Comparable timing requires saved period estimates, significance and observation-sufficiency evidence')
    periods=pd.to_numeric(rows.best_period_hours,errors='coerce').to_numpy(float)
    passed=rows.rhythmic.fillna(False).astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
    sufficient=rows.period_underdetermined.astype(str).str.lower().isin(['false','0','0.0']).to_numpy()
    if not len(periods) or not (np.isfinite(periods)&(periods>0)&passed&sufficient).all():
        raise ValueError('Comparable timing requires supported significant periods for every contributing cell and signal')
    if not np.allclose(periods,periods[0],rtol=0,atol=1e-9):
        raise ValueError('These saved periods differ; cycle fractions do not establish comparable timing')
    return float(periods[0])
