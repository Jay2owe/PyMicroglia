"""Unknown or differently timed rhythms never become daily/comparable by plotting."""
from types import SimpleNamespace
import pandas as pd
import pytest
from pymicroglia.figure_tables.rhythm_eligibility import common_period,daily_profiles
from pymicroglia.figure_tables.saved import Tables
from tests.test_motion_parity import FROZEN_RUN


def supported(periods):
    return pd.DataFrame({'best_period_hours':periods,'rhythmic':[True]*len(periods),'period_underdetermined':[False]*len(periods)})


def test_different_supported_periods_are_not_made_comparable_by_cycle_fractions():
    with pytest.raises(ValueError,match='periods differ'):common_period(supported([8.,24.]))
    assert common_period(supported([8.,8.]))==8.


@pytest.mark.parametrize('column,value',[('rhythmic',False),('period_underdetermined',True),('best_period_hours',float('nan'))])
def test_missing_support_refuses_comparable_timing(column,value):
    rows=supported([24.,24.]);rows.loc[0,column]=value
    with pytest.raises(ValueError,match='supported significant'):common_period(rows)


def test_default_measurements_do_not_authorize_daily_summary():
    source=Tables(FROZEN_RUN,'parity_A1');rows=source.table('rhythms')
    with pytest.raises(ValueError,match='explicitly enabled'):daily_profiles(source,rows)


def test_explicit_recorded_daily_summary_is_eligible():
    source=SimpleNamespace(module_params=lambda _: {'daily_profile_measures':True})
    daily_profiles(source,pd.DataFrame({'metric':['area']}))
    with pytest.raises(ValueError,match='explicitly enabled'):
        daily_profiles(source,pd.DataFrame({'daily_profile_measures_enabled':[False]}))
