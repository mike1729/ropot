"""Default artifact locations, independent of stage implementation imports."""

from pathlib import Path


DATA = Path(__file__).resolve().parents[2] / "data"
RAW_APPS = DATA / "raw" / "apps"
QUALIFIED_POOL = DATA / "01_qualified_pool"
POOL_ANALYSIS = DATA / "01_pool_analysis"
TEST_POOL = DATA / "01_test_pool"
COHORT = DATA / "02_cohort"
