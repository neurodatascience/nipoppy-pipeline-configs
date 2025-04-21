"""Test that all pipelines and steps in the repo are tested."""

import json
from pathlib import Path

import pytest

from nipoppy.config.pipeline import BasePipelineConfig

from conftest import DPATH_PIPELINES, PIPELINE_INFO_BY_TYPE


@pytest.mark.parametrize("fpath_config", DPATH_PIPELINES.glob("*/*/config.json"))
def test_pipeline_in_conftest(fpath_config: Path):
    pipeline_config = BasePipelineConfig(**json.loads(fpath_config.read_text()))
    for step_config in pipeline_config.STEPS:
        pipeline_info = (
            pipeline_config.NAME,
            pipeline_config.VERSION,
            step_config.NAME,
        )
        assert (
            pipeline_info in PIPELINE_INFO_BY_TYPE[pipeline_config.PIPELINE_TYPE]
        ), f"Missing {pipeline_info} in conftest.py"
