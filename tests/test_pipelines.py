"""Test pipeline configurations."""

import json
import warnings
from pathlib import Path

import pytest

from nipoppy.config.pipeline import BidsPipelineConfig
from nipoppy.config.container import ContainerInfo
from nipoppy.env import PipelineTypeEnum
from nipoppy.layout import DatasetLayout
from nipoppy.utils import TEMPLATE_REPLACE_PATTERN
from nipoppy.workflows import (
    PipelineValidateWorkflow,
    BidsConversionRunner,
    ExtractionRunner,
    PipelineRunner,
    PipelineTracker,
)

from conftest import DPATH_PIPELINES, PIPELINE_INFO_AND_TYPE, PIPELINE_INFO_BY_TYPE


@pytest.mark.parametrize("dpath_pipeline", DPATH_PIPELINES.glob("*/*-*"))
def test_nipoppy_pipeline_validate(dpath_pipeline: Path):
    """Test that pipeline bundles in the repo are valid."""
    PipelineValidateWorkflow(dpath_pipeline).run()


@pytest.mark.parametrize(
    "fpath_config", DPATH_PIPELINES.glob("bidsification/*-*/config.json")
)
def test_bids_pipeline_configs(fpath_config: Path):
    pipeline_config = BidsPipelineConfig(**json.loads(fpath_config.read_text()))
    if not any(
        [step.ANALYSIS_LEVEL == "participant_session" for step in pipeline_config.STEPS]
    ):
        pytest.xfail(
            (
                "UPDATE_STATUS cannot be enabled because no steps are at "
                f"participant-session level for pipeline {pipeline_config.NAME}"
                f" {pipeline_config.VERSION}"
            )
        )
    count = sum([step.UPDATE_STATUS for step in pipeline_config.STEPS])
    assert count == 1, (
        f"BIDS pipeline {pipeline_config.NAME} {pipeline_config.VERSION}"
        f" should have exactly one step with UPDATE_STATUS=true (got {count})"
    )


@pytest.mark.parametrize(
    "fpath_invocation", DPATH_PIPELINES.glob("extraction/*-*/invocation.json")
)
def test_extraction_invocation(fpath_invocation: Path):
    invocation = json.loads(fpath_invocation.read_text())
    fpath_script = invocation.get("script_path")
    if fpath_script is None:
        # check if pipeline has a non-empty container info
        fpath_config = fpath_invocation.parent / "config.json"
        config = ExtractionPipelineConfig(**json.loads(fpath_config.read_text()))
        if config.CONTAINER_INFO == ContainerInfo():
            raise RuntimeError(
                (
                    "Expected script_path in invocation since the pipeline "
                    f"doesn't use a container: {invocation}"
                )
            )
        else:
            pytest.xfail(
                "No extraction script expected since pipeline uses a container"
            )
    fpath_script = fpath_script.replace(
        "[[NIPOPPY_DPATH_PIPELINES]]", str(DPATH_PIPELINES)
    )
    assert Path(fpath_script).exists(), f"Extractor script not found: {fpath_script}"


@pytest.mark.parametrize(
    "pipeline_info,pipeline_type",
    PIPELINE_INFO_AND_TYPE,
)
def test_runner(
    pipeline_info: tuple[str, str, str],
    pipeline_type: PipelineTypeEnum,
    single_subject_dataset,
):
    """Test that pipelines run successfully in "simulate" mode."""
    pipeline_name, pipeline_version, pipeline_step = pipeline_info
    layout, participant_id, session_id = single_subject_dataset
    layout: DatasetLayout

    runner_class = {
        PipelineTypeEnum.BIDSIFICATION: BidsConversionRunner,
        PipelineTypeEnum.PROCESSING: PipelineRunner,
        PipelineTypeEnum.EXTRACTION: ExtractionRunner,
    }[pipeline_type]
    runner = runner_class(
        dpath_root=layout.dpath_root,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_step=pipeline_step,
        simulate=True,
    )
    runner.layout = layout

    # expect failure if descriptor/invocation files are defined
    if (
        runner.pipeline_step_config.DESCRIPTOR_FILE is None
        and runner.pipeline_step_config.INVOCATION_FILE is None
    ):
        pytest.xfail(f"Pipeline {pipeline_info} has no descriptor or invocation file")

    if (fpath_container := runner.pipeline_config.get_fpath_container()) is not None:
        fpath_container.touch()

    invocation_str, descriptor_str = runner.run_single(
        participant_id=participant_id, session_id=session_id
    )

    assert TEMPLATE_REPLACE_PATTERN.search(invocation_str) is None
    assert TEMPLATE_REPLACE_PATTERN.search(descriptor_str) is None


@pytest.mark.parametrize(
    "pipeline_info", PIPELINE_INFO_BY_TYPE[PipelineTypeEnum.PROCESSING]
)
def test_tracker(pipeline_info, single_subject_dataset):
    pipeline_name, pipeline_version, pipeline_step = pipeline_info
    layout, participant_id, session_id = single_subject_dataset
    layout: DatasetLayout
    tracker = PipelineTracker(
        dpath_root=layout.dpath_root,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_step=pipeline_step,
    )

    if tracker.pipeline_step_config.TRACKER_CONFIG_FILE is None:
        pytest.xfail(f"Pipeline {pipeline_info} has no tracker config file")

    # make sure all template strings are replaced
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracker.run_single(participant_id=participant_id, session_id=session_id)
