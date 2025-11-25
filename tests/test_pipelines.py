"""Test pipeline configurations."""

import io
import json
import re
import warnings
from pathlib import Path

import pytest
from conftest import DPATH_PIPELINES, PIPELINE_INFO_AND_TYPE, PIPELINE_INFO_BY_TYPE
from nipoppy.config.container import ContainerInfo
from nipoppy.config.main import Config
from nipoppy.config.pipeline import (
    BIDSificationPipelineConfig,
    ExtractionPipelineConfig,
)
from nipoppy.env import PipelineTypeEnum
from nipoppy.layout import DatasetLayout
from nipoppy.utils.utils import TEMPLATE_REPLACE_PATTERN
from nipoppy.workflows.bids_conversion import BIDSificationRunner
from nipoppy.workflows.extractor import ExtractionRunner
from nipoppy.workflows.pipeline_store.install import PipelineInstallWorkflow
from nipoppy.workflows.pipeline_store.validate import PipelineValidateWorkflow
from nipoppy.workflows.processing_runner import ProcessingRunner
from nipoppy.workflows.runner import Runner
from nipoppy.workflows.tracker import PipelineTracker

VARIABLE_REPLACE_PATTERN = re.compile(r"\[\[(.*?)\]\]")


@pytest.fixture
def pipeline_variables(tmp_path: Path) -> dict[str, str]:
    return {
        "HEUDICONV_HEURISTIC_FILE": str(tmp_path / "heuristic.py"),
        "DCM2BIDS_CONFIG_FILE": str(tmp_path / "dcm2bids_config.json"),
        "FREESURFER_LICENSE_FILE": str(tmp_path / "freesurfer_license.txt"),
        "TEMPLATEFLOW_HOME": str(tmp_path / "templateflow"),
    }


def _install_pipeline(
    layout: DatasetLayout,
    pipeline_name: str,
    pipeline_version: str,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_variables: dict[str, str] = None,
    pipeline_type: PipelineTypeEnum = PipelineTypeEnum.PROCESSING,
):
    if pipeline_variables is None:
        pipeline_variables = {}

    fpath_pipeline = (
        DPATH_PIPELINES / pipeline_type.value / f"{pipeline_name}-{pipeline_version}"
    )
    paths_to_install = [fpath_pipeline]
    if pipeline_type == PipelineTypeEnum.EXTRACTION:
        pipeline_config = ExtractionPipelineConfig(
            **json.loads((fpath_pipeline / "config.json").read_text())
        )
        for info in pipeline_config.PROC_DEPENDENCIES:
            paths_to_install.append(
                DPATH_PIPELINES
                / PipelineTypeEnum.PROCESSING.value
                / f"{info.NAME}-{info.VERSION}"
            )
    for path in paths_to_install:
        monkeypatch.setattr("sys.stdin", io.StringIO("n"))  # do not install container
        installer = PipelineInstallWorkflow(dpath_root=layout.dpath_root, source=path)
        installer.run()

    # set pipeline variables
    config = Config.load(layout.fpath_config)
    variables_to_set = {
        variable: value
        for variable, value in pipeline_variables.items()
        if (
            variable
            in config.PIPELINE_VARIABLES.get_variables(
                pipeline_type, pipeline_name, pipeline_version
            )
        )
    }
    config.PIPELINE_VARIABLES.set_variables(
        pipeline_type, pipeline_name, pipeline_version, variables_to_set
    )
    config.save(layout.fpath_config)


@pytest.mark.parametrize("dpath_pipeline", DPATH_PIPELINES.glob("*/*-*"))
def test_nipoppy_pipeline_validate(dpath_pipeline: Path):
    """Test that pipeline bundles in the repo are valid."""
    PipelineValidateWorkflow(dpath_pipeline).run()


@pytest.mark.parametrize(
    "fpath_descriptor", DPATH_PIPELINES.glob("*/*-*/descriptor*.json")
)
def test_descriptors(fpath_descriptor: Path):
    descriptor = json.loads(fpath_descriptor.read_text())
    command_line = descriptor["command-line"]

    # old Nipoppy template strings not allowed
    assert "[[NIPOPPY_CONTAINER_COMMAND]]" not in command_line, str(fpath_descriptor)
    assert "[[NIPOPPY_FPATH_CONTAINER]]" not in command_line, str(fpath_descriptor)

    if descriptor["name"] not in ("static_FC", "fs_stats"):
        assert command_line.lstrip()[0] != "[", str(fpath_descriptor)
    else:
        assert command_line.startswith("python [SCRIPT_PATH]"), str(fpath_descriptor)

    with pytest.raises(KeyError):
        descriptor["custom"]["nipoppy"], str(fpath_descriptor)


@pytest.mark.parametrize(
    "fpath_config", DPATH_PIPELINES.glob("bidsification/*-*/config.json")
)
def test_bids_pipeline_configs(fpath_config: Path):
    pipeline_config = BIDSificationPipelineConfig(
        **json.loads(fpath_config.read_text())
    )
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
    pipeline_variables: dict[str, str],
    single_subject_dataset,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that pipelines run successfully in "simulate" mode."""
    pipeline_name, pipeline_version, pipeline_step = pipeline_info
    layout, participant_id, session_id = single_subject_dataset
    layout: DatasetLayout

    # install the pipeline + any proc dependencies
    _install_pipeline(
        layout=layout,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        monkeypatch=monkeypatch,
        pipeline_variables=pipeline_variables,
        pipeline_type=pipeline_type,
    )

    runner_class = {
        PipelineTypeEnum.BIDSIFICATION: BIDSificationRunner,
        PipelineTypeEnum.PROCESSING: ProcessingRunner,
        PipelineTypeEnum.EXTRACTION: ExtractionRunner,
    }[pipeline_type]
    runner: Runner = runner_class(
        dpath_root=layout.dpath_root,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_step=pipeline_step,
        simulate=True,
    )
    runner.study.layout = layout

    # expect failure if descriptor/invocation files are defined
    if (
        runner.pipeline_step_config.DESCRIPTOR_FILE is None
        and runner.pipeline_step_config.INVOCATION_FILE is None
    ):
        pytest.xfail(f"Pipeline {pipeline_info} has no descriptor or invocation file")

    if (fpath_container := runner.pipeline_config.CONTAINER_INFO.FILE) is not None:
        fpath_container.touch()

    descriptor_str, invocation_str = runner.run_single(
        participant_id=participant_id, session_id=session_id
    )

    assert TEMPLATE_REPLACE_PATTERN.search(invocation_str) is None
    assert TEMPLATE_REPLACE_PATTERN.search(descriptor_str) is None
    assert VARIABLE_REPLACE_PATTERN.search(invocation_str) is None


@pytest.mark.parametrize(
    "pipeline_info", PIPELINE_INFO_BY_TYPE[PipelineTypeEnum.PROCESSING]
)
def test_tracker(
    pipeline_info,
    single_subject_dataset,
    pipeline_variables: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline_name, pipeline_version, pipeline_step = pipeline_info
    layout, participant_id, session_id = single_subject_dataset
    layout: DatasetLayout

    # install the pipeline
    _install_pipeline(
        layout=layout,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        monkeypatch=monkeypatch,
        pipeline_variables=pipeline_variables,
        pipeline_type=PipelineTypeEnum.PROCESSING,
    )

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
