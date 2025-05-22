import itertools
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import pytest
import pytest_mock

from nipoppy.env import PipelineTypeEnum
from nipoppy.config import Config
from nipoppy.layout import DatasetLayout
from nipoppy.tabular import Manifest, generate_curation_status_table
from nipoppy.tabular.dicom_dir_map import DicomDirMap
from nipoppy.workflows import InitWorkflow

DPATH_TESTS = Path(__file__).parent
DPATH_PIPELINES = DPATH_TESTS.parent / "pipelines"
FPATH_CONFIG = DPATH_TESTS / "data" / "global_config.json"

PIPELINE_INFO_BY_TYPE: [PipelineTypeEnum, Iterable[Tuple[str, str, str]]] = {
    PipelineTypeEnum.BIDSIFICATION: (
        ("heudiconv", "0.12.2", "prepare"),
        ("heudiconv", "0.12.2", "convert"),
        ("dcm2bids", "3.1.0", "prepare"),
        ("dcm2bids", "3.1.0", "convert"),
        ("dcm2bids", "3.2.0", "prepare"),
        ("dcm2bids", "3.2.0", "convert"),
        ("bidscoin", "4.3.2", "prepare"),
        ("bidscoin", "4.3.2", "edit"),
        ("bidscoin", "4.3.2", "convert"),
    ),
    PipelineTypeEnum.PROCESSING: (
        ("bids-validator", "2.0.3", "default"),  # no tracker
        ("freesurfer", "7.3.2", "default"),  # tracker only
        ("freesurfer", "6.0.1", "default"),  # tracker only
        ("fmriprep", "20.2.7", "default"),
        ("fmriprep", "23.1.3", "default"),
        ("fmriprep", "24.1.1", "default"),
        ("mriqc", "23.1.0", "default"),
        ("qsiprep", "0.23.0", "default"),
    ),
    PipelineTypeEnum.EXTRACTION: (
        ("fs_stats", "0.2.1", "default"),
        ("static_FC", "0.1.0", "default"),
        ("dmri_freewater", "2.0.0", "default")
    ),
}

PIPELINE_INFO_AND_TYPE: list[Tuple[Tuple[str, str, str], PipelineTypeEnum]] = (
    itertools.chain.from_iterable(
        [(pipeline_info, pipeline_type) for pipeline_info in pipeline_infos]
        for pipeline_type, pipeline_infos in PIPELINE_INFO_BY_TYPE.items()
    )
)


@pytest.fixture()
def single_subject_dataset(
    tmp_path: Path, mocker: pytest_mock.MockerFixture
) -> DatasetLayout:
    dataset_root = tmp_path / "my_dataset"
    participant_id = "01"
    session_id = "01"
    container_command = "apptainer"
    substitutions = {
        "[[HEUDICONV_HEURISTIC_FILE]]": str(tmp_path / "heuristic.py"),
        "[[DCM2BIDS_CONFIG_FILE]]": str(tmp_path / "dcm2bids_config.json"),
        "[[FREESURFER_LICENSE_FILE]]": str(tmp_path / "freesurfer_license.txt"),
        "[[TEMPLATEFLOW_HOME]]": str(tmp_path / "templateflow"),
    }

    participants_and_sessions = {participant_id: [session_id]}

    # create dataset structure
    workflow = InitWorkflow(dpath_root=dataset_root)
    workflow.run()
    layout = workflow.layout

    # copy pipelines
    shutil.copytree(DPATH_PIPELINES, layout.dpath_pipelines, dirs_exist_ok=True)

    # generate config file
    config = Config.load(FPATH_CONFIG, apply_substitutions=False)
    config.SUBSTITUTIONS = substitutions
    config.save(layout.fpath_config)
    for placeholder, fpath in substitutions.items():
        if "FILE" in placeholder:
            Path(fpath).touch()

    # create manifest and curation status files
    manifest = Manifest().add_or_update_records(
        {
            Manifest.col_participant_id: participant_id,
            Manifest.col_visit_id: session_id,
            Manifest.col_session_id: session_id,
            Manifest.col_datatype: None,
        }
    )
    manifest.save_with_backup(layout.fpath_manifest)
    curation_status_table = generate_curation_status_table(
        manifest,
        DicomDirMap.load_or_generate(
            manifest, fpath_dicom_dir_map=None, participant_first=True
        ),
    )
    curation_status_table.save_with_backup(layout.fpath_curation_status)

    # patch so that the test runs even if the command is not available
    mocker.patch(
        "nipoppy.config.container.check_container_command",
        return_value=container_command,
    )

    return layout, participant_id, session_id
