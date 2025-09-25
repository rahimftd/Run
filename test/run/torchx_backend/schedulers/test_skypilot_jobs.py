import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, Role

from nemo_run.core.execution.skypilot_jobs import SkypilotJobsExecutor
from nemo_run.run.torchx_backend.schedulers.skypilot_jobs import (
    SkypilotJobsScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def skypilot_jobs_executor():
    return SkypilotJobsExecutor(
        job_dir=tempfile.mkdtemp(),
        gpus="V100",
        gpus_per_node=1,
        cloud="aws",
    )


@pytest.fixture
def skypilot_jobs_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, SkypilotJobsScheduler)
    assert scheduler.session_name == "test_session"


def test_skypilot_jobs_scheduler_methods(skypilot_jobs_scheduler):
    assert hasattr(skypilot_jobs_scheduler, "_submit_dryrun")
    assert hasattr(skypilot_jobs_scheduler, "schedule")
    assert hasattr(skypilot_jobs_scheduler, "describe")
    assert hasattr(skypilot_jobs_scheduler, "_validate")


def test_submit_dryrun(skypilot_jobs_scheduler, mock_app_def, skypilot_jobs_executor):
    with mock.patch.object(SkypilotJobsExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = skypilot_jobs_scheduler._submit_dryrun(mock_app_def, skypilot_jobs_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_schedule(skypilot_jobs_scheduler, mock_app_def, skypilot_jobs_executor):
    class MockHandle:
        def get_cluster_name(self):
            return "test_cluster_name"

    with (
        mock.patch.object(SkypilotJobsExecutor, "package") as mock_package,
        mock.patch.object(SkypilotJobsExecutor, "launch") as mock_launch,
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
    ):
        mock_package.return_value = None
        mock_launch.return_value = (123, MockHandle())
        mock_status.return_value = None

        skypilot_jobs_executor.job_name = "test_job"
        skypilot_jobs_executor.experiment_id = "test_session"

        dryrun_info = skypilot_jobs_scheduler._submit_dryrun(mock_app_def, skypilot_jobs_executor)
        app_id = skypilot_jobs_scheduler.schedule(dryrun_info)

        # Note: SkypilotJobsExecutor uses 3-component app_id format (no experiment_id prefix)
        assert app_id == "test_cluster_name___test_role___123"
        mock_package.assert_called_once()
        mock_launch.assert_called_once()


def test_cancel_existing(skypilot_jobs_scheduler):
    with mock.patch.object(SkypilotJobsExecutor, "cancel") as mock_cancel:
        skypilot_jobs_scheduler._cancel_existing("test_cluster_name___test_role___123")
        mock_cancel.assert_called_once_with(app_id="test_cluster_name___test_role___123")


def test_describe_no_status(skypilot_jobs_scheduler):
    with (
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
        mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot_jobs._get_job_dirs") as mock_get_job_dirs,
    ):
        mock_status.return_value = None
        mock_get_job_dirs.return_value = {}

        result = skypilot_jobs_scheduler.describe("test_cluster___test_role___123")
        assert result is None


def test_describe_with_status(skypilot_jobs_scheduler):
    from sky.jobs.state import ManagedJobStatus

    task_details = {
        "status": ManagedJobStatus.RUNNING,
        "job_id": 123
    }

    with (
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
        mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot_jobs._save_job_dir") as mock_save,
    ):
        mock_status.return_value = task_details

        result = skypilot_jobs_scheduler.describe("test_cluster___test_role___123")

        assert result is not None
        assert result.app_id == "test_cluster___test_role___123"
        assert len(result.roles) == 1
        assert result.roles[0].name == "test_role"
        mock_save.assert_called_once()


def test_describe_with_past_jobs(skypilot_jobs_scheduler):
    past_apps = {
        "test_cluster___test_role___123": {
            "job_status": "SUCCEEDED"
        }
    }

    with (
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
        mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot_jobs._get_job_dirs") as mock_get_job_dirs,
    ):
        mock_status.return_value = None
        mock_get_job_dirs.return_value = past_apps

        result = skypilot_jobs_scheduler.describe("test_cluster___test_role___123")

        assert result is not None
        assert result.app_id == "test_cluster___test_role___123"
        # The state should be mapped from SUCCEEDED status
        from torchx.specs import AppState
        assert result.state == AppState.SUCCEEDED
