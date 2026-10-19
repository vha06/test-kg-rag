import os
import sys
import tempfile
import gc
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.seed_platform_fixtures import seed_fixtures
from evaluation.platform_verifier import PlatformVerifier


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def test_01_seed_platform_fixtures_execution(temp_db_path):
    summary = seed_fixtures(
        fixture_path="data/fixtures/synthetic_multi_domain_fixtures.json",
        profiles_yaml_path="config/platform_profiles.yaml",
        db_path=temp_db_path,
    )
    assert summary["status"] == "SUCCESS"
    assert summary["seeded_documents"] >= 4
    assert summary["seeded_entities"] >= 10
    assert summary["seeded_relationships"] >= 7
    assert summary["seeded_evidences"] >= 10
    assert summary["cross_domain_mappings"] >= 1
    assert "SYNTHETIC_TEST_ONLY" in summary["fixture_notice"]


def test_02_platform_verifier_all_11_checks(temp_db_path):
    # Seed first
    seed_fixtures(
        fixture_path="data/fixtures/synthetic_multi_domain_fixtures.json",
        profiles_yaml_path="config/platform_profiles.yaml",
        db_path=temp_db_path,
    )

    with tempfile.TemporaryDirectory() as temp_reports:
        verifier = PlatformVerifier(
            fixture_path="data/fixtures/synthetic_multi_domain_fixtures.json",
            profiles_yaml_path="config/platform_profiles.yaml",
            db_path=temp_db_path,
            reports_dir=temp_reports,
        )
        report = verifier.run_all_checks()

        assert report["overall_status"] == "PASSED"
        assert report["total_checks"] == 11
        assert report["passed_checks"] == 11
        assert report["pass_rate"] == "100.0%"
        assert "SYNTHETIC TEST ENVIRONMENT NOTICE" in report["disclaimer"]

        # Check all 11 individual check statuses
        for check in report["checks"]:
            assert check["status"] == "PASSED", f"Check {check['check_id']} failed: {check.get('details')}"
