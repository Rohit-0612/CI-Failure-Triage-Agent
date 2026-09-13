from ci_triage.git_local import CommitInfo
from ci_triage.miner.fix_matcher import FixEvidence, score_fix


def c(sha: str, message: str = "fix") -> CommitInfo:
    return CommitInfo(sha=sha.ljust(40, "0"), parents=(), author_date="", message=message)


def ev(**kwargs) -> FixEvidence:
    defaults = dict(
        has_green=True,
        relation="ahead",
        window_commits=[c("a1")],
        fix_files=["src/app.py"],
        same_job_passed_in_green=True,
    )
    return FixEvidence(**{**defaults, **kwargs})


def test_no_green_run():
    result = score_fix(FixEvidence(has_green=False))
    assert result.status == "no_green_found" and result.confidence is None


def test_rerun_on_same_commit_is_flaky_not_a_fix():
    result = score_fix(ev(relation="identical"))
    assert result.status == "flaky_rerun"
    assert result.candidate_fix_commit is None


def test_single_commit_fix_is_high_confidence():
    result = score_fix(ev())
    assert (result.status, result.confidence) == ("matched", "high")
    assert result.candidate_fix_commit == c("a1").sha


def test_amend_counts_as_single_commit_fix():
    result = score_fix(ev(relation="amend"))
    assert result.confidence == "high"
    assert "relation_amend" in result.reasons


def test_diverged_history_is_ambiguous():
    result = score_fix(ev(relation="diverged"))
    assert (result.status, result.confidence) == ("ambiguous", "low")
    assert result.candidate_fix_commit is None


def test_empty_code_change_is_treated_as_flaky():
    result = score_fix(ev(fix_files=[]))
    assert result.status == "flaky_rerun"
    assert "fix_window_changes_no_files" in result.reasons


def test_few_commits_is_medium_without_single_candidate():
    result = score_fix(ev(window_commits=[c("a1"), c("a2")]))
    assert (result.status, result.confidence) == ("matched", "medium")
    assert result.candidate_fix_commit is None


def test_large_window_is_ambiguous():
    commits = [c(f"a{i}") for i in range(6)]
    result = score_fix(ev(window_commits=commits))
    assert result.status == "ambiguous"
    assert "fix_window_too_large" in result.reasons


def test_missing_job_in_green_run_downgrades():
    result = score_fix(ev(same_job_passed_in_green=None))
    assert result.confidence == "medium"
    assert "failed_job_absent_in_green_run" in result.reasons


def test_docs_only_fix_is_low_confidence():
    result = score_fix(ev(fix_files=["docs/index.rst", "README.md"]))
    assert (result.status, result.confidence) == ("ambiguous", "low")


def test_requirements_txt_is_not_treated_as_docs():
    result = score_fix(ev(fix_files=["requirements.txt"]))
    assert result.confidence == "high"


def test_overlap_with_log_files_is_recorded():
    result = score_fix(ev(log_referenced_files=["src/app.py"]))
    assert "fix_overlaps_log_referenced_files" in result.reasons


def test_revert_of_breaking_commit_becomes_candidate():
    breaking = "b" * 40
    revert = c("r1", f'Revert "add thing"\n\nThis reverts commit {breaking[:12]}.')
    result = score_fix(ev(window_commits=[c("a1"), revert], breaking_shas=[breaking]))
    assert result.candidate_fix_commit == revert.sha
    assert result.confidence == "medium"
    assert "revert_of_breaking_commit" in result.reasons
