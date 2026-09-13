from ci_triage.miner.runs import RunInfo, find_transitions


def run(run_id, conclusion, minute, sha=None, event="pull_request", repo="fork/r", branch="b"):
    return RunInfo(
        run_id=run_id,
        workflow_id=1,
        workflow_name="CI",
        workflow_path=".github/workflows/ci.yml",
        event=event,
        head_repo=repo,
        head_branch=branch,
        head_sha=sha or f"{run_id:040x}",
        conclusion=conclusion,
        created_at=f"2026-09-01T10:{minute:02d}:00Z",
        run_attempt=1,
        html_url="",
    )


def test_consecutive_reds_collapse_into_one_streak():
    runs = [
        run(1, "success", 0),
        run(2, "failure", 1),
        run(3, "failure", 2),
        run(4, "success", 3),
    ]
    [t] = find_transitions(runs)
    assert [r.run_id for r in t.reds] == [2, 3]
    assert t.first_red.run_id == 2 and t.last_red.run_id == 3
    assert t.prev_green.run_id == 1
    assert t.next_green.run_id == 4


def test_input_order_does_not_matter():
    runs = [run(4, "success", 3), run(2, "failure", 1), run(1, "success", 0)]
    [t] = find_transitions(runs)
    assert t.prev_green.run_id == 1 and t.next_green.run_id == 4


def test_non_signal_conclusions_are_ignored():
    runs = [
        run(1, "failure", 0),
        run(2, "cancelled", 1),
        run(3, None, 2),  # still in progress
        run(4, "success", 3),
    ]
    [t] = find_transitions(runs)
    assert [r.run_id for r in t.reds] == [1]
    assert t.prev_green is None
    assert t.next_green.run_id == 4


def test_open_streak_has_no_next_green():
    [t] = find_transitions([run(1, "success", 0), run(2, "failure", 1)])
    assert t.next_green is None


def test_forks_and_events_are_separate_streaks():
    runs = [
        run(1, "failure", 0, repo="alice/r", branch="main"),
        run(2, "success", 1, repo="bob/r", branch="main"),  # different fork, same branch name
        run(3, "failure", 2, event="push", repo="alice/r", branch="main"),
    ]
    transitions = find_transitions(runs)
    assert len(transitions) == 2
    assert all(t.next_green is None for t in transitions)


def test_multiple_streaks_on_one_branch():
    runs = [
        run(1, "failure", 0),
        run(2, "success", 1),
        run(3, "failure", 2),
        run(4, "success", 3),
    ]
    first, second = find_transitions(runs)
    assert first.first_red.run_id == 1 and first.next_green.run_id == 2
    assert second.prev_green.run_id == 2 and second.next_green.run_id == 4


def test_from_api_handles_missing_head_repository():
    info = RunInfo.from_api(
        {
            "id": 9,
            "workflow_id": 3,
            "event": "push",
            "head_sha": "a" * 40,
            "created_at": "2026-09-01T00:00:00Z",
            "head_repository": None,
            "conclusion": "failure",
        }
    )
    assert info.head_repo == "" and info.run_attempt == 1
