import pytest

from scripts.check_branch_name import is_valid_branch_name, main


@pytest.mark.parametrize(
    "branch",
    ["main", "feat/novo-parser", "fix/erro-de-decode", "docs/2-exemplos"],
)
def test_accepts_main_and_valid_topic_branches(branch):
    assert is_valid_branch_name(branch)


@pytest.mark.parametrize(
    "branch",
    ["feature/x", "feat/", "feat/Novo", "fix/erro_bug", "feat/a--b", "feat/-a"],
)
def test_rejects_unknown_type_or_invalid_slug(branch):
    assert not is_valid_branch_name(branch)


def test_cli_accepts_explicit_branch():
    assert main(["--branch", "ci/validar-pr"]) == 0


def test_cli_rejects_invalid_branch(capsys):
    assert main(["--branch", "feature/invalid"]) != 0
    assert "Invalid branch name" in capsys.readouterr().err
