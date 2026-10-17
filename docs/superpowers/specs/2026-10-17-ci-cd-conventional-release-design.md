# CI/CD and Conventional Workflow Design

## Goal

Require Conventional Commit messages and predictable branch names, run those
checks locally and in pull requests, and automate semantic versioning and PyPI
publishing when qualifying changes reach `main`.

## Current state

- `ci.yml` runs pytest on Python 3.10, 3.11, and 3.12 for pushes to `main` and
  pull requests targeting `main`.
- `publish.yml` builds distributions and publishes to PyPI with Trusted
  Publishing when a release is published, or when manually dispatched.
- The canonical GitHub repository is `Diaszano/captix`; `captix` is a new PyPI
  distribution name. The existing `txtcaptcha` distribution and its Trusted
  Publisher belong to the upstream project and must not be reused.
- The repository has tag `v0.1.0`; `pyproject.toml` currently declares
  `0.1.0.9000`. The Python import package is renamed to `captix` with no `txtcaptcha` import compatibility.
- The repo has no pre-commit configuration or commit/branch validation.

## Contribution rules

### Commit messages

Use the Conventional Commits form:

```text
type(scope)!: description
```

`scope` and `!` are optional. A breaking change may also be indicated by a
`BREAKING CHANGE:` footer. Accepted types are `feat`, `fix`, `docs`,
`refactor`, `test`, `chore`, `ci`, `build`, and `perf`.

Every commit in a pull request and the pull request title must pass validation.
Squash merging is required so the validated pull request title becomes the
commit message evaluated by the release automation.

### Branch names

Topic branches use `<type>/<slug>` where `<type>` is one of the accepted commit
types and `<slug>` is lowercase kebab-case, for example `feat/novo-parser` or
`fix/erro-de-decode`. The `main` branch is exempt. Branch names are checked by
the local hook and in pull request CI.

### Local and CI checks

- Add a `.pre-commit-config.yaml` with lightweight file hygiene checks and the
  commit-message check.
- Provide a local branch-name check integrated with pre-commit.
- Run the same commit, PR-title, and branch checks in GitHub Actions so the
  result does not depend on contributors installing local hooks.
- Keep the existing Python test matrix.
- Document how to install pre-commit hooks, including the `commit-msg` hook.

## Automated release

Use `python-semantic-release`, configured in `pyproject.toml`, to inspect
Conventional Commits on `main` and update `project.version`.

Release rules:

| Commit | Version change |
| --- | --- |
| `feat` | Minor |
| `fix`, `perf` | Patch |
| `BREAKING CHANGE` footer or `!` | Major |
| `docs`, `refactor`, `test`, `chore`, `ci`, `build` | No release by itself |

The distribution name in project metadata is `captix`; Python imports continue
to use `captix`. Its version starts from the repository's existing `0.1.0`
tag baseline; this does not publish a release. Configure `allow_zero_version`
so ordinary feature and fix releases stay in `0.x`, while breaking changes
advance to `1.0.0`. Future
qualifying commits merged to `main` cause semantic-release to update the
version, create a `vX.Y.Z` tag, and create a GitHub Release.
Since the current version is `0.x`, a breaking change advances it to `1.0.0`.
The generated version commit uses `chore(release): {version}` and changelog
generation is disabled.

## Workflow and publishing

Keep the automated path in the existing `publish.yml` so the PyPI Trusted
Publisher continues to identify the same workflow file. Trigger it on pushes
to `main`, keep its existing `release: published` and `workflow_dispatch`
triggers, and distinguish the paths. The push path runs:

1. Semantic-release, with `contents: write`, to create a version commit, tag,
   and GitHub Release.
2. Only when semantic-release reports a release, build the source distribution
   and wheel and run `twine check`.
3. Only after a successful build, publish to PyPI using the existing `pypi`
   environment and OIDC (`id-token: write`).

The publish job runs directly after the release job in this workflow. It must
not rely on the `release` event to start another workflow: events caused by
`GITHUB_TOKEN` do not start a new workflow run. A human-created release keeps
using the existing `release: published` path; manual dispatch remains
available to retry or publish an explicitly selected ref. The generated
version commit does not contain a release-triggering Conventional Commit, so
its push does not produce a second release.

No personal access token or new repository secret is required. GitHub creates
`GITHUB_TOKEN` for the release job; PyPI authentication remains OIDC. Configure
a PyPI pending Trusted Publisher for `Diaszano/captix`, workflow `publish.yml`,
and environment `pypi`; the first successful upload creates the new `captix`
project.

## Repository settings

Protect `main` with pull-request-only merges and require the CI, Conventional
Commit/branch validation checks to pass. Require squash merging and code-owner
review for changes under `.github/workflows/`, with the repository owner listed
in `.github/CODEOWNERS` for that path. Allow the GitHub Actions app to bypass
the pull-request restriction so semantic-release can push its version commit
and tag. GitHub does not scope that app bypass to a specific commit; mitigate
it by granting `contents: write` only to the release job and requiring code
owner review for workflow changes. Do not grant a general user bypass for
ordinary changes.

## Changelog

Keep `CHANGELOG.md` manually maintained. Automatic changelog generation is out
of scope.

## Acceptance criteria

1. A malformed Conventional Commit fails the local `commit-msg` hook and the
   corresponding CI check.
2. A pull request with a malformed title or branch name fails CI.
3. A commit using a non-release type passes validation but does not create a
   release by itself.
4. A `feat` merge to `main` creates a minor version, tag, and GitHub Release;
   `fix`/`perf` create patch versions; breaking changes create major versions.
5. The package version in the built distributions matches the release tag.
6. PyPI publishing runs only after a successful release build and uses OIDC.
7. Existing tests continue to run on Python 3.10, 3.11, and 3.12.
8. A release commit does not start a second release run.
9. Local hook installation and recovery from a failed publish are documented.

## Out of scope

- Automatic changelog generation.
- Prereleases from topic branches.
- New repository secrets or personal access tokens.
- Changes to package APIs, test behavior, or documentation publishing.

## References

- [GitHub `GITHUB_TOKEN` workflow-trigger behavior](https://docs.github.com/en/actions/concepts/security/github_token)
- [Python Semantic Release GitHub Actions](https://python-semantic-release.readthedocs.io/en/latest/configuration/automatic-releases/github-actions.html)
