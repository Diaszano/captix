# Contributing to Captix

The PyPI distribution is `captix`; Python imports use `captix`.

## Local setup

```bash
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

Before opening a pull request, run `uv run pre-commit run --all-files` and
`uv run pytest -q`.

## Branches and commits

Use a branch named `<type>/<short-kebab-case-description>`, where `<type>` is
`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `build`, or `perf`.
For example, `fix/empty-image`. Commits and pull request titles use Conventional
Commit syntax: `<type>(optional-scope): description`. The same types are
accepted; a breaking change uses `!` or a `BREAKING CHANGE:` footer.

Pull requests are squash merged. Give the squash commit a valid Conventional
Commit message, usually by setting the PR title: semantic-release parses that
final message. Preserve the intended release type there, and mark breaking
changes with `!` or a `BREAKING CHANGE:` footer. For example, renaming the
Python package needs `refactor!: rename Python package to captix`; without the
breaking marker, it will not trigger the major version bump. A `feat` release
increments the minor version; `fix` and `perf`
increment the patch version. A breaking change increments the major version.
Other accepted types do not trigger a release unless marked breaking.
While the version is `0.x`, ordinary features and fixes remain in `0.x`;
a breaking change advances to `1.0.0`.

## Release setup and recovery

On GitHub, protect `main` with pull requests, code owner review, and the
required CI checks: `validate` and each `test` Python version matrix job.
Allow squash merging and disable other merge methods. Give the GitHub Actions
app a bypass for the pull request restriction so semantic-release can push its
version commit and tag. GitHub cannot limit that app bypass to release commits;
review of `.github/workflows/` by `@Diaszano` and release-job-only
`contents: write` limit its scope.

Before the first release, configure a **pending Trusted Publisher** on PyPI for
owner/repository `Diaszano/captix`, workflow `publish.yml`, and environment
`pypi`. Publishing uses GitHub OIDC; no new repository secrets are needed.

The `publish.yml` workflow creates a release from qualifying commits on
`main`, then builds and publishes its tag. If the PyPI upload fails after the
tag exists, rerun the workflow manually against that **released tag**:

```bash
gh workflow run publish.yml --ref vX.Y.Z
```

Replace `vX.Y.Z` with the existing release tag. This dispatch rebuilds that
tag and retries the upload without creating another release.
