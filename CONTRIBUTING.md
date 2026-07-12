# Contributing

Thanks for your interest in PTAIE. The project is young and governance is deliberately simple for now.

## Workflow

1. Fork the repository (or create a branch, if you have access) and make your changes.
2. Open a pull request against `main`.
3. The maintainer (@danielxmed) reviews every PR, may request changes, and is the only person who merges.

`main` is protected by a repository ruleset: direct pushes, force pushes, and branch deletion are blocked for everyone, including the maintainer. All changes land through pull requests.

## Ground rules for PRs

- Read `docs/NORTH_STAR.md` and `CLAUDE.md` / `AGENTS.md` first. PRs that violate the environment/trainer boundary or the listed design invariants will be declined regardless of code quality.
- `CLAUDE.md` and `AGENTS.md` mirror each other. If your PR changes one, apply the identical change to the other.
- Keep PRs focused; unrelated changes belong in separate PRs.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0, the same license that covers the project.

As the project grows, this process will be revisited (official reviewers, CODEOWNERS, CI gates).
