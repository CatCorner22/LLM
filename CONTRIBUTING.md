# Contributing to Pioneer ML

Thank you for contributing to Pioneer ML. This project is open source under the [MIT License](LICENSE).

## Getting started

1. Fork the repository and create a feature branch from `main`.
2. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

3. Run the quality gate before opening a pull request:

   ```bash
   make ci
   make license-check
   ```

## Pull request guidelines

- Keep changes focused and well-tested.
- Follow existing code style (ruff + mypy strict).
- Add or update tests for behavior changes.
- Update documentation when APIs or configuration change.
- Ensure all commits are your own work or properly attributed.

## License agreement

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

Do not submit code that is proprietary, GPL-licensed (without explicit approval), or copied from sources with incompatible licenses.

## Dependency policy

When adding dependencies:

1. Prefer permissive OSS licenses (MIT, BSD, Apache-2.0, ISC).
2. Run `make licenses` to regenerate third-party notices.
3. Run `make license-check` to verify license compatibility.
4. Document any dependency that requires separate legal review (e.g. NVIDIA CUDA packages in the `training` extra).

See [docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md) for the full open source policy.

## Code of conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report security issues privately. See [SECURITY.md](SECURITY.md).
