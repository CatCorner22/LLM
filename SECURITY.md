# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please **do not** open public GitHub issues for security vulnerabilities.

Instead, report security issues privately to the repository maintainers through
GitHub Security Advisories or by contacting the project owner directly.

Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

We aim to acknowledge reports within 3 business days and provide a remediation
timeline based on severity.

## Secure development practices

This project uses:

- `pip-audit` in CI for dependency vulnerability scanning
- Pre-commit hooks for code quality
- Minimal required permissions in CI workflows

When deploying Pioneer ML:

- Store API keys in environment variables, never in source code
- Use `.env` locally and a secrets manager in production
- Keep dependencies updated and run `pip-audit` regularly
