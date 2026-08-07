# Open Source Policy

Pioneer ML is released under the [MIT License](../LICENSE), a permissive open
source license widely used in commercial and research software.

## Project license

| Item | License |
| ---- | ------- |
| Pioneer ML source code | MIT |
| Documentation | MIT |
| Contributor Covenant (Code of Conduct) | CC BY 4.0 (upstream) |

## Approved dependency licenses

Core and optional dependencies should use licenses compatible with MIT distribution.
The following are **automatically approved** for direct dependencies:

| License family | Examples | Notes |
| -------------- | -------- | ----- |
| MIT | MIT, MIT License | Preferred |
| BSD | BSD-2-Clause, BSD-3-Clause, BSD License | Permissive |
| Apache | Apache-2.0, Apache Software License | Permissive; attribution in notices |
| ISC | ISC License | Permissive |
| PSF | Python Software Foundation License | Standard for Python ecosystem |
| MPL | Mozilla Public License 2.0 | Weak copyleft; OK as dependency |
| Unlicense / CC0 | Public domain dedications | Permissive |

## Licenses requiring review

The following require maintainer review before adding as a direct dependency:

| License | Reason |
| ------- | ------ |
| GPL-2.0 / GPL-3.0 | Strong copyleft |
| LGPL | Copyleft (may be acceptable as dynamic dependency) |
| AGPL | Network copyleft |
| Proprietary / UNKNOWN | Not open source |
| NVIDIA proprietary | Bundled with CUDA/PyTorch wheels |

## Optional extras and license notes

| Extra | Primary licenses | Review notes |
| ----- | ---------------- | ------------ |
| Core | MIT, BSD, Apache | No copyleft |
| `llm` | MIT, Apache | OpenAI SDK is Apache-2.0 |
| `serve` | MIT, BSD | FastAPI, Uvicorn, Starlette |
| `rag` | Apache, MIT | ChromaDB is Apache-2.0 |
| `training` | BSD, MIT, **NVIDIA proprietary** | PyTorch/CUDA wheels include proprietary NVIDIA components |
| `observability` | Apache-2.0 | OpenTelemetry |

Install only the extras you need. Production deployments using `training` should
review NVIDIA and PyTorch license terms separately.

## Compliance workflow

```bash
# Regenerate third-party license inventory
make licenses

# Verify core dependencies against allowlist (CI gate)
make license-check

# Audit optional [all] extras (informational; may include proprietary wheels)
make license-check-all
```

## Attribution requirements

When distributing Pioneer ML or derivative works:

1. Include the [LICENSE](../LICENSE) file.
2. Include [licenses/THIRD_PARTY_LICENSES.md](../licenses/THIRD_PARTY_LICENSES.md)
   for bundled dependencies.
3. Do not remove existing copyright or license notices from source files.

## Adding new dependencies

1. Check the package license on PyPI or the upstream repository.
2. Confirm it matches the approved list above.
3. Add the dependency to `pyproject.toml`.
4. Run `make licenses && make license-check`.
5. Update this document if the dependency introduces a new license category.

## References

- [MIT License](https://opensource.org/licenses/MIT)
- [OSI Approved Licenses](https://opensource.org/licenses)
- [SPDX License List](https://spdx.org/licenses/)
