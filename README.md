# Physics of Complex Networks: Project Repository

**Student:** Jacopo Bagno  
**Student ID:** `[replace with your student ID]`  
**Academic year:** `[replace if needed]`

This repository contains the numerical implementations, processed data,
publication-ready figures, and LaTeX sources developed for the Physics of
Complex Networks course projects.

> **Before submission:** replace every value enclosed in square brackets with
> your personal information and final Moodle scores.

---

## Selected Course Projects

| Task # | Project name | Type | Moodle score |
|:---:|---|---|:---:|
| #8 | Dark Web: reproduction of the De Domenico--Arenas model | Network-model reproduction | `[add score]` |
| #39 | Epidemic spreading on cruise-ship temporal contact networks | Data project | `[add score]` |

## Repository contents

| Path | Contents |
|---|---|
| `main.tex` | Main LaTeX document that combines the two projects. |
| `sections/` | Report sections, including the Dark Web and cruise-ship analyses. |
| `code/` | Python scripts for network generation, measurements, and simulations. |
| `results/` | Generated numerical outputs and figures used in the report. |
| `report.pdf` | Current compiled version of the report. |

## Building the report

Compile `main.tex` from this directory with a LaTeX distribution such as
MiKTeX or TeX Live. The resulting PDF is `main.pdf`; `report.pdf` is the
submission-ready copy.

## Reproducibility

The scripts in `code/` generate the synthetic networks, compute their
structural properties, and run the epidemic simulations. The processed
cruise-ship CSV files are kept in the external data directory referenced by
the simulation scripts; update those paths if the repository is moved to a
different computer.
