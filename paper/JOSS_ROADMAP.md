# JOSS Submission Roadmap

Roadmap for submitting GastroPy to the
[Journal of Open Source Software](https://joss.theoj.org/).

JOSS requires **six months of public development history** with evidence of
releases, issues, and pull requests. The earliest eligible submission date
is approximately **August 2026**.

---

## Phase 1 — Community & Packaging (immediate)

- [x] Add `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- [x] Add `CHANGELOG.md` (v0.1.0 release notes)
- [x] Add GitHub issue templates (bug report, feature request)
- [x] Add GitHub pull request template
- [x] Add author email, affiliation, and ORCID to `pyproject.toml` and `CITATION.cff`
- [ ] Create tagged `v0.1.0` release on GitHub
- [ ] Open issues for planned work (`io`, `stats`, `eeg`/`meg` modules)
- [ ] Register repository on Zenodo for automatic DOI minting

## Phase 2 — Paper Draft (months 1-2)

- [x] Write skeleton `paper.md` (750-1750 words) with required JOSS sections
- [x] Write `paper.bib` with key references
- [ ] Flesh out Statement of Need (gap in Python EGG tooling)
- [ ] Flesh out State of the Field (NeuroKit2, MNE, MATLAB toolboxes)
- [ ] Flesh out Software Design (layered architecture, composability)
- [ ] Draft Research Impact Statement (semi_precision pipeline, reproducibility)
- [ ] Write AI Usage Disclosure
- [ ] Iterate with co-authors / reviewers

## Phase 3 — Sustained Development (months 1-6)

Use issues and pull requests for **all** changes from here forward.

- [ ] Implement `gastropy.io` (data I/O, BIDS reader)
- [ ] Implement `gastropy.stats` (statistical testing utilities)
- [ ] Release v0.2.0 (new features)
- [ ] Release v0.x.y bugfix releases as needed
- [ ] Seek early adopters and collect feedback via issues
- [ ] Optional: publish preprint to establish research impact
- [ ] Optional: recruit co-authors who contribute code, methodology, or testing

## Phase 4 — Pre-Submission (~month 5-6)

- [ ] Finalize `paper.md` — self-review against JOSS checklist
- [ ] Bump to v1.0.0 for submission release
- [ ] Verify Zenodo DOI mints correctly for the release
- [ ] Run through the full JOSS review checklist (below)
- [ ] Submit at https://joss.theoj.org

---

## JOSS Review Checklist

Reviewers will verify each of these items. Check them off before submitting.

### General

- [ ] Repository is publicly accessible
- [ ] OSI-approved `LICENSE` file present (MIT)
- [ ] Submitting author made major contributions; author list is appropriate
- [ ] Software demonstrates research impact or scholarly significance

### Development History

- [ ] 6+ months of public history with releases, issues, and PRs
- [ ] Commit history shows iterative development
- [ ] Follows open-source best practices (licensing, docs, tests, releases)

### Functionality

- [ ] Installation proceeds as documented
- [ ] Core functional claims confirmed
- [ ] Performance claims verified (if any)

### Documentation

- [ ] Statement of problems solved and target audience
- [ ] Dependency list with automated package management
- [ ] Usage examples solving real-world problems
- [ ] Core functionality documented (API reference)
- [ ] Automated test suite
- [ ] Community guidelines for contributions and issue reporting

### Software Paper (paper.md)

- [ ] Summary for non-specialist audience
- [ ] Statement of Need (problems, audience, relation to existing work)
- [ ] State of the Field (comparison, build-vs-contribute justification)
- [ ] Software Design (architecture, trade-offs)
- [ ] Research Impact Statement (evidence of impact)
- [ ] AI Usage Disclosure
- [ ] 750-1750 words
- [ ] References are complete with full venue names

### Post-Acceptance

- [ ] Create tagged release
- [ ] Deposit with Zenodo/figshare to obtain DOI
- [ ] Update JOSS review thread with version and DOI
