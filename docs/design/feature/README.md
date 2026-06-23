# Feature Design Documents

One Markdown design document per feature, named `<slug>.md` (e.g.
`image-quality-label.md`). Write the design **before** you create the feature —
it is the up-front contract the feature creation then implements.

This is the design-first (Specify) phase of `/deriva-ml:create-feature`
(Phase 1). Author docs here with `/deriva-ml:design-experiment`, which carries
the standardized feature template (Purpose · Requirements · Validation ·
Upstream designs · Status & links) and a worked example.

## How this relates to the other records

- **This directory** = the **plan**: what the feature captures and why, the
  target table, the feature type/vocabulary, who writes the values, and how
  you'll validate it (coverage, value sanity, provenance, the consumer can read
  it). Written first, then implemented by `create-feature`.
- **`tacit-knowledge.md`** (project root) = the **running journal**: what you
  *learned* creating and populating it. The two cross-link.
- **The catalog** holds the factual provenance — the feature definition, its
  values, and the executions that produced them.

Note the input-vs-output distinction: a feature the model *trains on* is an
input (the model-design names it upstream); a feature the model *predicts* is
an output (this feature-design names the producing model-design). Keeping that
straight is what keeps the design dependency graph acyclic.
