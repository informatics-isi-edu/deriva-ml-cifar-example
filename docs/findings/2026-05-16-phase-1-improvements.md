# Phase 1 findings — improvement opportunities

**Source:** End-to-end DerivaML platform test session, Phase 1 (catalog bootstrap + direct/indirect verification).
**Date:** 2026-05-16.
**Companion docs:**
- Session journal: `docs/e2e-test-2026-05-13-journal.md`
- Already-filed bug write-up: `docs/bugs/2026-05-16-bag-loader-stale-path-builder.md`

The items below are grouped by repo so each owner can pick up the
relevant set. Each item lists what was observed, why it matters, and
a concrete (suggested) fix shape.

---

## A. Root cause of the original question: why didn't the MCP **resources** get used?

This was the user's flagged finding and the spine of the rest of this
document. Recording the diagnosis up front so the rest of the items
can be read as consequences.

The deriva MCP server's tool catalog surfaced with explicit
server-level instructions (delivered as a `<system-reminder>`):

> "DERIVA MCP server for catalog introspection, querying, entity
> CRUD, annotation management, and file operations. Call the
> relevant guide prompt (query_guide, entity_guide, annotation_guide,
> catalog_guide) before your first use of each tool group in a
> conversation."

The server exposes **six guide prompts** (`query_guide`, `entity_guide`,
`annotation_guide`, `catalog_guide`, `deriva_ml_concepts`,
`deriva_ml_getting_started`) and **10+ ML resource templates**
(`ml_datasets`, `ml_dataset_detail`, `ml_dataset_spec`,
`ml_dataset_members`, `ml_workflows`, `ml_workflow_detail`, ...).
The agent skipped both — read no prompts, used no resources, went
straight to tools.

The mechanical reasons:

1. **The instruction was buried.** The `<system-reminder>` arrived
   alongside a list of 122 tool names. The tool list dominates
   attention.

2. **Claude Code's tool-discovery affordances are tool-shaped, not
   resource-shaped.** `ToolSearch` and the deferred-tool catalog
   are explicitly for tools. There's no equivalent "resource
   search" or "prompt search." `ListMcpResourcesTool` exists but
   has to be remembered, not auto-surfaced.

3. **The deferred-tool name list looked complete.** With
   `deriva_ml_list_datasets`, `deriva_ml_get_dataset`,
   `deriva_ml_list_features`, etc. all visible, the resource
   templates feel redundant. They're not — resources are snapshot-
   shaped (single read returns up to 1000 rows + truncation hint),
   tools are pagination-shaped (preflight → page → advance).
   "Verify the state of a freshly-created catalog" wants snapshots.

4. **Stale habits.** The agent had spent the whole session using
   `deriva-ml` Python API (`ml.find_datasets()`, etc.). When
   switching to the indirect channel, it looked for tool-shaped
   analogs of the methods it already knew. Resources never
   entered the search space.

5. **No skill or prompt enforced the read-the-guide step.** No
   deriva-ml skill said "read `deriva_ml_getting_started` before
   you start." The MCP server said so; the server's instruction
   was easy to skip.

Everything below traces back to these mechanics. Each fix targets
a specific link in the chain.

---

## B. `deriva-skills` and `deriva-ml-skills`

**Repo:** `informatics-isi-edu/deriva-skills`,
`informatics-isi-edu/deriva-ml-skills`.

### B1. New skill: read-mcp-guide-first

Add a skill in `deriva-ml-skills` (or possibly tier-1 in
`deriva-skills` since the four core guides — `query_guide` etc. —
are tier-1) whose only job is:

> Before your first deriva-mcp call in a conversation, read the
> `deriva_ml_concepts` and `deriva_ml_getting_started` MCP prompts.
> When reaching for a `deriva_ml_*` tool, also read the relevant
> per-group guide prompt.

Trigger phrases should be high-confidence for any context that
involves a Deriva catalog operation — "list datasets", "check
catalog state", "look at this catalog", "verify schema", etc. The
skill body is short: it tells the agent to fetch the guide prompts
via the MCP protocol's `prompts/get` method (or whatever Claude
Code wrapper surfaces it), or to invoke `ListMcpResourcesTool` /
read each prompt by URI.

This mirrors the server's instruction at the layer where Claude
Code's auto-discovery affordances actually fire. The MCP server's
instruction is correct but invisible at use-time.

### B2. Cross-reference resource templates from existing skills

Every existing skill that reaches for `deriva_ml_*` list-style tools
(`dataset-lifecycle`, `compare-model-runs`, `create-feature`,
`work-with-assets`, etc.) should mention the corresponding **resource
template** as the preferred path for snapshot reads, and the tool
as the path for paginated drilling.

Concrete: in `dataset-lifecycle/SKILL.md`, when the agent is
verifying or browsing existing datasets, prefer
`ReadMcpResourceTool(server="...", uri="deriva://catalog/{h}/{c}/ml/datasets")`
over `deriva_ml_list_datasets`. Same shape decisions for
workflows, executions, features.

### B3. `maintain-experiment-notes` should capture entry-point choices

Today this skill triggers on dataset / feature / workflow
*content* decisions. It should also trigger on **plumbing decisions
the future maintainer would need to understand** — "I read this via
resource X rather than tool Y because the catalog was fresh and a
snapshot was sufficient." These aren't experiment-design decisions
in the narrow sense, but they're tacit-knowledge decisions of the
same character.

Add a section to the skill's description / examples covering this
shape of decision: "MCP entry-point choices when there's more than
one valid path."

---

## C. `deriva-mcp-core` (the MCP server framework)

**Repo:** `informatics-isi-edu/deriva-mcp-core`.

### C1. Surface the guide prompts more aggressively

> **Status: PARTIALLY DONE (C1c) + remainder NOT PLANNED — closed 2026-05-17.**
> **C1c is shipped** — `deriva_ml_getting_started` and `deriva_ml_concepts`
> are exposed as both prompts AND resources at
> `deriva://deriva-ml/getting-started` and `deriva://deriva-ml/concepts`
> (resource handlers in `deriva-ml-mcp/src/deriva_ml_mcp/resources/ml.py`).
> Resource-walking clients now find them.
> **C1a, C1b, C1d are NOT PLANNED** — no current resourcing for further
> `deriva-mcp-core` changes. The user-facing impact of C1 (agents skipping
> the orientation) is mitigated in `deriva-ml-skills` by the
> `using-deriva-mcp` skill ([PR #17](https://github.com/informatics-isi-edu/deriva-ml-skills/pull/17))
> which encodes the cold-start discipline at the skill layer using the
> resource form C1c provides. Re-open C1a/b/d if non-Claude-Code clients
> need server-side enforcement.

Server-level instructions ("Call the relevant guide prompt...") are
correct but easily missed. Options, in increasing order of
intrusiveness:

- (C1a) Repeat the "read the guide prompt" reminder in each tool's
  `description` field. Per-tool repetition is ugly but actually-read.
- (C1b) Emit a structured `prompts` payload alongside the tool
  catalog on `initialize`. Today prompts require a separate
  `prompts/list` round-trip — almost no client does that
  automatically.
- (C1c) Expose `deriva_ml_getting_started` as both a `prompt` AND a
  `resource` (URI like `deriva://meta/getting-started`, always
  reachable). Resource-walking clients find it.
- (C1d) Make the server's `<system-reminder>` text itself include a
  preamble like "BEFORE USING ANY TOOL: read the
  `deriva_ml_getting_started` prompt." Doesn't help clients that
  collapse all server reminders, but helps the ones that show them
  verbatim.

C1c is the most universal — agents that don't read prompts often
do read resources.

### C2. Document the resource-vs-tool decision

> **Status: NOT PLANNED at server layer — mitigated at skill layer.** Closed
> 2026-05-17. No current resourcing for `deriva-mcp-core` / `deriva-ml-mcp`
> prompt edits. The teaching is now carried in `deriva-ml-skills`:
> `deriva-ml-context`'s "Read-side questions: fetch the resource first"
> section is the canonical statement, reinforced by individual skills
> (`dataset-lifecycle`, `compare-model-runs`, `create-feature`,
> `work-with-assets`) that cross-reference the resource alternative next
> to their `deriva_ml_list_*` examples ([PR #18](https://github.com/informatics-isi-edu/deriva-ml-skills/pull/18)).
> This addresses the user-visible problem (Claude Code agents defaulting
> to tools when a resource would serve) but leaves non-Claude-Code clients
> with no server-side guidance. Re-open if those clients matter.

In `deriva_ml_getting_started` (or a new `entry_points_guide`),
add a "when to use a resource vs a tool" section. The shape:

- **Resource** — single read, snapshot-shaped, suitable for
  verification / introspection / "show me what's there." Capped at
  some bound (1000 rows for `ml_datasets`); truncated reads return
  a hint to switch to the tool.
- **Tool** — paginated, cursor-based, suitable for full enumeration
  or interactive drilling. Pay the preflight cost (or skip it for
  fixed-shape queries).

Without this, agents like me default to tools because the deferred-
tool catalog is the affordance Claude Code surfaces.

### C3. Auth-failure error wording

> **Status: NOT PLANNED — closed 2026-05-17.** No current resourcing for
> `deriva-mcp-core` server-side changes. Improving the error wording is
> still the right fix and remains a low-risk targeted change for whoever
> picks it up; the finding's recommended replacement text is good as-is.
> Re-open when there's bandwidth for `deriva-mcp-core` work.

When the MCP server has no auth context and the underlying ERMrest
returns 401, the server currently surfaces:

```
{"error": "You are not authorized to access this catalog. Please
check your credentials and make sure you have logged in."}
```

This points the user at the user-credential layer (where the
credenza session is fine), not at the MCP-server-doesn't-have-OAuth-
wired layer. Better wording:

```
{"error": "This MCP server has no auth context for the request.
The connecting client must be configured with OAuth — see
deriva-ml-mcp README §'Connecting Claude Code to the dockerized
server', or use the stdio recipe for single-user dev."}
```

Distinguish "client wasn't sent any credentials" (config issue)
from "credentials were sent but rejected" (user-credential issue).
Today both look the same to the caller.

### C4. Resource reads should fail fast, not time out

> **Status: NOT PLANNED — closed 2026-05-17.** No current resourcing for
> `deriva-mcp-core` server-side changes. The 30s hang on auth-walled
> resource reads remains a real user-experience issue; the fix (apply
> the same auth-detection envelope to resource handlers as the tools
> already have) is well-scoped and inexpensive when bandwidth opens up.
> Re-open then.

Same auth condition that gives the tools a 401 made the resource
reads time out at 30s (MCP error 32001). The resource path should
return a structured error like the tools, not silently hang.
Symptom: I spent ~30s waiting for `ReadMcpResourceTool` against
a deriva://catalog/.../ml/datasets URI before realizing it was an
auth wall.

Add the same auth-detection envelope to the resource handlers.

### C5. Per-server attribution in `<system-reminder>` collapses

> **Status: NOT PLANNED — closed 2026-05-17.** Primarily a Claude Code
> rendering issue (per the finding's own note) rather than something
> `deriva-mcp-core` can fix definitively. No current resourcing for
> server-side mitigation either. If multiple-MCP-server agent confusion
> becomes a recurring debugging cost in real sessions, the server-side
> prefix workaround (`[deriva-mcp-core] ...` in its instruction text)
> is a one-line change worth re-opening for.

The aggregated `<system-reminder>` for "MCP Server Instructions"
concatenates all servers' instructions into one block. With one
server, fine. With multiple, the agent has to figure out which
server an instruction belongs to. Some lightweight per-server
attribution (`## <server-name>` headings) would help.

This is more a Claude-Code-side rendering issue than a server-side
one, but the server could mitigate by prefixing its own instruction
with `[deriva-mcp-core] ...`.

---

## D. `deriva-ml-mcp` (the deriva-ml plugin loaded by the core)

**Repo:** `informatics-isi-edu/deriva-ml-mcp`.

### D1. Cross-link to the model-template README

The deriva-ml-mcp README §"Connecting Claude Code to the dockerized
server" has the correct recipe (`claude mcp add -t http
dev-localhost https://localhost/mcp --client-id deriva-mcp
--callback-port 8080`). The model template's README §"Set Up
Claude Code (Optional)" has a different, incorrect-for-dev recipe
(`docker run -i --rm ... ghcr.io/informatics-isi-edu/deriva-mcp`).

Either:
- Refer the model template README directly at the deriva-ml-mcp
  README for the dev recipe, OR
- Lift the deriva-ml-mcp recipe verbatim into the model template
  with attribution.

Today they diverge silently, and the agent (and any new dev) hits
the wrong recipe.

### D2. The "pre-release" caveat box in the README is load-bearing

Currently:

> "Pre-release status. This workflow exists because deriva-docker
> support for the deriva-ml-mcp plugin is currently pre-release.
> When the final deriva-docker release ships, this section and the
> helper script will be replaced by deriva-docker's canonical
> workflow."

For new users this reads as "skip this section, it'll be replaced."
It's actually the only working dev-localhost recipe. Either tag
it more visibly as "current canonical recipe" or remove the
pre-release language until deriva-docker's canonical workflow is
actually available.

### D3. Guide-prompt content for `deriva_ml_getting_started`

The prompt body (I read the first ~60 lines) covers the (hostname,
catalog_id) rule and the pagination contract. Two additions would
have helped me in Phase 1:

- A "resources vs tools" section (matches C2 above).
- A pointer to the deriva-mcp-core core guides (`query_guide`,
  `entity_guide`, `annotation_guide`, `catalog_guide`) for clients
  that haven't yet read them.

The prompt already says it's layered above core; making the
layering visible (which guides apply at which layer) would help an
agent navigate.

---

## E. `deriva-ml` (the Python library)

**Repo:** `informatics-isi-edu/deriva-ml`.

### E1. `ml.refresh_schema()` doesn't suppress the stale-cache warning

> **Status: NOT A BUG — the cache round-trip is correct.** Investigated
> 2026-05-17 (deriva-ml PR #162). The 8 unit tests in
> `tests/core/test_refresh_schema_warning.py` pin the
> correctness invariants: `SchemaCache.write` persists the
> snapshot, the on-disk file authoritatively reflects the latest
> write, the warning has exactly one source in `core/base.py`,
> and `refresh_schema` is structurally wired to read the live
> snapshot and write it to the cache in that order. The reported
> behavior is one of three environmental cases (multi-tenant
> catalog advancement; silent `refresh_schema` exception swallowed
> by an outer try; `working_dir` mismatch between sessions). If
> reported again, ask the user which of those three applies
> before opening a bug. **No code change in deriva-ml.**

The warning fires:

```
WARNING:deriva_ml.core.base:schema cache is at snapshot
353-YMFJ-98W2; live catalog is at 353-YMJ5-C4KJ.
Using cached schema. Call ml.refresh_schema() to update.
```

When I explicitly call `ml.refresh_schema()` immediately after
construction, the warning still fires for subsequent operations.
Either the warning checks a different cache than `refresh_schema()`
updates, or `refresh_schema()` isn't actually invalidating the
right layer.

Repro:
```python
ml = DerivaML(hostname="localhost", catalog_id="8", check_auth=True)
ml.refresh_schema()  # explicit refresh
# Now use ml.find_datasets(), ml.find_features(...), etc.
# Warning still fires for these subsequent calls.
```

### E2. Inconsistent verb: `find_features` vs `list_*`

> **Status: NOT A BUG — the convention is intentional.** Resolved
> 2026-05-17 by deriva-ml PR #163 (DerivaML class docstring) +
> deriva-ml-skills#15 (deriva-ml-context skill addition). The
> `find_*` vs `list_*` distinction is the project convention:
> `find_*` is a catalog-wide search of entities of a kind
> (optionally filtered); `list_*` is enumeration scoped to a
> specific parent entity passed as the first argument. The
> muscle-memory miss was a discoverability issue, not an
> inconsistency. `ml.list_features()` doesn't exist because
> features aren't scoped to a parent entity the way dataset
> members are scoped to a dataset — the catalog-wide search is
> `ml.find_features()`. **No code change in deriva-ml.**

- `ml.find_features(table)` ✓ exists
- `ml.list_features()` ✗ doesn't exist
- `ml.list_assets("Image")` ✓ exists
- `ml.list_workflows()` ✓ exists
- `ml.list_executions()` ✓ exists

The plan I wrote had `ml.list_features()` from muscle memory. Real
code crashed with `AttributeError: 'DerivaML' object has no
attribute 'list_features'. Did you mean: 'find_features'?`

**Resolution analysis:** This finding's premise (the inconsistency)
was incorrect. The library's full call surface uses the two
prefixes consistently:

| Prefix | Source | Examples |
|---|---|---|
| `find_*` | Live catalog search | `find_features`, `find_datasets`, `find_workflows`, `find_executions`, `find_experiments`, `find_assets`, `find_incomplete_executions` |
| `list_*` | Local workspace or scoped enumeration | `list_assets(table)`, `list_dataset_members(dataset)`, `list_dataset_children(dataset)`, `list_workflow_executions(workflow)`, `list_vocabulary_terms(table)`, `list_executions(...)` (local SQLite registry; distinct from `find_executions` which queries ERMrest) |

The two `*_executions` methods are the load-bearing example
that disambiguates the rule: `find_executions(...)` hits the
live catalog; `list_executions(...)` reads the local registry
in `~/.deriva-ml/.../workspace.db`. The bullets in this
finding's original text said "`ml.list_workflows()` ✓ exists"
and "`ml.list_executions()` ✓ exists" — `list_executions` is
real (it's the registry reader), and **`list_workflows` does
not exist** on `DerivaML` (the catalog-search is `find_workflows`).
So even the original finding's spot-checks were not what they
seemed.

The fix is to document the convention where LLMs and humans
will see it — done in two places (Python class docstring;
always-on skill). Closed as documentation, not as behavior
change.

Two reasonable fixes:
- Add `list_features` as an alias of `find_features`.
- Rename `find_features` → `list_features` (deprecate the old
  name on a release boundary).

### E3. `ml.feature_values(...)` returns a generator, not a list

Every other `list_*` returns a list. `feature_values()` returns a
generator, which silently breaks code that does `len(values)` or
`values[:5]`.

Fix: either standardize to a list (and add an explicit
`iter_feature_values` for streaming), or rename the method to
something that signals lazy iteration (`iter_feature_values`,
`stream_feature_values`).

### E4. Schema-mutating methods don't invalidate the path-builder cache

> **Status: NOT NEEDED NOW — closed without code change.** Closed
> 2026-05-17 after auditing every ``getPathBuilder()`` consumer
> in ``deriva-ml/src/``. The load-bearing failure (fresh-catalog
> upload flow) was fixed upstream in deriva-py's
> ``BagCatalogLoader._ensure_path_builder()`` (calls
> ``getPathBuilder(refresh=True)`` on first build). Verified by
> deriva-ml PR #161, pinned by the unit tests in
> ``tests/execution/test_bag_loader_path_builder_refresh.py``.
> Every remaining ``getPathBuilder()`` consumer in deriva-ml
> reads from tables that exist at catalog creation time
> (``deriva-ml`` schema, ``public.ERMrest_RID_Lease``) — none of
> them are vulnerable to in-process schema mutations because the
> tables they touch don't change after ``create_ml_schema``. An
> invalidation hook on ``create_asset`` / ``create_vocabulary`` /
> etc. would be defensive code with no demonstrable consumer.
> **If a future deriva-ml caller is added that mutates the
> catalog schema and then queries the path-builder for the
> newly-added table in the same process, E4 should be revisited
> at that point.**

The bag-loader stale-path-builder bug (filed separately as
`docs/bugs/2026-05-16-bag-loader-stale-path-builder.md`,
fix landed in deriva-py `de19aaf1`) has its **root cause** in
deriva-ml: every method that mutates the catalog schema
(`create_asset`, `create_vocabulary`, `add_term`, `create_feature`,
`add_column`, etc.) leaves a stale `ErmrestCatalog._path_builder_cache`
behind.

The deriva-py fix is defensive (bag-loader passes `refresh=True`
unconditionally). It paid the cost of an extra `/schema` walk per
loader instance. A cleaner deriva-ml-side fix: after every schema
mutation, invalidate the cache:

```python
def create_asset(self, ...):
    # ... do the work ...
    self.catalog._path_builder_cache = None
    # or: self.catalog.getPathBuilder(refresh=True)
```

With that in place, the deriva-py defensive `refresh=True` would
still be correct but redundant.

**Audit conclusion (2026-05-17):** Of the 9
``getPathBuilder()`` call sites in ``deriva_ml/src/``:

- 3 are in ``catalog/clone_via_bag.py`` /
  ``catalog/localize.py`` / ``schema/create_schema.py`` —
  these run on freshly-connected catalogs (no prior schema
  mutations in the same process to be stale relative to).
- 3 are in ``execution/state_machine.py`` — they write to
  the ``Execution`` table (a ``deriva-ml`` schema table
  created at ``create_ml_schema`` time, never mutated after).
- 1 is in ``execution/rid_lease.py`` — reads from
  ``public.ERMrest_RID_Lease`` (system table, present from
  catalog creation).
- 1 is in ``dataset/bag_builder.py`` — reads from
  ``Dataset_Dataset`` (deriva-ml schema, present from
  ``create_ml_schema``).
- 1 is in ``core/mixins/path_builder.py`` — the public
  ``pathBuilder()`` accessor; the user is responsible for
  freshness here, and ``DerivaML.refresh_schema()`` is the
  documented escape hatch.

The bag-loader path was the only load-bearing consumer that
hit the stale-cache scenario, and that's fixed upstream.

### E5. Bootstrapping ships an opinionated vocab term set

> **Status: FIXED.** Resolved by deriva-ml PR #164 (released in
> v1.36.5, 2026-05-17). Four domain-specific terms removed from
> the ``Workflow_Type`` seeds in ``initialize_ml_schema``:
> ``VGG19``, ``RETFound``, ``Multimodal`` (specific architectures /
> model / research-area), plus ``Embedding`` (its description
> named "foundation models" — also project-orientation).
>
> The remaining 9 ``Workflow_Type`` seeds describe genuine
> cross-domain workflow shapes: ``Training``, ``Testing``,
> ``Prediction``, ``Feature_Creation``, ``Visualization``,
> ``Analysis``, ``Ingest``, ``Data_Cleaning``,
> ``Dataset_Management``.
>
> ``initialize_ml_schema``'s docstring now records a
> "Term-selection principle" that future contributors land on
> before proposing a new term: every seeded term must describe a
> platform-level concept, not a domain-specific one. The new
> regression test ``tests/schema/test_seeded_vocab_terms.py``
> (8 tests) pins both what stays seeded and what must not be.
> A future PR that re-adds ``VGG19``/``RETFound``/``Multimodal``
> fails the regression test and gets forced to either
> demonstrate platform-level relevance or move the term to a
> per-project initializer.
>
> Existing-catalog impact: none. ``_ensure_terms`` is
> skip-if-exists; already-initialized catalogs keep their terms.

A fresh catalog gets:
- 17 `Workflow_Type` terms including `RETFound`, `VGG19`,
  `Embedding`, `Multimodal`
- 8 `Dataset_Type` terms including `Validation`, `File`

Some of these are clearly project-specific. The CIFAR loader
adds three more (`CIFAR_Data_Load`, `Image Classification`,
`ROC Analysis Notebook`). A user adapting the template for their
own data inherits the platform's defaults, which may not match
their domain.

Options:
- Prune the defaults to a true minimum (probably:
  `Workflow_Type` = `Training`, `Testing`, `Prediction`,
  `Analysis`; `Dataset_Type` = `Training`, `Testing`, `Validation`,
  `Complete`, `Labeled`, `Unlabeled`, `Split`).
- Split into a "core" set and an optional "examples" extension that
  ships its own vocab terms when loaded.

The current behavior — load every term ever — is the same shape as
the "ship every config you've ever used" anti-pattern.

---

## F. `deriva-py` (the underlying client library)

**Repo:** `informatics-isi-edu/deriva-py`.

### F1. `ErmrestCatalog.getPathBuilder()` cache is invalidation-free

This is the substrate beneath E4. Docstring acknowledges the
staleness window only for cross-process schema changes:

> "Schema rows added by another process between calls are not seen
> unless the caller passes refresh=True; this is the same staleness
> window every other catalog-model read has."

But the same-process scenario — schema mutation followed by read —
is much more common in test harnesses, fresh-catalog loaders, and
init scripts. The bag-loader fix (de19aaf1) handles one consumer's
exposure. The underlying library could:

- Auto-invalidate `_path_builder_cache` after any successful
  `POST /schema/...` made through the same `ErmrestCatalog`
  instance. This is the cleanest fix and prevents every future
  consumer from making the same mistake.
- Or: provide a callback registration so schema-mutating layers
  (deriva-ml, custom code) can wire their mutations into the
  invalidation.

### F2. Cheap "is the cache stale?" check

`getPathBuilder(refresh=True)` walks the full `/schema` every time
it's called. The bag-loader fix pays that cost once per loader
instance — acceptable. If other layers start sprinkling
`refresh=True` defensively, the cost compounds.

A cheaper check using the catalog's snapshot ETag or version
counter would let consumers conditionally refresh:

```python
def getPathBuilder(self, refresh=False, if_stale=False):
    if if_stale:
        current_snap = self._head_snapshot()  # cheap HEAD or GET /
        if current_snap == self._path_builder_snap:
            return self._path_builder_cache
        refresh = True
    # ... rest as today ...
```

### F3. Auth state should be opaque to end users — but currently isn't

**Principle: the user should never know `~/.deriva/credential.json`
exists.** Auth is one thing the user does ("log in") and one thing
they see ("am I logged in?"). The fact that there's a file on disk
holding an access token, that the token is backed by a credenza
session, that these can diverge — these are implementation details
that should never leak.

In Phase 1, the implementation leaked badly:

- The first 401 fired with a generic "Access requires
  authentication. Detail: catalog" message. No indication that
  this was *the user's* problem to fix (vs. a server config issue),
  nor what action to take.
- The agent (me) had to inspect `~/.deriva/credential.json` to
  diagnose. The Claude Code auto-mode classifier *correctly*
  blocked that as a credential-exploration action — but the user
  workflow shouldn't have made it the obvious diagnostic step in
  the first place.
- `deriva-credenza-auth-utils get-session` is the only way to
  distinguish "session live" from "session expired but cached
  token still works for reads." But the user shouldn't need to
  run a second diagnostic CLI to interpret a 401.
- Two distinct failure modes ("you never logged in" vs "your
  session has expired") give the same 401 with the same generic
  text.

The fix is *not* to expose more of the auth internals — that's
the wrong direction. It's to consolidate the user-facing surface:

- **`deriva-py` should self-diagnose 401s** and translate them into
  user-actionable error text. When a 401 comes back from a
  write-style operation on a host the user has a stored token for,
  the message should be "Your `<host>` login has expired. Run
  `deriva-credenza-auth-utils --host <host> login` to refresh."
  Not "Access requires authentication. Detail: catalog."
- **`deriva-py` should provide a single `is_authenticated(host)`
  helper** that returns one of three values: `unauthenticated` /
  `expired` / `live`. Callers (deriva-ml, the MCP server, the
  model template's startup banner) use that one helper rather than
  poking at credential.json directly or shelling out to
  `get-session`.
- **No skill, doc, or template should ever instruct the user to
  read or edit `~/.deriva/credential.json`.** Today none of the
  template docs do this, but the agent ended up there anyway
  because the auth failure was opaque. Treat the file as a
  deriva-py implementation detail.

Today's user-facing auth surface should be exactly:

| Action | Tool |
|---|---|
| "Log in" | `deriva-credenza-auth-utils --host <h> login` |
| "Log out" | `deriva-credenza-auth-utils --host <h> logout` |
| "Am I logged in?" | (implicit via meaningful error messages) |

Anything else — token storage, session lifecycle, scope claims,
realm IDs — is platform internals.

This finding belongs in deriva-py because deriva-py owns the
client-side auth surface, but the implication propagates into:
- **G2 (auth-model doc)** should be removed or substantially
  scoped down. There's no end-user "auth model" to document
  beyond "log in via the credenza utility." See revised G2 below.
- **The model template's README §"Authenticate"** should stay
  one line: "Log in to your Deriva server: `deriva-credenza-auth-utils
  --host <hostname> login`." No mention of token caches, no
  mention of session lifetime, no troubleshooting beyond
  "re-run if you get an auth error."

### F4. `InsecureRequestWarning` spam against localhost

Every `requests` call to `https://localhost/...` emits an
`InsecureRequestWarning` from `urllib3`. The deriva-localhost cert
chain is self-signed (DERIVA Dev Local CA, valid through 2035) but
Python's `requests` doesn't see the `NODE_EXTRA_CA_CERTS` Claude
Code uses; it only honors `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE`.

Three reasonable fixes:
- deriva-py honors `NODE_EXTRA_CA_CERTS` (probably wrong scope
  but cheap).
- deriva-py looks for a deriva-installed CA bundle in a known
  location (`~/.deriva/ca-bundle.crt`?) and adds it to the
  `requests` session's trust store.
- deriva-py adds a one-time "to silence this for trusted dev
  hosts, set `REQUESTS_CA_BUNDLE=$HOME/.config/deriva/deriva-dev-ca.crt`"
  hint to its logger setup.

Even (c) — the documentation-only fix — would have saved me
multiple minutes of confusion.

### F5. FK-cycle warning log spam — fix in `deriva-py`

**The cycle is by design.** `Dataset ↔ Dataset_Version` is
intentional in the deriva-ml schema (a Dataset has a
`current_version` FK to Dataset_Version; a Dataset_Version has a
`Dataset` FK back). Each direction is operationally needed. The
schema is not changing.

This finding is **purely about output noise from the consumer
side** (`deriva-py`). The fix:

- `deriva-py`'s bag pipeline knows the `Dataset ↔ Dataset_Version`
  cycle is intentional in the deriva-ml schema and that its own
  manual cycle-handling logic handles it correctly. For **that
  specific known-and-handled cycle**, both the upstream SAWarning
  and the bag-loader's own follow-up log line should be silenced.
- Other cyclic FK warnings — schemas where the consumer doesn't
  *know* the cycle is intentional, or where deriva-py's
  cycle-breaker can't safely resolve them — should still surface
  normally. The cycle warnings exist for a reason; they're a
  legitimate signal that a schema design needs scrutiny. **A
  blanket filter would mask future, genuinely problematic
  cycles** in user schemas (e.g., a domain schema someone designs
  with an accidental cycle).

So the fix is **targeted suppression**, not blanket suppression:

| Where | What |
|---|---|
| `deriva-py/bag/schema_io.py:414` | Check whether the cycle being warned about is the known `(Dataset, Dataset_Version)` pair. If yes, silence the SAWarning at this call site. If no, let it propagate normally. |
| `deriva-py/bag/loader.py:248-250` | Same predicate: if `cycle` is the known `Dataset ↔ Dataset_Version` pair, log at `DEBUG`. Otherwise log at `WARNING` so the user/operator sees it. |

Concrete shape — define a small constant somewhere in the bag
module:

```python
# Cycles deriva-py knows how to handle silently because they're
# part of the deriva-ml core schema by design. Anything not in
# this list still warns normally — cycle warnings are how we
# notice accidental schema bugs.
_KNOWN_INTENTIONAL_CYCLES = frozenset({
    frozenset({"deriva-ml.Dataset", "deriva-ml.Dataset_Version"}),
})
```

Then at the two call sites, check the cycle against this set
before deciding whether to surface the warning. If/when the
deriva-ml schema gains another intentional cycle (or another
deriva-* library does), it's a one-line addition to the set.

**Trace of the noise chain (for the deriva-py maintainer):**

- `deriva-py/bag/schema_io.py:414` accesses `metadata.sorted_tables`.
  SQLAlchemy emits `SAWarning: Cannot correctly sort tables ...
  unresolvable cycles between tables "deriva-ml.Dataset,
  deriva-ml.Dataset_Version"`. The warning text contains the
  table names involved — usable for the cycle-identity check.
- `deriva-py/bag/loader.py:248-250` then emits
  `logger.warning(f"Breaking cycle in FK dependencies: {...}")`
  with the cycle list (e.g., `["deriva-ml.Dataset",
  "deriva-ml.Dataset_Version", "deriva-ml.Dataset"]`). The `cycle`
  variable at that point holds the cycle directly; check it
  against `_KNOWN_INTENTIONAL_CYCLES` before logging.

The cycle stays. Targeted noise goes. General cycle warnings
keep working — which is what they're there for.

During every bag-pipeline schema export (Stage 3 of load-cifar10,
plus any `cache_dataset`, `download_dataset_bag`, or similar
operation), the user sees this block emitted repeatedly:

```
SAWarning: Cannot correctly sort tables; there are unresolvable
cycles between tables "deriva-ml.Dataset, deriva-ml.Dataset_Version",
which is usually caused by mutually dependent foreign key
constraints. Foreign key constraints involving these tables will
not be considered; this warning may raise an error in a future
release.
  for sql_table in metadata.sorted_tables:
WARNING:deriva.bag.loader:Breaking cycle in FK dependencies:
  deriva-ml.Dataset -> deriva-ml.Dataset_Version -> deriva-ml.Dataset
WARNING:deriva.bag.loader:Breaking cycle in FK dependencies:
  deriva-ml.Dataset -> deriva-ml.Dataset_Version -> deriva-ml.Dataset
```

This block fires ~6× during a single `load-cifar10` invocation —
once per dataset bag built during Stage 3.

**Status:**
- The warning is benign — the bag-loader has manual cycle-breaking
  that handles the situation that SAWarning is complaining about.
- The FK cycle `Dataset ↔ Dataset_Version` is intentional in the
  deriva-ml schema (a Dataset has `current_version` FK to
  Dataset_Version; a Dataset_Version has `Dataset` FK back).
  SQLAlchemy's `sorted_tables` doesn't handle cycles, so it warns.
- The user-visible WARNING-level log line is **redundant** — it's
  announcing that the bag-loader handled the very thing the
  SAWarning was complaining about.

**Repro:** Run any operation that triggers bag schema export.
Cheapest: `load-cifar10 --hostname localhost --create-catalog X
--num-images 50` — Stage 3 emits the block multiple times.

**Two-line fix (both in `deriva-py`):**

1. **`deriva/bag/schema_io.py:414`** — wrap `metadata.sorted_tables`
   in `warnings.catch_warnings()` to silence the SAWarning once
   the code is known to handle the cycle:

   ```python
   import warnings
   with warnings.catch_warnings():
       warnings.filterwarnings("ignore", category=SAWarning,
                               message="Cannot correctly sort tables")
       for sql_table in metadata.sorted_tables:
           ...
   ```

2. **`deriva/bag/loader.py`** — change `Breaking cycle in FK
   dependencies` from `logger.warning(...)` to `logger.debug(...)`.
   It's an expected operational message about an intentional
   schema feature, not user-actionable.

After both, the warning block disappears from normal user output.
Maintainers debugging schema-traversal issues can still see the
"Breaking cycle" messages by enabling DEBUG logging on
`deriva.bag.loader`.

### F6. `check_auth` probe hits a non-existent endpoint on credenza servers

When `DerivaML(..., check_auth=True)`, the deriva-py constructor
calls `server.get_authn_session()` which GETs
`https://<host>/authn/session`. On credenza-based deriva-localhost
servers, that endpoint **doesn't exist** — credenza uses
`/authn/service/token` and `/authn/session/<id>` patterns, not
`/authn/session`. The 404 from the legacy endpoint gets translated
by deriva-ml into:

```
deriva_ml.core.exceptions.DerivaMLException: You are not authorized
to access this catalog. Please check your credentials and make
sure you have logged in.
```

This is **doubly misleading**:
- The user IS logged in (credenza session valid).
- The check is failing because the URL doesn't exist, not because
  of an auth failure.

**Repro:** Against any deriva-localhost server running credenza
(not the legacy webauthn flow), call:
```python
DerivaML(hostname="localhost", catalog_id="<id>", check_auth=True)
```

**Fix shape:**

This straddles two repos:

- **`deriva-py`** — `server.get_authn_session()` should either
  (a) detect the credenza endpoint shape and probe `/authn/session/<my-session-id>`,
  or (b) fall back to a different probe when `/authn/session`
  returns 404 (e.g., a benign read against `/ermrest/catalog`
  to confirm credentials work).
- **`deriva-ml`** — the error translation in
  `core/base.py:_init_online` should distinguish "auth probe
  failed because credentials are bad" from "auth probe failed
  because the probe endpoint doesn't exist." The current
  catch-all "You are not authorized..." message is wrong for the
  404 case.

This is one of the symptoms of the auth-implementation-leakage
problem (F3) — the user-facing error tells the user to fix
something that doesn't need fixing.

**Workaround for the test session:** use `check_auth=False` in
direct `DerivaML(...)` calls. The catalog reads themselves will
fail with proper 401 if creds are bad, so we don't lose much by
skipping the probe.

---

## H. `deriva-ml-model-template` (the model template repo)

**Repo:** `informatics-isi-edu/deriva-ml-model-template`.

This repo's findings are mostly documentation-shaped — the template
ships docs that point users at platform behaviors, and several of
those pointers are stale relative to the actual current platform.

### H1. Stale auth-CLI references (`deriva-globus-auth-utils` → `deriva-credenza-auth-utils`)

Five files reference the wrong auth CLI:

- `README.md` line 113 (§5 "Authenticate")
- `docs/getting-started/quick-start.md` line 80
- `docs/getting-started/environment-setup.md` lines 142, 150-151,
  171, 178
- `docs/workflow/experiments.md` lines 11, 170
- `src/scripts/_cifar10_schema.py` module docstring (preserved
  from the original `load_cifar10.py` during the A5 refactor of
  this session; my mistake to preserve verbatim without checking)

The replacement command is `deriva-credenza-auth-utils --host
<hostname> login` (note the flag ordering: `--host` is global
before the subcommand). All five references should be updated.

**Severity:** serious — new users following the README hit a 401
they can't diagnose because the docs point them at a CLI that
exists but uses the legacy Globus auth flow that's no longer
operative on credenza-fronted servers. (See also F3 / G2 for the
broader auth-opacity argument: the user should never see this
implementation detail, period — once F3 is addressed, the auth
command name itself becomes less prominent in user-facing docs
because errors auto-redirect.)

**Fix:** single sweep across the five files; one commit on `main`.
Pure docs change, no code.

### H2. MCP setup section points at wrong recipe for dev users

`README.md` §"Set Up Claude Code (Optional)" documents wiring
Claude Code to the published `ghcr.io/informatics-isi-edu/deriva-mcp`
image via stdio transport with a `docker run -i --rm ...`
command. **This is the right recipe for a remote production
DerivaML server. It is wrong for the dev-localhost setup that
template users typically have.**

The correct dev-localhost recipe is in `deriva-ml-mcp`'s README
§"Connecting Claude Code to the dockerized server":

```bash
claude mcp add -t http dev-localhost https://localhost/mcp \
    --client-id deriva-mcp --callback-port 8080
```

**Fix shape:** either (a) cross-link the model template's MCP
section at `deriva-ml-mcp/README.md` for the dev-localhost recipe,
or (b) lift that section into the template README directly.

Bundling note: this overlaps with D1 in section D (which asks
deriva-ml-mcp to coordinate with the model template). The
template-side fix is independent and worth doing even if D1 isn't
yet addressed.

### H3. `NODE_EXTRA_CA_CERTS` requirement isn't documented anywhere

When Claude Code's HTTP MCP transport talks to a self-signed
deriva-localhost server, Node.js's TLS verification rejects the
chain unless `NODE_EXTRA_CA_CERTS` is set in the workspace's
`.claude/settings.local.json` (`env` block). The deriva-localhost
CA cert is at
`/usr/local/share/ca-certificates/deriva-dev-ca.crt` inside the
deriva-webserver container.

Neither the model template README nor any deriva-ml-skills doc
mentions this. New users get a confusing TLS failure or an MCP
server that silently can't reach the HTTPS endpoint.

**Fix:** add a single subsection to README §"Set Up Claude Code
(Optional)" (or to the new dev-localhost recipe per H2) that
covers:

1. Extracting the CA cert: `docker cp deriva-mcp-test:/usr/local/share/ca-certificates/deriva-dev-ca.crt ~/.config/deriva/deriva-dev-ca.crt`
2. Setting `NODE_EXTRA_CA_CERTS=$HOME/.config/deriva/deriva-dev-ca.crt`
   in `.claude/settings.local.json`'s `env` block.
3. Why: Claude Code's MCP HTTP transport runs in Node.js, which
   uses its own CA bundle. Without the trust anchor, the MCP
   connection fails silently.

This is closely related to F4 (the Python-side
`InsecureRequestWarning` against the same self-signed cert). Both
should be documented together — H3 from the Claude Code side, F4
from the Python side.

---

## G. Cross-cutting / test-plan / process

These don't live in any single repo but should be reflected in the
test plan and onboarding docs.

### G1. The E2E test plan's dual-channel verification is incomplete

`docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md` §4
"Test phases" defines the dual-channel verification as:

- Direct: `deriva-py` / `deriva-ml` Python API
- Indirect: `deriva_ml_*` MCP tools

It does NOT mention MCP **resources**. This is the most important
plan correction from Phase 1 alone: indirect verification should
exercise both tool paths AND resource paths, and the diff against
direct should cover both.

Update §4 and §6 ("Direct vs indirect verification — channels") to
treat resources as first-class.

### G2. Auth implementation has three layers, but the user surface should have one

Internally there are three layers:

- Credenza session (server-side, bounded lifetime, OIDC-issued)
- deriva-py token storage on the client (implementation detail —
  see F3 — never user-facing)
- MCP server forwarding (each MCP-server instance needs its own
  auth context; OAuth client-id / callback-port pair from
  `claude mcp add` provides it)

The user-facing surface should be exactly **one** thing per
context the user works in:

- **In a shell:** `deriva-credenza-auth-utils --host <h> login` /
  `logout`. Re-run on failure. Done.
- **In Claude Code with an MCP server:** the OAuth callback flow
  triggered by `claude mcp add -t http ... --client-id ...
  --callback-port ...`. Browser flow, done. The "you need to
  re-OAuth" failure mode should be detected by the MCP client
  layer and trigger a re-callback — the user shouldn't need to
  know whether the underlying problem is shell-side or
  MCP-server-side.

The previous version of this finding asked for an "auth model
diagram" — that was wrong. **The user should never need to see
or care about the model.** Anything we put in front of the user
that requires them to understand the three layers has failed.

What's actually needed:

- **Detect-and-redirect at each user-facing layer.** When the
  shell user gets a 401, the error message tells them to re-run
  `deriva-credenza-auth-utils login`. Period. (See F3.)
- **MCP-side equivalent:** when the MCP server gets a 401 from
  the upstream catalog, surface that to the Claude Code client
  in a way that triggers the OAuth callback re-flow rather than
  asking the user to do anything manually. (See C3.)
- **No "auth architecture" doc that exposes the layers as such.**
  If there's documentation, it's "how to log in" not "how auth
  works internally."

Three-layer awareness IS still needed for *platform maintainers*
— but in a separate maintainer-facing doc that's clearly labeled
"deriva-platform internals" and not in any README the end user
will read.

**Sharpened rule (the only one that matters):** in normal use, the
user authenticates once per session and that is the entire auth
experience. They never see the three layers, never see token
storage, never see scope claims, never see session refresh
mechanics. If they hit a failure, the failing layer detects it
and either auto-redirects to the appropriate re-auth flow or
emits a single user-action error message ("Run X to log in.")
that points at the *one* command that always works. The user
should not need to know which layer failed in order to fix it.

If any normal-use workflow forces the user to know more than
"log in," that workflow has failed.

### G3. Self-finding (agent process)

When a server installs instructions, read them before its first
use, not after the first failure. The deferred-tool mechanism
makes it tempting to skip prompt-fetching because the tools "just
work." But "just work" optimizes for the easy path, not the right
path.

This is in the journal as a self-tagged `#skill-issue`. The
corresponding fix is B1 above (a skill that enforces this).

### G4. `uv lock --upgrade-package <name>` silently no-ops on package-name mismatch

The package commonly referred to as `deriva-py` is named `deriva`
in `pyproject.toml` and `uv.lock`. When I ran:

```
uv lock --upgrade-package deriva-py
```

`uv` silently produced zero output and zero changes — no warning,
no error, no "unknown package" hint. Only when I read the lock and
realized the package is `deriva` (not `deriva-py`) and re-ran with
the correct name did the upgrade actually happen.

**Severity:** low — once you know about it. Easy to miss.

**Fix shape:** this lives in `uv` upstream (astral-sh/uv), not in
any Deriva repo. Worth filing an issue there:
"`uv lock --upgrade-package <name>` should warn if `<name>` doesn't
match any package in the lock."

**Workaround for now:** document in the workspace `CLAUDE.md` (or a
deriva-ml-context note) that the package name is `deriva` in
lockfiles, even though everyone says `deriva-py` in conversation.
Save the next maintainer the lookup. Single-line addition.

### G5. Claude Code: mid-session `.mcp.json` edits don't propagate to live conversation

After hand-editing `.mcp.json` to add the deriva MCP server,
`claude mcp list` (CLI level) immediately reflected the new
server. But the **live Claude Code conversation** didn't surface
the new tools until I restarted Claude Code. `ListMcpResourcesTool({server: "deriva"})`
returned "Server 'deriva' not found" even though `claude mcp list`
showed ✓ Connected.

Conclusion: Claude Code snapshots the MCP server set from
`.mcp.json` + `~/.claude/mcp.json` at conversation start. Mid-
conversation file edits update CLI-level state but do NOT register
servers into the running conversation context. A restart is
required.

**Severity:** moderate — costs the user one restart, and the
failure mode is silent (looks like the server is wired but the
tools just aren't there).

**Fix shape:** this is a Claude Code feature gap, not a Deriva
issue. Two possible asks for the Claude Code team:

1. Auto-reload MCP server set on `.mcp.json` change (matches the
   "auto-detected" claim in their network-config docs).
2. Surface a clear "MCP config changed — restart Claude Code to
   pick up changes" notice when the file watcher detects an edit.

**Workaround for now:** a workspace-`CLAUDE.md` note: "If you
edit `.mcp.json` mid-session, restart Claude Code (Ctrl-C,
re-launch from the same directory) before the new server's tools
are usable."

---

## Section addendum: missing skill (was B4 from earlier journal note)

### B4. New skill: `clone-catalog-slice` (deferred)

> **Status: FIXED — closed 2026-05-17.** Addressed by the
> [`setup-ml-catalog` skill](https://github.com/informatics-isi-edu/deriva-ml-skills/blob/main/skills/setup-ml-catalog/SKILL.md)
> in `deriva-ml-skills`, merged as
> [PR #16](https://github.com/informatics-isi-edu/deriva-ml-skills/pull/16).
> The skill covers the same workflow this finding requested (clone
> a slice of an existing catalog into a fresh destination) plus the
> sibling "create from scratch via `create_ml_catalog` + a phased
> loader" path. Branch 2 of the new skill walks the three-step
> sequencing the finding flagged: create the destination catalog,
> install the deriva-ml schema, then call `clone_via_bag` with
> anchors that define the slice. The skill teaches anchor assembly
> as its own decision point (pick your roots — Dataset, Subject,
> Experiment, Workflow) and explains why the upstream
> `clone_via_bag` defaults produce a working ML catalog with
> bounded scope (the `terminal_tables` asymmetry follows outbound
> FKs for full provenance, blocks inbound FKs to prevent cross-
> anchor over-fetch). Naming chose `setup-ml-catalog` over
> `clone-catalog-slice` so the skill could cover both bootstrap
> paths (from-scratch and from-slice) under one user-facing entry
> point.

This was flagged in the journal but not yet captured in this
findings doc.

**Repo:** `deriva-ml-skills`.

**What:** A new tier-2 skill that walks the user through cloning
a slice of an existing catalog into a fresh test/staging catalog,
using `deriva_ml.catalog.clone.create_ml_workspace` (or
`clone_via_bag.clone_via_bag` for finer control).

**Why:** This is a common workflow for staging tests, demos, and
branched experiments — copy a subset of a production catalog into
a new one, with the deriva-ml schema bootstrapped. The mechanics
exist (`create_ml_workspace` in deriva-ml), but the workflow has
non-obvious sequencing: create a fresh ERMrest catalog first, init
the ml_schema in it, *then* call `create_ml_workspace` against it.
`create_ml_workspace` does NOT create its destination catalog —
the docstring explicitly says so but a user reading the API
shape would easily assume it does.

A skill encodes the three-step workflow so the agent doesn't have
to reconstruct it from primitives.

**Note:** this finding is **deferred** — it's a new skill effort
that's significant enough to warrant its own session. Adding it
here so it doesn't get lost between the journal and any GitHub
issues filed.

---

## Priority suggestion

If addressing in order, this is the impact-ordered cut I'd take:

1. **B1** (new skill: read-mcp-guide-first) — highest leverage,
   fixes the original miss at the layer where Claude Code's
   discovery affordances actually fire.
2. **C1c** (`deriva_ml_getting_started` as both prompt AND
   resource) — universal fix on the server side; helps even
   non-skill-aware clients.
3. **E4 / F1** (schema-mutation invalidation of path-builder
   cache) — fixes the substrate; lets the deriva-py bag-loader
   refresh become defensive rather than load-bearing.
4. **H1** (stale `deriva-globus-auth-utils` references) — five-file
   doc sweep, blocks new users from getting past §"Authenticate"
   in the template README. Cheap and high-blast-radius.
5. **C3 / C4 / F6** (auth-failure error wording + resource read
   fail-fast + `check_auth` 404) — reduces user/agent confusion
   when wiring is wrong, related to the F3 auth-opacity principle.
6. **F3** (auth state opaque to end users — the principle) — the
   parent of items #4 + #5; addressing this changes the shape of
   all error-handling and user-visible auth surface.
7. **E2 / E3** (find_features alias + feature_values list/iter)
   — small API ergonomics with high muscle-memory impact.
8. **G1** (test plan: resources as first-class) — corrects the
   plan so future test phases exercise the right surface.
9. **D1 / D2 / H2** (deriva-ml-mcp README cross-link + drop
   pre-release caveat + model-template README dev-localhost recipe)
   — eliminates the wrong-recipe-in-template-README issue. Three
   coordinated doc updates.
10. **H3** (`NODE_EXTRA_CA_CERTS` setup not documented) —
    paired with H2; cheap doc fix once the dev-localhost recipe
    lands.
11. **F5** (FK-cycle warning log spam) — two-line `deriva-py`
    fix; low-priority but quick.
12. **B4** (new `clone-catalog-slice` skill) — its own session;
    don't block the rest of the list on it.
13. Everything else (E1, E5, F2, F4, G4, G5) — quality-of-life
    improvements, lower leverage individually. G4 and G5 are
    upstream (uv, Claude Code) — no Deriva-team work needed
    beyond filing.

This is a suggestion, not a mandate. The repo owners can re-order
based on local priorities and bandwidth.

---

## Appendix 1 — Draft text: "RESOURCES VS TOOLS" addition to `deriva_ml_getting_started`

**Target file:** the `deriva_ml_getting_started` prompt body in
`deriva-ml-mcp` (search for the existing "PAGINATION CONTRACT"
section and add this immediately after it). One source of truth
for every MCP client.

Draft text to insert:

````
RESOURCES VS TOOLS
------------------
Every browseable group of entities in this plugin (datasets,
workflows, executions, features, ...) is exposed two ways:

- A **resource template** — single read, returns a bounded
  snapshot (≤1000 rows), suitable for verification, introspection,
  "show me what's there." URIs have the shape
  ``deriva://catalog/{hostname}/{catalog_id}/ml/<group>``.
- A **list tool** (``deriva_ml_list_<group>``) — paginated,
  cursor-based, suitable for filtered queries and full
  enumeration beyond the snapshot bound.

**Default: prefer the resource when you want "all of X" as a
snapshot.** It is one call, no preflight, no cursor handling.
The response includes ``truncated`` and ``next_after_rid``; when
``truncated`` is true, switch to the corresponding ``list`` tool
for full cursor pagination.

**Use the tool when:** you need a filtered subset, you need more
than 1000 rows, the snapshot is paginated and you must drill
beyond the first page, or you are mutating state (no resource is
ever a write surface).

**Pairings:**

| Snapshot resource | Paginated tool |
|---|---|
| ``deriva://catalog/.../ml/datasets`` | ``deriva_ml_list_datasets`` |
| ``deriva://catalog/.../ml/dataset/{rid}`` | ``deriva_ml_get_dataset`` |
| ``deriva://catalog/.../ml/dataset/{rid}/members`` | ``deriva_ml_list_dataset_members`` |
| ``deriva://catalog/.../ml/dataset/{rid}/spec`` | ``deriva_ml_get_dataset_spec`` |
| ``deriva://catalog/.../ml/workflows`` | ``deriva_ml_list_workflows`` |
| ``deriva://catalog/.../ml/workflow/{rid}`` | ``deriva_ml_get_workflow`` |
| ``deriva://catalog/.../ml/executions`` (if exists) | ``deriva_ml_list_executions`` |
| ``deriva://catalog/.../ml/features`` (if exists) | ``deriva_ml_list_features`` |
| ``deriva://catalog/.../schema`` | ``list_schemas`` + ``get_schema`` |
| ``deriva://catalog/.../tables`` | (no direct list tool — use the resource) |
| ``deriva://catalog/.../table/{schema}/{table}`` | ``get_table`` |

(Resource template list complete as of the version of the server
in front of you — call ``resources/templates/list`` for the
authoritative current set.)

**Anti-pattern.** Reaching for ``deriva_ml_list_<x>`` when you
just want a snapshot of all-of-X is a process smell: you pay the
preflight + page cycle for data the resource would have returned
in one read. The pagination contract is for cases where the data
genuinely doesn't fit in one snapshot — not as a default.
````

Estimated insertion size: ~50 lines including the table. Sits
naturally between the existing "(HOSTNAME, CATALOG_ID) RULE"
section and the "PAGINATION CONTRACT" section, or immediately
after "PAGINATION CONTRACT."

---

## Appendix 2 — Draft skill: `using-deriva-mcp` (deriva-ml-skills)

**Target plugin:** `deriva-ml-skills`. Place under
`skills/using-deriva-mcp/SKILL.md`. Plugin marketplace
metadata adds it as a tier-2 skill alongside the existing
twenty-three.

Draft `SKILL.md` (frontmatter + body):

````markdown
---
name: using-deriva-mcp
description: "ALWAYS use this skill on the first deriva-mcp call in any conversation, or when the user asks anything about a Deriva catalog that would be answered by querying it. Read the deriva_ml_getting_started MCP prompt before reaching for tools. Prefer MCP resources for snapshot reads ('what's in this catalog?', 'list all datasets', 'show me the workflows'); prefer MCP tools only for filtered queries, paginated drilling, or mutations. Triggers on: 'list datasets', 'show datasets', 'browse catalog', 'verify catalog', 'what's in the catalog', 'check schema', any catalog inspection request, any deriva-ml-mcp call. Do NOT trigger for shell-only catalog work (load-cifar10 CLI, deriva-ml Python API only) that bypasses the MCP surface entirely."
user-invocable: false
---

# Using the deriva MCP Server

You are about to make a call against a Deriva catalog via the
`mcp__deriva__*` or `mcp__<your-server-name>__*` tools or
resources. Two things to do first:

1. **Read the `deriva_ml_getting_started` MCP prompt** if you
   haven't already in this conversation. It is the canonical
   cold-start doc for this plugin and contains:
   - The `(hostname, catalog_id)` rule (mandatory on every call).
   - The pagination contract (preflight → page → advance).
   - The **resources-vs-tools rule** (see step 2).
   - The error envelope conventions.

   Fetch the prompt via the MCP protocol's `prompts/get` method
   (or `ReadMcpResourceTool` if your client surfaces prompts as
   resources). If you've already read it in this conversation,
   skip.

2. **Pick resource vs tool correctly.** The plugin exposes both
   for every group of entities. Default to the **resource** when
   you want a snapshot of "all of X." Default to the **tool**
   when you need a filtered subset, more than 1000 rows, paginated
   drilling beyond the first page, or any write operation.

   See `deriva_ml_getting_started` §"RESOURCES VS TOOLS" for the
   full pairing table. Common pairings:

   - "List all datasets" → resource `deriva://catalog/{h}/{c}/ml/datasets`,
     not `deriva_ml_list_datasets`.
   - "Show me workflow X" → resource
     `deriva://catalog/{h}/{c}/ml/workflow/{rid}`, not
     `deriva_ml_get_workflow`.
   - "Get the schema for catalog C" → resource
     `deriva://catalog/{h}/{c}/schema`, not `list_schemas` +
     `get_schema`.
   - "Find datasets where description contains 'training'" → tool
     `deriva_ml_list_datasets` with pagination (resource has no
     filter parameter).
   - "Create / update / delete anything" → always a tool. There
     are no write resources.

## When this skill applies vs. doesn't

**Applies** to any conversation that involves reading or mutating
the catalog via the MCP surface — even if the user didn't
explicitly say "use MCP." If the user asks "verify what's in
catalog 8," that's an MCP-surface operation. If they ask "build
a model," and you reach for catalog state on the way, that's an
MCP-surface operation.

**Doesn't apply** to:
- Shell-only invocations (`load-cifar10` CLI, the model template's
  `deriva-ml-run` CLI). Those use the deriva-ml Python API
  directly and bypass the MCP server.
- deriva-ml-skills `*-context` skills that you've already loaded
  earlier in the conversation. Those provide the conceptual frame
  but not the entry-point selection rules.

## Truncation handoff

When a resource read returns `truncated: true` in its envelope,
switch to the corresponding tool. Pass the `next_after_rid` from
the resource response as the `after_rid` argument on the tool
call. Same cursor token, no resampling needed.

## Reference

- The MCP prompt `deriva_ml_getting_started` (this skill's
  source of truth — read it).
- The MCP prompt `deriva_ml_concepts` (read first if you don't
  have the DerivaML mental model yet).
- The `using-deriva-mcp` skill (this file) is the trigger; the
  prompts above are the rules.
````

**Skill design notes:**

- `description` field deliberately broad — high-confidence match
  on any catalog-inspection intent, plus explicit anti-trigger
  language for shell-only flows. The job is to fire whenever the
  agent is about to reach for the MCP surface, including cases
  where the agent might not have realized that's what they're
  about to do.
- `user-invocable: false` — this skill is automatic discipline,
  not a user-facing command. The user shouldn't have to know it
  exists.
- Body stays short. **The skill is the trigger; the prompt is
  the rule.** Resource-vs-tool decisions live in
  `deriva_ml_getting_started` so updates to the rule (new
  resources added, pairings change) propagate via the prompt to
  every MCP client.
- The "When this skill applies vs. doesn't" section is
  load-bearing — without it the skill would fire on
  shell-only template work too, which wouldn't help.

**Companion update:** `deriva-ml-context` should mention this
skill in its "always-load this" body so any session that starts
with a deriva-ml-plugin loaded has both `deriva-ml-context` and
`using-deriva-mcp` available before the first tool call. Draft
text for that update is in Appendix 3.

---

## Appendix 3 — Draft update: `deriva-ml-context` to announce `using-deriva-mcp`

**Target plugin:** `deriva-ml-skills`. Modify the existing
`skills/deriva-ml-context/SKILL.md`. This skill loads on every
session that has the `deriva-ml` plugin enabled (per its
"always-load" semantics), so it's the right place to make
`using-deriva-mcp` discoverable at session start without
requiring the agent to also load it explicitly.

### Why this companion update is needed

`using-deriva-mcp` (Appendix 2) fires automatically when the
agent reaches for a deriva MCP call. But "automatic" means "when
Claude Code's skill dispatch matches the user's intent against
the skill's description." Two failure modes that bypass that:

1. **Agent reaches for the MCP tool without an MCP-shaped user
   request.** E.g., user asks "what models are in our catalog?"
   and the agent's first instinct is to query the catalog
   directly via deriva-py. If the skill's trigger phrases don't
   include enough catalog-y wording, the skill never fires.
2. **Skill match confidence is low.** Description-based matching
   is fuzzy. A request the human reads as obviously catalog-y
   may not score high enough against `using-deriva-mcp`'s
   description for it to load.

`deriva-ml-context` is the always-on context skill. By
referencing `using-deriva-mcp` from it, the agent learns about
the skill *at session start* — even if subsequent skill dispatch
fails to match. The agent can then load `using-deriva-mcp` itself
when it recognizes the situation, rather than relying entirely on
description matching.

### Draft addition to `deriva-ml-context/SKILL.md`

Find the existing section that describes the deriva-ml-plugin's
skill surface (typically near the end of the file — the "what's
available in this plugin" overview). Add the following block.
If the context skill has a "Skills you should know about"
section, append this entry; otherwise add the section.

````markdown
## When you reach for the MCP surface

The `deriva-ml-mcp` server exposes both **resources** (snapshot
reads of bounded-size views: all datasets, all workflows, one
dataset's members, etc.) and **tools** (paginated drilling,
filtered queries, mutations). They are not interchangeable —
the correct entry-point depends on the shape of what you're
asking for.

Before your first `mcp__deriva__*` (or equivalently-named MCP
server) call in any conversation, load the **`using-deriva-mcp`**
skill. It is the trigger-companion to the
`deriva_ml_getting_started` MCP prompt and handles two cold-start
disciplines for you:

1. Ensures `deriva_ml_getting_started` and `deriva_ml_concepts`
   prompts have been read.
2. Encodes the resource-vs-tool selection rule so you don't
   default to a `list_*` tool when a snapshot resource would
   answer the question in one read.

**Routing rule:** for any user request that would be answered by
reading catalog state ("what's in this catalog?", "list the
datasets", "show me the workflows", "verify the schema",
"check feature values"), load `using-deriva-mcp` before
reaching for the MCP tools or resources directly. Skip the skill
only when the entire interaction stays on the shell/Python side
(`deriva-ml-run`, `load-cifar10`, direct `deriva-ml` library
calls in a script) and never crosses the MCP boundary.
````

### Why this lands in `deriva-ml-context` specifically

The deriva-ml plugin has two context skills:
- `deriva-ml-context` (tier 2, deriva-ml-skills) — always-load,
  carries the DerivaML conceptual frame.
- `deriva-context` (tier 1, deriva-skills) — always-load, carries
  the generic Deriva catalog frame.

`using-deriva-mcp` is a tier-2 concern (it's about the
`deriva-ml-mcp` plugin's tool/resource surface, which doesn't
exist without the deriva-ml domain). So the announcement goes in
`deriva-ml-context`, not `deriva-context`. The tier-1 context
skill stays purely about generic catalog concepts.

If a tier-1 `using-deriva-mcp-core` skill is later added for the
non-DerivaML core (`query_guide`, `entity_guide`,
`annotation_guide`, `catalog_guide` prompts and the
`get_entities`/`list_schemas`/etc. tools), THAT skill would get
announced from `deriva-context`. Two skills, two announcement
sites, no overlap.

### Size of the change

About 30 lines added to one existing file. No new file. No
plugin-marketplace metadata changes (the context skill is
already enabled). No risk to non-MCP workflows because the
announcement is conditional ("Skip the skill only when the entire
interaction stays on the shell/Python side...").

### Verification after landing

After both this companion update and the `using-deriva-mcp` skill
(Appendix 2) land, test in a fresh Claude Code conversation:

1. User asks: "what datasets are in catalog 8 on localhost?"
2. Agent should load `using-deriva-mcp` (via its description match
   OR via the `deriva-ml-context` announcement).
3. Agent should fetch `deriva_ml_getting_started` if not already
   read.
4. Agent should use
   `ReadMcpResourceTool(server=..., uri="deriva://catalog/localhost/8/ml/datasets")`
   — NOT `deriva_ml_list_datasets`.

If step 4 still ends up at the tool, the skill body or the
context-skill announcement needs further refinement.
