# Bounded semantic modeling

Aculptoi can enrich primitive blockouts through a small typed modeling surface while
preserving the same plan-first, checkpointed action boundary used for repeated
hard-surface geometry. It does not provide arbitrary Blender editing, raw `bpy`, shell
access, raw mesh-element identifiers, general sculpt brushes, materials, UVs, rigging,
animation, or a guarantee of production topology.

## Target Brief

For a new run, the Actor makes one structured Target Brief request before construction
planning. The original Goal remains authoritative. The brief records a concise subject,
visual priorities, explicit constraints, explicit non-goals, and controlled transferable
form traits such as `organic`, `hard_surface`, `tapered_form`, `appendages`, and
`repeated_geometry`.

The run persists the request as `target-brief-prompt.json` and the validated result as
`target-brief.json`; `run-state.json` records the fixed artifact path and content digest.
The brief is reused across all later planning and Critic requests in that run. Resume never
silently regenerates a missing pending brief. Older runs without this artifact receive the
deterministic fallback derived directly from their saved Goal.

Construction items may additionally record an optional, bounded set of transferable
`form_traits` for the modeling problem that item solves. These use the same controlled
vocabulary as the Target Brief; they describe properties such as appendages or bilateral
symmetry, never a subject name or implementation command. Existing plans without the field
remain valid with an empty list.

## Role-specific modeling guidance

Small original Markdown knowledge cards under `src/aculptoi/modeling/knowledge/` use TOML
front matter with a stable ID, controlled topics, permitted logical roles, optional source
provenance, and three required sections:

- `Construction guidance`
- `Evaluation signals`
- `Common failure modes`

Selection is deterministic: it admits only cards permitted for the recipient role and
keeps whole cards within the role's card-count and 12,000-character budgets. Work-item
Actor selection ranks overlap with that item's form traits first, then Target Brief traits,
then lexical relevance, and finally card ID. An item without traits uses the established
Target Brief-first behavior. Card content hashes and the selected role/sections/rank are
recorded with Actor and Critic prompt artifacts.

Actor construction planning and work-item requests receive construction guidance plus
common failure modes. Critic Discovery and focused Critic Issue Analysis receive evaluation
signals plus common failure modes. The Inspection Reviewer receives neither Target Brief
modeling guidance nor an action catalog: it assesses whether inspection evidence is
technically sufficient, not whether the object meets the artistic target.

The planning Actor receives a capability summary. The work-item Actor alone receives the
schema-generated action catalog, including payload shapes, and schema-owned action semantics.
Those semantics make normalized region coordinates, selection rules, pivots, join/remesh
topology effects, and other execution meaning part of the actual Actor request rather than
human documentation only. Critic roles never receive the catalog or semantics and remain
read-only.

The full Actor catalog also carries compact, exact numeric constraints beside each relevant
payload field—for example exclusive voxel-size minima, component-scale bounds, normalized-region
bounds, region movement and scaling limits, smoothing factor/iterations, and subdivision cuts.
These values share the typed action contract's bounds; examples are illustrative rather than the
only legal payloads.

The Pydantic action schema is the authoritative host-side payload contract. The compact catalog
derives required and optional fields, enums, and straightforward scalar numeric bounds from that
schema rather than maintaining a second copy. It retains only semantic annotations that schema
metadata cannot express compactly, such as normalized-region relationships and vector-component
bounds. `object.create.primitive` remains optional only for historic artifact compatibility; one
narrow Actor-facing override still requires new model proposals to name their primitive. The
Blender worker independently validates every action at its separate trust boundary.

## Semantic mesh operations

The allowlisted action language retains primitive creation and object transforms, then adds:

| Action | Bounded purpose |
| --- | --- |
| `object.join` | Join two to sixteen editable mesh objects while preserving the explicit target's name. |
| `mesh.transform_region` | Translate and/or scale selected vertices in a normalized local mesh region. |
| `mesh.extrude_region` | Extrude one connected selected face region, then apply a bounded local offset and scale. |
| `mesh.subdivide` | Add bounded topology density while approximately preserving the existing surface shape. |
| `mesh.smooth_region` | Apply simultaneous bounded Laplacian smoothing to selected vertices. |
| `object.shade_smooth` | Enable smooth shading for every polygon without changing topology. |

Regions use the current local-space mesh AABB at the beginning of each action. Each
coordinate is normalized per axis from `-1` to `1`, is inclusive at its bound, and must be
strictly nonempty. A normalized translation or extrusion-offset unit means half the current
local extent on that axis, so behavior does not depend on unapplied object scale. Region
scaling pivots around the selected region's centroid. Extrusion uses selected face centers
and rejects disconnected face sets.

`mesh.subdivide` applies direct mesh subdivision with `cuts` from one through three. It adds
editable topology when the current surface cannot represent a required deformation, but does not
intentionally smooth, voxelize, or change object transforms. The worker checks conservative
pre-operation estimates and actual post-operation topology against its mesh complexity limits.

`mesh.smooth_region` is geometry smoothing, not smooth shading: selected vertices move toward
their adjacent-vertex averages. It does not add topology and can shrink or flatten a form,
especially with sparse topology, high factor, or many iterations. `object.shade_smooth` changes
polygon shading only.

`sculpt.voxel_remesh` is destructive topology reconstruction. It is useful for intentionally
rebuilding a surface or fusing overlapping masses into one continuous volume, but it is not
generic subdivision or the ordinary way to obtain more editable mesh density. Its result depends
strongly on `voxel_size` and may soften form or erase thin features. The worker fingerprints the
mesh before and after it runs; a complete no-change result becomes a recoverable
`no_topology_change` failure rather than a silent success.

The model names objects and regions, never vertex, edge, or face IDs. The worker independently
checks allowed fields, names, editability, finite and bounded numbers, object availability,
selection validity, action-batch limits, and resulting mesh complexity. A failed batch reloads
the active run's saved canonical scene so partial mutations do not remain active.

## Execution feedback and work-item continuation

Aculptoi owns execution continuity; the Actor does not retain a chat session or receive an
ever-growing transcript. Each work-item request remains a fresh compilation of the immutable
plan and completion criteria, current Blender scene inspection, relevant modeling context, and
at most one immediate `recent_execution` outcome.

Expected bounded failures use `validation_error` or `execution_error` with a compact action
index, command, stable code, public message, number of preceding actions, and rollback status.
The harness reloads the prior canonical scene, obtains a fresh inspection, persists the failed
attempt beside its Actor response, and asks for the same work item again. The retry receives the
same completion criteria and knows that no mutation from the rolled-back batch remains applied.

All proposed actions in an attempted batch consume the normal action budget, even when rollback
removes their scene changes; retries also consume normal Actor-request and time budgets. A
`worker_error` is an internal Aculptoi or Blender-worker fault, not modeling feedback. It stops
mutation after restoration is attempted and is surfaced for diagnosis rather than sent to the
Actor for adaptation.

Actor proposals that fail typed work-item validation are a distinct, recoverable case: no Blender
mutation or action budget is consumed. Aculptoi saves the raw response through its normal local
diagnostic path, records a compact `recent_proposal_validation` payload, and asks statelessly for
the same work item again. Request and wall-clock budgets still apply. By default, three
consecutive invalid proposals stop the item with a clear error rather than allowing an unbounded
schema-repair loop.

The first response for a work item establishes its immutable non-empty completion-criteria list.
Every work-item request contains a small deterministic `response_requirements` object: before
criteria exist it requires an `array[string]` with the schema-owned one-to-ten item limit; after
they exist it says that criteria must be omitted. This is a turn contract, not a duplicate JSON
schema. A missing or malformed first response remains a non-mutating, bounded proposal-validation
retry with compact feedback.

## Modeling Steps and Actor viewport observation

A **Modeling Step** is the Actor's mutation unit: one bounded, transactional,
human-meaningful change toward the active construction item's immutable completion
criteria. It carries a concise semantic `intent` plus the typed actions needed only for
that change. The worker still accepts no more than 25 actions, but that is a safety
ceiling—not the intended size of an Actor turn. A typical step uses one to five actions.

The work-item Actor has three typed responses: `modeling_step`,
`observation_request`, and `complete`. A Modeling Step requires at least one action and
one intent. An observation request selects one bounded semantic view (`front`, `top`,
or a three-quarter view, with predictable framing/projection) and cannot contain
mutating actions. Completion cannot contain actions and, in UI mode, follows an
observation of the current successful scene.

Before choosing a tool, the Actor evaluates topology suitability relative to the intended form
and deformation. Bounding dimensions alone do not prove that a mesh can support gradual curves,
local silhouette changes, taper, or several independently controllable regions; counts and their
distribution are considered alongside current visual evidence. Construction plans likewise state
semantic target outcomes. They do not append generic smoothing, cleanup, polish, or remesh stages
without a target- or critique-justified visual objective.

In Observer Mode, the worker deterministically reserves the largest available `VIEW_3D`
area while a run is active. Before each capture it resets solid shading, neutral
background, overlays, framing, and semantic orientation; user navigation therefore does
not become Actor state. The worker captures only that editor region, never the desktop or
other Blender panels. It returns one bounded PNG in the local HTTP response, which the
provider sends as the current multimodal Actor input. The prompt artifact redacts this
data URL and ordinary images are not written to the run directory.

The normal cadence is: initial item observation, one Modeling Step, successful canonical
save, post-step observation, then a fresh Actor request. The Actor may ask for a bounded
number of additional semantic views per item (default 12); those turns consume Actor and
wall-clock budgets but no action budget. Recoverable failed steps restore canonical state
and receive a fresh rollback observation before retry. If UI observation is unavailable,
including headless mode, the Actor receives explicit capability metadata and continues
from structured scene state. This working sensor is independent from the persistent,
deterministic Inspection atlas used by the Inspection Reviewer and Critic.

## Development smoke target

A simple stylized fish is a development and manual smoke target for coherent organic forms:
a readable continuous body, tapering, bilateral features, and appendage transitions. It is
not a subject-specific runtime recipe, a special operating mode, or a tested-example claim.
The same transferable actions and guidance apply to compatible form characteristics rather
than to named subjects.
