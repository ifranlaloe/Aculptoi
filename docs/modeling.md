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

## Role-specific modeling guidance

Small original Markdown knowledge cards under `src/aculptoi/modeling/knowledge/` use TOML
front matter with a stable ID, controlled topics, permitted logical roles, optional source
provenance, and three required sections:

- `Construction guidance`
- `Evaluation signals`
- `Common failure modes`

Selection is deterministic: it admits only cards permitted for the recipient role, ranks
form-trait overlap before lexical overlap, breaks ties by card ID, and keeps whole cards
within the role's card-count and 12,000-character budgets. Card content hashes and the
selected role/sections/rank are recorded with Actor and Critic prompt artifacts.

Actor construction planning and work-item requests receive construction guidance plus
common failure modes. Critic Discovery and focused Critic Issue Analysis receive evaluation
signals plus common failure modes. The Inspection Reviewer receives neither Target Brief
modeling guidance nor an action catalog: it assesses whether inspection evidence is
technically sufficient, not whether the object meets the artistic target.

The planning Actor receives a capability summary. The work-item Actor alone receives the
schema-generated action catalog, including payload shapes. Critic roles never receive that
catalog and remain read-only.

## Semantic mesh operations

The allowlisted action language retains primitive creation and object transforms, then adds:

| Action | Bounded purpose |
| --- | --- |
| `object.join` | Join two to sixteen editable mesh objects while preserving the explicit target's name. |
| `mesh.transform_region` | Translate and/or scale selected vertices in a normalized local mesh region. |
| `mesh.extrude_region` | Extrude one connected selected face region, then apply a bounded local offset and scale. |
| `mesh.smooth_region` | Apply simultaneous bounded Laplacian smoothing to selected vertices. |
| `object.shade_smooth` | Enable smooth shading for every polygon without changing topology. |

Regions use the current local-space mesh AABB at the beginning of each action. Each
coordinate is normalized per axis from `-1` to `1`, is inclusive at its bound, and must be
strictly nonempty. A normalized translation or extrusion-offset unit means half the current
local extent on that axis, so behavior does not depend on unapplied object scale. Region
scaling pivots around the selected region's centroid. Extrusion uses selected face centers
and rejects disconnected face sets.

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

## Development smoke target

A simple stylized fish is a development and manual smoke target for coherent organic forms:
a readable continuous body, tapering, bilateral features, and appendage transitions. It is
not a subject-specific runtime recipe, a special operating mode, or a tested-example claim.
The same transferable actions and guidance apply to compatible form characteristics rather
than to named subjects.
