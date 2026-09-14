<!-- aculptoi-prompt-version: v1 -->

# Aculptoi Actor: Target Brief

Derive one compact structured interpretation aid for the supplied raw user goal. The raw
goal remains authoritative; preserve its explicit intent and do not invent unsupported
requirements. Identify transferable geometric and form characteristics, not subject
classes or construction recipes.

Return JSON only with exactly these fields:

```json
{
  "subject": "short description of the requested result",
  "visual_priorities": ["most important visible outcome"],
  "constraints": ["explicit requirement that limits the result"],
  "non_goals": ["explicitly excluded detail"],
  "form_traits": ["organic"]
}
```

`form_traits` may contain only: `organic`, `hard_surface`, `continuous_form`,
`bilateral_symmetry`, `radial_symmetry`, `tapered_form`, `elongated_form`,
`serpentine_form`, `appendages`, `thin_features`, `repeated_geometry`, or
`layered_forms`.

Keep every string concise. Include only explicit constraints and non-goals from the
goal; use empty arrays when none are supplied. Do not output Blender actions, code,
shell commands, Python, bpy, filesystem paths, network operations, Markdown fences, or
extra fields.
