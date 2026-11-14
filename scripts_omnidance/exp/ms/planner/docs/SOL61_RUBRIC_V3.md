# Planner caption rubric v3 (for the paper)

Purpose: score a caption as a **dance plan for this clip**, not as prose. Every
dimension is defined so a reader can check the claim, and so a skeptical reviewer
cannot attribute the result to fluency or length.

Judge protocol
- Input to the judge: the 5.0625 s audio + one caption. No reference, no method
  name, no other caption.
- Two captions per case in both A/B orders; `tie` allowed. The first protocol run
  showed a 69.8% slot-A bias, so a single order is not admissible.
- Length control: judges are told word count is not a criterion; we also report a
  length-matched subset (captions within ±15% of each other) as a robustness check.
- Two judges from different model families; report agreement (quadratic-weighted
  kappa). Self-preference is measured by scoring a caption against itself.

Dimensions (1-5 each, anchors literal)

D1 Executability (weight .25)
 5 every clause names a body part and a movement verb; a choreographer could
   perform it without asking a question
 4 one vague clause at most
 3 mix of concrete and vague
 2 mostly vague ("moves energetically", "flows with the beat")
 1 no performable content

D2 Temporal structure (weight .20)
 5 ordering or simultaneity is explicit throughout ("then", "as", "while", or
   clause order is unambiguous)
 4 clear for most of the caption
 3 partially clear; one list-like stretch
 2 items listed with no relation
 1 no ordering information

D3 Music grounding (weight .20)
 5 at least one correspondence to a specific audible event (accent, onset,
   tempo change, pause) is stated or clearly implied
 4 plausible general correspondence (energy, tempo feel)
 3 generic but not contradicted by the clip
 2 little relation to this clip
 1 contradicts the clip

D4 Specificity / non-template (weight .15)
 5 could not be reused for a different clip without editing
 4 mostly specific
 3 half template
 2 mostly template
 1 pure template

D5 Evidence discipline (weight .10)
 5 no claim beyond what audio can support
 4 one unsupported detail
 3 two unsupported details
 2 several
 1 mostly invented (instruments, lyrics, crowd, camera, facial expression)

D6 Density discipline (weight .10)
 5 clause rate within the reference band (0.9-1.2 clauses per 100 characters)
 4 slightly outside
 3 outside by less than 2x
 2 outside by more than 2x
 1 degenerate (one endless clause or one clause per word)

Judge output (strict, parsed):
```
D1=<n> D2=<n> D3=<n> D4=<n> D5=<n> D6=<n>
evidence_music=<quote or NONE>
unsupported=<comma list or NONE>
overall=<1-5>
```

Reported statistics
- per-dimension mean with 95% bootstrap CI over source groups
- weighted total = .25 D1 + .20 D2 + .20 D3 + .15 D4 + .10 D5 + .10 D6
- pairwise win/tie/loss with both orders, plus a length-matched subset
- judge agreement between two families, and self-consistency (same caption twice)

Why these six: D1/D2 are what a video model needs to execute a plan; D3/D5 stop
the score from rewarding invented detail; D4/D6 stop it from rewarding template
or length. D1-D3 correspond to the three TX rubric dimensions (action clarity,
temporal organisation, music fit), so results stay comparable with the TX side.
