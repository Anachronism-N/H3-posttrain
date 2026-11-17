"""Rebuild comparable D1-D3 ratings from frozen public per-case records."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "results/d123_refine_v1"
BASE = ROOT / "results/d123_baselines"
DIMS = ("D1", "D2", "D3")
PROTOCOL = "r4"
MODELS = [
    ("qwen3-omni-instruct", "Qwen3-Omni-Instruct"),
    ("moss-audio", "MOSS-Audio"), ("moss-music", "MOSS-Music"),
    ("music-flamingo", "Music-Flamingo"), ("audio-flamingo3", "Audio-Flamingo3"),
    ("audio-flamingo-next", "Audio-Flamingo-Next"), ("qwen25-omni", "Qwen2.5-Omni"),
    ("Ours", "Ours"),
]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid(record):
    return (record["status"] == "valid_structure" and
            type(record["score"]) is int and 1 <= record["score"] <= 5)


def main():
    om = json.loads((OURS/"manifest.json").read_text(encoding="utf-8"))
    bm = json.loads((BASE/"manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((BASE/"provenance.json").read_text(encoding="utf-8"))
    for name, expected in om["artifacts_sha256"].items():
        assert sha(OURS/name) == expected
    for name, expected in bm["files"].items():
        assert sha(BASE/name) == expected["sha256"]
    outputs, judgments = rows(OURS/"outputs.jsonl"), rows(OURS/"judgments.jsonl")
    ours_ids = {r["case_id"] for r in outputs}
    assert len(ours_ids) == len(outputs) == len(judgments) == 2000
    texts = {r["case_id"]: r for r in outputs}
    for r in judgments:
        output = texts[r["case_id"]]
        assert r["text_sha256"] == output["text_sha256"]
        assert hashlib.sha256(output["text"].encode("utf-8")).hexdigest() == r["text_sha256"]
    base_outputs = rows(BASE/"outputs_2000.jsonl")
    for method,_ in MODELS[:-1]:
        ids = [r["case_id"] for r in base_outputs if r["method"] == method]
        assert len(ids) == len(set(ids)) == 2000 and set(ids) == ours_ids
    assert set(bm["methods"]) == {m for m,_ in MODELS[:-1]}
    scores = {method: {dim: {} for dim in DIMS} for method,_ in MODELS}
    splits = {}
    for r in rows(BASE/"judgments_r4_holdout.jsonl"):
        method,dim,case = r["method"],r["dimension"],r["case_id"]
        assert case not in scores[method][dim]
        scores[method][dim][case] = r
        assert case not in splits or splits[case] == r["split"]
        splits[case] = r["split"]
    shared = set(next(iter(scores.values()))["D1"])
    assert len(shared) == 1900 and shared <= ours_ids
    for method,_ in MODELS[:-1]:
        for dim in DIMS:
            assert set(scores[method][dim]) == shared
    for r in judgments:
        if r["case_id"] in shared:
            for dim in DIMS:
                scores["Ours"][dim][r["case_id"]] = r["scores"][PROTOCOL][dim]
    common = {dim: set.intersection(*[
        {case for case,record in scores[method][dim].items() if valid(record)}
        for method,_ in MODELS]) for dim in DIMS}
    means = {method: {dim: statistics.mean(scores[method][dim][case]["score"]
                                          for case in common[dim])
                      for dim in DIMS} for method,_ in MODELS}
    for dim in DIMS:
        assert len(common[dim]) == audit["common_per_dimension"][dim]
        for method,_ in MODELS:
            assert math.isclose(means[method][dim],
                audit["common_per_dimension_means"][method][dim], abs_tol=1e-12)
    maxima = {dim: max(row[dim] for row in means.values()) for dim in DIMS}
    tex = r"""% Generated from matched r4 records by results/build_d123_refinement_table.py.
\begin{table*}[t]
\centering
\caption{Automatic caption assessment on a 1--5 scale (higher is better). Bold marks the highest score per dimension. M1--M3 remain unreported. Evaluation and inference details appear in Appendix~\ref{sec:experimental-details}.}
\label{tab:refinement-d123}
\begingroup\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
Method & Q1 & Q2 & Q3 & M1 & M2 & M3 \\
\midrule
"""
    for method,name in MODELS:
        if method == "Ours":
            tex += r"\midrule" + "\n"
            name = r"\textbf{Ours}"
        cells = [name]
        for dim in DIMS:
            cell = f"{means[method][dim]:.2f}"
            if math.isclose(means[method][dim],maxima[dim],abs_tol=1e-12):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        tex += " & ".join(cells + ["--","--","--"]) + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular*}
\endgroup
\end{table*}
"""
    (ROOT/"tables/refinement_d123.tex").write_text(tex,encoding="utf-8")
    receipt = dict(schema="paper.automatic_caption_ratings.v2",
        source_commits=dict(ours="dc6dfd9894fb702acb857524f930481efec829df",
                            baselines=audit["source_commit"]),
        primary_rating_version=PROTOCOL,
        protocol_selection="Only recorded scoring family published for all methods; no cross-version maxima.",
        baseline_rubric_version=bm["r4"]["rubric_version"],
        ours_rubric_sha256=om["score_provenance"][PROTOCOL]["rubric_sha256"],
        exact_prompt_hash_parity_verified=False,
        aggregation="All-method common valid intersection per dimension on the same registered cases.",
        display_method="Ours",per_cell_counts_displayed=False,means=means,maxima=maxima,
        registered_cases=len(shared),common_valid_counts={d:len(v) for d,v in common.items()},
        common_valid_ids_sha256={d:hashlib.sha256(("\n".join(sorted(v))+"\n").encode()).hexdigest()
                                for d,v in common.items()},
        missing_scores_imputed=False,baseline_comparison_available=True,
        source_dimension_mapping_verified=False,unreported_dimensions=["D4","D5","D6"])
    (OURS/"table.source.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(protocol=PROTOCOL,means=means,
                         common_valid_counts=receipt["common_valid_counts"]),indent=2))


if __name__ == "__main__":
    main()
