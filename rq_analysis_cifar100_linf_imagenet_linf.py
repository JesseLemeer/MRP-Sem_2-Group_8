from pathlib import Path
import pandas as pd

BASE = Path("analysis_cifar100_linf_imagenet_linf")
BENCHMARKS = ["cifar100_linf", "imagenet_linf"]
OUT = BASE / "research_question_analysis"
OUT.mkdir(exist_ok=True)

def load_csv(benchmark, filename):
    path = BASE / benchmark / filename
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    df.insert(0, "benchmark", benchmark)
    return df

def save(df, filename):
    path = OUT / filename
    df.to_csv(path, index=False)
    print(f"Saved {path} -> {df.shape}")

# ============================================================
# Load data
# ============================================================
asr = pd.concat([load_csv(b, "asr_results.csv") for b in BENCHMARKS], ignore_index=True)
sat = pd.concat([load_csv(b, "sat_results.csv") for b in BENCHMARKS], ignore_index=True)
transfer = pd.concat([load_csv(b, "transferability_long.csv") for b in BENCHMARKS], ignore_index=True)
cka_group_corr = pd.concat([load_csv(b, "cka_group_transfer_correlations.csv") for b in BENCHMARKS], ignore_index=True)
cka_layer_corr = pd.concat([load_csv(b, "cka_transfer_layer_correlations.csv") for b in BENCHMARKS], ignore_index=True)
layer_group = pd.concat([load_csv(b, "cka_layer_group_summary.csv") for b in BENCHMARKS], ignore_index=True)
cross_shift = pd.concat([load_csv(b, "cross_layer_cka_shift_summary.csv") for b in BENCHMARKS], ignore_index=True)
pairwise = pd.concat([load_csv(b, "pairwise_summary.csv") for b in BENCHMARKS], ignore_index=True)

# Remove self-transfer for research interpretation
sat_nonself = sat[sat["source_model"] != sat["target_model"]].copy()
transfer_nonself = transfer[transfer["source_model"] != transfer["target_model"]].copy()

# ============================================================
# RQ1: Does adversarial transferability correlate with CKA?
# ============================================================
rq1 = cka_group_corr.copy()
rq1["interpretation"] = rq1["rho"].apply(
    lambda r: "positive relation" if r > 0.3 else
              "negative relation" if r < -0.3 else
              "weak or no relation"
)
rq1["statistical_note"] = rq1["p_value"].apply(
    lambda p: "statistically significant at p<0.05" if p < 0.05 else
              "not statistically significant"
)
save(rq1, "rq1_cka_transfer_cifar100_linf_imagenet_linf.csv")

# ============================================================
# RQ1a: Which layers are most related to transfer?
# ============================================================
rq1a = cka_layer_corr.copy()
rq1a["abs_rho"] = rq1a["rho"].abs()
rq1a = rq1a.sort_values(["benchmark", "abs_rho"], ascending=[True, False])
rq1a["layer_importance_note"] = rq1a.apply(
    lambda row: "strong local relation" if abs(row["rho"]) >= 0.7 else
                "moderate local relation" if abs(row["rho"]) >= 0.4 else
                "weak local relation",
    axis=1
)
save(rq1a, "rq1a_layer_importance_cifar100_linf_imagenet_linf.csv")

# ============================================================
# RQ1b: Do similar weaknesses align at different depths?
# Cross-layer best-match shift analysis
# ============================================================
rq1b = cross_shift[
    [
        "benchmark",
        "source_model",
        "target_model",
        "mean_diagonal_cka",
        "mean_off_diagonal_cka",
        "mean_best_match_cka",
        "mean_abs_best_layer_shift",
        "max_abs_best_layer_shift",
        "transfer_asr",
        "source_training_type",
        "target_training_type",
    ]
].copy()

rq1b["alignment_type"] = rq1b["mean_abs_best_layer_shift"].apply(
    lambda x: "mostly same-depth alignment" if x <= 0.25 else
              "moderate shifted-depth alignment" if x <= 0.75 else
              "clear shifted-depth alignment"
)
save(rq1b, "rq1b_cross_layer_alignment_cifar100_linf_imagenet_linf.csv")

# ============================================================
# RQ1c: Where do representations diverge or reconverge?
# Use early/middle/late CKA pattern
# ============================================================
rq1c = layer_group[
    [
        "benchmark",
        "source_model",
        "target_model",
        "early_cka",
        "middle_cka",
        "late_cka",
        "mean_layerwise_cka",
        "transfer_asr",
        "source_training_type",
        "target_training_type",
    ]
].copy()

rq1c["early_to_middle_change"] = rq1c["middle_cka"] - rq1c["early_cka"]
rq1c["middle_to_late_change"] = rq1c["late_cka"] - rq1c["middle_cka"]
rq1c["pattern"] = rq1c.apply(
    lambda r: "diverge then reconverge"
    if r["early_to_middle_change"] < 0 and r["middle_to_late_change"] > 0
    else "progressive divergence"
    if r["early_to_middle_change"] < 0 and r["middle_to_late_change"] <= 0
    else "progressive convergence"
    if r["early_to_middle_change"] >= 0 and r["middle_to_late_change"] > 0
    else "mixed or stable pattern",
    axis=1
)
save(rq1c, "rq1c_divergence_reconvergence_cifar100_linf_imagenet_linf.csv")

# ============================================================
# RQ2: How do training choices affect representation geometry?
# Summarize mean CKA by training-type pairs
# ============================================================
rq2 = (
    layer_group.groupby(
        ["benchmark", "source_training_type", "target_training_type"], dropna=False
    )
    .agg(
        n_pairs=("mean_layerwise_cka", "count"),
        mean_early_cka=("early_cka", "mean"),
        mean_middle_cka=("middle_cka", "mean"),
        mean_late_cka=("late_cka", "mean"),
        mean_layerwise_cka=("mean_layerwise_cka", "mean"),
        mean_transfer_asr=("transfer_asr", "mean"),
    )
    .reset_index()
)
save(rq2, "rq2_training_geometry_cifar100_linf_imagenet_linf.csv")

# ============================================================
# RQ3: How do training choices affect transferability?
# Summarize ASR and SAT by benchmark/model/training type
# ============================================================
model_info_cols = ["benchmark", "model", "clean_accuracy", "robust_accuracy", "asr"]
rq3_asr = asr[model_info_cols].copy()

rq3_sat = (
    sat_nonself.groupby(["benchmark", "source_model"])
    .agg(
        mean_outgoing_sat=("sat", "mean"),
        min_outgoing_sat=("sat", "min"),
        max_outgoing_sat=("sat", "max"),
    )
    .reset_index()
    .rename(columns={"source_model": "model"})
)

rq3 = rq3_asr.merge(rq3_sat, on=["benchmark", "model"], how="left")
rq3["robustness_transfer_note"] = rq3.apply(
    lambda r: "high ASR and high outgoing transfer" if r["asr"] >= 0.6 and r["mean_outgoing_sat"] >= 0.6 else
              "high ASR but limited outgoing transfer" if r["asr"] >= 0.6 else
              "lower ASR but transfer still possible",
    axis=1
)
save(rq3, "rq3_training_transferability_cifar100_linf_imagenet_linf.csv")

# ============================================================
# Benchmark-level summary
# ============================================================
benchmark_summary = (
    asr.groupby("benchmark")
    .agg(
        n_models=("model", "count"),
        mean_clean_accuracy=("clean_accuracy", "mean"),
        mean_robust_accuracy=("robust_accuracy", "mean"),
        mean_asr=("asr", "mean"),
        min_asr=("asr", "min"),
        max_asr=("asr", "max"),
    )
    .reset_index()
)

sat_summary = (
    sat_nonself.groupby("benchmark")
    .agg(
        mean_cross_model_sat=("sat", "mean"),
        min_cross_model_sat=("sat", "min"),
        max_cross_model_sat=("sat", "max"),
    )
    .reset_index()
)

benchmark_summary = benchmark_summary.merge(sat_summary, on="benchmark", how="left")
save(benchmark_summary, "benchmark_summary_cifar100_linf_imagenet_linf.csv")

# ============================================================
# Report-ready text findings
# ============================================================
lines = []
lines.append("Report-ready findings for CIFAR-100 Linf and ImageNet Linf")
lines.append("=" * 70)
lines.append("")

for _, row in benchmark_summary.iterrows():
    lines.append(
        f"{row['benchmark']}: mean ASR = {row['mean_asr']:.3f}, "
        f"mean robust accuracy = {row['mean_robust_accuracy']:.3f}, "
        f"mean cross-model SAT = {row['mean_cross_model_sat']:.3f}."
    )

lines.append("")
lines.append("RQ1: CKA-transfer relationship")
for _, row in rq1.iterrows():
    lines.append(
        f"- {row['benchmark']} | {row['cka_feature']}: rho={row['rho']:.3f}, "
        f"p={row['p_value']:.3f}. Interpretation: {row['interpretation']}; "
        f"{row['statistical_note']}."
    )

lines.append("")
lines.append("RQ1a: Layer importance")
for benchmark in BENCHMARKS:
    top = rq1a[rq1a["benchmark"] == benchmark].head(2)
    for _, row in top.iterrows():
        lines.append(
            f"- {benchmark}: layer {int(row['layer_idx'])} has rho={row['rho']:.3f} "
            f"(p={row['p_value']:.3f}), classified as {row['layer_importance_note']}."
        )

lines.append("")
lines.append("Overall interpretation:")
lines.append(
    "The two benchmarks show that adversarial vulnerability and transferability are not explained by a simple global CKA relationship. "
    "CIFAR-100 Linf and ImageNet Linf both have high attack success rates, but the CKA-transfer correlations are weak or inconsistent. "
    "This suggests that transferability depends not only on representation similarity, but also on attack direction, architecture, decision-boundary alignment, and the specific robustness training strategy."
)

report_path = OUT / "report_ready_findings_cifar100_linf_imagenet_linf.txt"
report_path.write_text("\n".join(lines))
print(f"Saved {report_path}")

print("\nDone. Research-question analysis completed.")
