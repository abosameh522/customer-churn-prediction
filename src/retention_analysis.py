import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rank_customers(y_true, probabilities):
    """Sort customers by predicted churn probability, highest first."""
    ranked = pd.DataFrame({
        "churn": y_true.to_numpy(),
        "probability": probabilities,
    })
    if ranked.empty or not ranked["churn"].isin([0, 1]).all():
        raise ValueError("Expected non-empty binary churn labels")
    if not np.isfinite(ranked["probability"]).all() or not ranked["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not ranked["churn"].sum():
        raise ValueError("Capture and lift require at least one churner")
    return ranked.sort_values("probability", ascending=False, kind="stable").reset_index(drop=True)


def priority_table(ranked, top_fractions=(0.10, 0.20, 0.30)):
    """Measure how many real churners sit in the highest-risk share of customers."""
    total_churners = int(ranked["churn"].sum())
    base_rate = total_churners / len(ranked)
    rows = []
    for fraction in top_fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Population shares must be in (0, 1]")
        reviewed = max(1, int(round(fraction * len(ranked))))
        top = ranked.head(reviewed)
        captured = int(top["churn"].sum())
        precision = captured / reviewed
        rows.append({
            "Population share": fraction,
            "Customers reviewed": reviewed,
            "Actual population share": reviewed / len(ranked),
            "Churners captured": captured,
            "Total churners": total_churners,
            "Capture rate": captured / total_churners,
            "Precision": precision,
            "Lift": precision / base_rate,
        })
    return pd.DataFrame(rows)


def save_cumulative_gains(ranked, table, figures_dir, model_name):
    total_churners = int(ranked["churn"].sum())
    population = np.arange(1, len(ranked) + 1) / len(ranked)
    cumulative_capture = ranked["churn"].cumsum().to_numpy() / total_churners

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(np.r_[0, population] * 100, np.r_[0, cumulative_capture] * 100, color="#1f6feb", linewidth=2.2, label=model_name)
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1.2, label="Random targeting")
    for fraction in table["Actual population share"]:
        index = max(0, int(round(fraction * len(ranked))) - 1)
        axes[0].scatter(fraction * 100, cumulative_capture[index] * 100, color="#1f6feb", zorder=5)
        axes[0].annotate(
            f"{cumulative_capture[index] * 100:.0f}%",
            (fraction * 100, cumulative_capture[index] * 100),
            textcoords="offset points", xytext=(6, -12), fontsize=9, color="#1f6feb",
        )
    axes[0].set(
        title="Cumulative gains",
        xlabel="Share of test customers prioritized (%)",
        ylabel="Share of churners captured (%)",
        xlim=(0, 100), ylim=(0, 100),
    )
    axes[0].legend(loc="lower right", fontsize=9)

    labels = [f"Top {fraction:.0%}" for fraction in table["Population share"]]
    bars = axes[1].bar(labels, table["Capture rate"] * 100, color="#1f6feb")
    axes[1].plot(labels, table["Actual population share"] * 100,
                 linestyle="--", marker="o", color="gray", label="Random targeting (expected)")
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1].set(title="Churners captured at each cut-off", ylabel="Churners captured (%)")
    axes[1].legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(figures_dir / "cumulative_gains.png", dpi=150)
    plt.close(fig)


def run_retention_analysis(pipeline, X_test, y_test, reports_dir, figures_dir, model_name):
    """Rank the test customers by predicted churn and measure capture at top cut-offs."""
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    ranked = rank_customers(y_test, probabilities)
    table = priority_table(ranked)
    save_cumulative_gains(ranked, table, figures_dir, model_name)
    table.to_csv(reports_dir / "retention_priority.csv", index=False)
    return table
