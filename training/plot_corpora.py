"""Render standalone research figures from completed aggregate corpus receipts."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from pairs import hash_file

NAMES = {"wi": "Write & Improve", "fce": "FCE", "c4": "C4_200M", "github-typo": "GitHub Typo",
         "wiked": "WikEd", "combined": "Available-source mix"}


def render(receipt_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    receipt = json.loads(receipt_path.read_text())
    arms = receipt["arms"]
    order = [name for name in ("wi", "fce", "c4", "github-typo", "wiked", "combined") if name in arms]
    if len(order) != len(arms) or not order:
        raise ValueError("Unknown or empty training arms")
    for arm in arms.values():
        if arm["test_guarded"]["sentences"] != 747 or arm["test_diagnostic"]["sentences"] != 747:
            raise ValueError("Chart requires complete regression populations")
    qualified = sum(arm["quality_gate_passed"] for arm in arms.values())
    source_count = sum(value["status"] == "trained" for value in receipt["sources"].values())
    devices = "/".join(sorted({arm["device"].upper() for arm in arms.values()}))
    ink, muted, paper, blue, orange = "#303632", "#606660", "#f8f6f1", "#466887", "#b87942"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": ink,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink,
                         "axes.edgecolor": "#c6c6bf", "svg.fonttype": "none",
                         "svg.hashsalt": "gamma-eh-corpus-training"})
    fig = plt.figure(figsize=(15, 10), facecolor=paper)
    grid = fig.add_gridspec(1, 2, left=.16, right=.94, top=.71, bottom=.29, wspace=.72)
    fig.text(.06, .95, "GAMMA EH  /  NATURAL-CORPUS TRAINING", color=blue, fontsize=12, weight="bold")
    fig.text(.06, .89, f"Training completed.\nQuality gates passed: {qualified}/{len(arms)}.", fontsize=29,
             weight="bold", va="top", linespacing=1.2)
    fig.text(.06, .765, f"{source_count} bounded source pilots + available-source mix · {receipt['schedule']['epochs']} epochs · {devices} recorded per arm",
             fontsize=13, color=muted)
    ax = fig.add_subplot(grid[0, 0], facecolor=paper)
    rows = [arms[name]["trained_rows"] for name in order]
    bars = ax.barh(range(len(order)), rows, height=.56, color=blue, zorder=3)
    ax.set_yticks(range(len(order)), [NAMES[name] for name in order])
    ax.invert_yaxis()
    maximum = max(rows)
    ax.set_xlim(0, maximum * 1.24)
    ax.set_xlabel("Training pairs within the edit vocabulary and context", fontsize=10, labelpad=12)
    ax.set_title("Pairs actually used for training", fontsize=15, loc="left", weight="bold", pad=22)
    ax.ticklabel_format(style="plain", axis="x")
    for bar, value in zip(bars, rows):
        ax.text(value + maximum * .025, bar.get_y() + bar.get_height()/2, f"{value:,}", va="center", weight="bold")

    ax2 = fig.add_subplot(grid[0, 1], facecolor=paper)
    diagnostic = [100 * arms[name]["test_diagnostic"]["edit_f0_5"] for name in order]
    guarded = [100 * arms[name]["test_guarded"]["edit_f0_5"] for name in order]
    bars = ax2.barh(range(len(order)), diagnostic, height=.56, color=orange, hatch="//", edgecolor="#925c2b", linewidth=.5, zorder=3)
    ax2.scatter(guarded, range(len(order)), color=blue, marker="o", s=38, zorder=4, clip_on=False)
    ax2.set_yticks(range(len(order)), [NAMES[name] for name in order])
    ax2.invert_yaxis()
    score_max = 110
    ax2.set_xlim(0, score_max)
    ax2.set_xticks((0, 25, 50, 75, 100))
    ax2.set_xlabel("ERRANT F0.5 (0–100); higher is better", fontsize=11, labelpad=12)
    ax2.set_title("JFLEG test: 747 complete sentences", fontsize=15, loc="left", weight="bold", pad=22)
    for bar, value in zip(bars, diagnostic):
        ax2.text(value + score_max * .025, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center", weight="bold")
    ax2.legend(handles=[Patch(facecolor=orange, hatch="//", edgecolor="#925c2b", label="Diagnostic: quality gates unenforced"),
                        Line2D([0], [0], marker="o", color=blue, linestyle="", label="Policy with quality gates")],
               loc="upper left", bbox_to_anchor=(-.04, -.17), frameon=False, fontsize=10)
    for axis in (ax, ax2):
        axis.grid(axis="x", color="#e4e2da", zorder=0)
        axis.tick_params(axis="y", length=0, pad=10)
        for name in ("top", "right", "left"):
            axis.spines[name].set_visible(False)
    pending = [name for name, value in receipt["sources"].items() if value["status"] == "access-pending"]
    fig.text(.06, .185, "Access pending: " + ", ".join({"nucle": "NUCLE", "lang8": "Lang-8", "clang8": "cLang-8"}.get(name, name) for name in pending),
             fontsize=14, weight="bold")
    fig.text(.06, .14, "Dev gates: ≥95% edit precision, ≤2% changes to reference-accepted text, ≥25 proposed edits; both ONNX formats.", fontsize=11, color=muted)
    fig.text(.06, .106, "Complete JFLEG regression diagnostic; best reference per sentence. Not official GLEU or browser reliability proof.", fontsize=11, color=muted)
    fig.text(.06, .072, "Unequal samples, vocabularies and devices; one seed. No full-corpus ranking. Shipped model unchanged.", fontsize=11, color=muted)
    fig.text(.06, .035, "Source: training-corpora-results.json · benchmark sources/references excluded from training · weights remain local", fontsize=10, color=muted)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for extension in ("png", "svg"):
        path = output / ("training-corpora-overview." + extension)
        fig.savefig(path, dpi=150, facecolor=paper, metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            namespace = "http://www.w3.org/2000/svg"
            ET.register_namespace("", namespace)
            tree = ET.parse(path)
            root = tree.getroot()
            root.set("role", "img")
            root.set("aria-labelledby", "corpora-title corpora-description")
            title = ET.Element(f"{{{namespace}}}title", {"id": "corpora-title"})
            title.text = f"Gamma EH training: {source_count} sources, {qualified} of {len(arms)} candidates qualify"
            description = ET.Element(f"{{{namespace}}}desc", {"id": "corpora-description"})
            description.text = "; ".join(f"{NAMES[name]}: {arms[name]['trained_rows']} trained pairs, diagnostic test F0.5 {diagnostic[index]:.2f}, guarded F0.5 {guarded[index]:.2f}" for index, name in enumerate(order))
            description.text += ". Access pending: " + ", ".join(pending) + ". Not official GLEU or browser quality. Unequal pilot sizes. Shipped model unchanged."
            root.insert(0, description)
            root.insert(0, title)
            tree.write(path, encoding="utf-8", xml_declaration=True)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    plt.close(fig)
    manifest = {"schema": 1, "source_receipt": receipt_path.name, "source_sha256": hash_file(receipt_path),
                "renderer_sha256": hash_file(Path(__file__)),
                "versions": {name: version(name) for name in ("matplotlib", "numpy")}, "files": files}
    (output / "training-corpora-figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("docs/training-corpora-results.json"))
    p.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = p.parse_args()
    print(json.dumps(render(args.input, args.output_dir), indent=2))
