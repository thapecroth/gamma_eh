"""Export publication figures directly from the aggregate benchmark receipt."""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo


def render(receipt_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    receipt = json.loads(receipt_path.read_bytes())
    jfleg, cweb = (receipt["datasets"][name] for name in ("jfleg", "cweb"))
    if jfleg["sentences"] != 747 or cweb["sentences"] != 6845:
        raise ValueError("Figures require the complete published benchmark populations")
    if jfleg["model"] != cweb["model"] or jfleg["engine_bundle_sha256"] != cweb["engine_bundle_sha256"]:
        raise ValueError("Figures require identical evaluated models and engines")
    if jfleg["scores"]["rules"] != jfleg["scores"]["combined"] or cweb["scores"]["rules"] != cweb["scores"]["combined"]:
        raise ValueError("Refresh the interpretation when rules and combined scores differ")

    ink, muted, background = "#36312e", "#625e58", "#f8f6f1"
    purple, olive, orange, grey = "#76609b", "#778b6b", "#c58251", "#c7c2ba"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": ink,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink,
                         "axes.edgecolor": grey, "svg.fonttype": "none", "svg.hashsalt": "gamma-eh-benchmark"})
    fig = plt.figure(figsize=(14, 11), facecolor=background)
    grid = fig.add_gridspec(2, 2, left=.14, right=.955, bottom=.19, top=.70, hspace=.74, wspace=.5,
                           height_ratios=(1.25, .8))
    fig.text(.065, .955, "GAMMA EH  /  PUBLIC BENCHMARK", fontsize=12, color=purple, weight="bold")
    fig.text(.065, .91, "The neural model adds little.\nThe checker still overcorrects.", fontsize=28, weight="bold", linespacing=1.25, va="top")
    fig.text(.065, .80, "7,592 natural test sentences · actual browser · fixed shipped policy", fontsize=13, color=muted)

    modes = ("unchanged", "model", "rules", "combined")
    labels = ("Do nothing", "Neural model", "Rules", "Combined")
    colors = (grey, purple, olive, olive)
    for column, (dataset, key, title, subtitle) in enumerate((
            (jfleg, "gleu", "Learner writing: JFLEG", "Corpus GLEU · 747 sentences"),
            (cweb, "f0_5", "Web writing: CWEB", "ERRANT F0.5 · 6,845 sentences"))):
        ax = fig.add_subplot(grid[0, column], facecolor=background)
        values = [100 * dataset["scores"][mode][key] for mode in modes]
        bars = ax.barh(range(4), values, height=.55, color=colors, zorder=3)
        ax.set_yticks(range(4), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 60)
        ax.set_xticks((0, 20, 40, 60))
        ax.grid(axis="x", color="#e7e3dc", zorder=0)
        ax.set_xlabel("Score (0–100); higher is better", fontsize=11, labelpad=10)
        ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=38)
        ax.text(0, 1.08, subtitle, transform=ax.transAxes, fontsize=11, color=muted)
        for bar, value in zip(bars, values):
            ax.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center", fontsize=13, weight="bold")
        ax.tick_params(axis="y", length=0, pad=10)
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)

    combined = cweb["scores"]["combined"]
    tp, fp = combined["true_positive_edits"], combined["false_positive_edits"]
    total = tp + fp
    ax = fig.add_subplot(grid[1, 0], facecolor=background)
    ax.set_title("Most web edits miss the reference", loc="left", fontsize=15, weight="bold", pad=44)
    ax.text(0, 1.14, f"Combined engine · {total:,} proposed edits", transform=ax.transAxes, fontsize=11, color=muted)
    ax.barh(0, tp, height=.45, color=olive)
    ax.barh(0, fp, left=tp, height=.45, color=orange, hatch="//", edgecolor="#ad6b3d", linewidth=.5)
    ax.text(tp/2, 0, str(tp), ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.text(tp + fp/2, 0, f"{fp} unmatched", ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.text(0, -.52, f"{tp} matched human edits", fontsize=12, color=olive)
    ax.text(0, -.95, f"{100*fp/total:.1f}% of proposed edits were unmatched", fontsize=12, weight="bold")
    ax.set_xlim(0, total)
    ax.set_ylim(-1.18, .42)
    ax.axis("off")

    accepted = combined["reference_accepted_sources"]
    changed = combined["reference_accepted_sources_changed"]
    rate = 100 * changed / accepted
    ax = fig.add_subplot(grid[1, 1], facecolor=background)
    ax.set_title("Accepted text gets changed", loc="left", fontsize=15, weight="bold", pad=44)
    ax.text(0, 1.14, "CWEB sources accepted by a human reference", transform=ax.transAxes, fontsize=11, color=muted)
    ax.barh(0, rate, height=.45, color=orange, hatch="//", edgecolor="#ad6b3d", linewidth=.5)
    ax.barh(0, 100-rate, left=rate, height=.45, color="#e3dfd7")
    ax.text(0, -.57, f"{rate:.2f}% changed", fontsize=22, weight="bold", color=ink)
    ax.text(0, -.98, f"{changed:,} of {accepted:,} reference-accepted sources", fontsize=12, color=muted)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.18, .42)
    ax.axis("off")

    jfleg_edits = jfleg["scores"]["model"]["all_sources_changed"]
    cweb_edits = cweb["scores"]["model"]["true_positive_edits"] + cweb["scores"]["model"]["false_positive_edits"]
    fig.text(.065, .12, f"Neural support is tiny: {jfleg_edits} JFLEG sentences changed; {cweb_edits} CWEB edits proposed.", fontsize=13, weight="bold")
    fig.text(.065, .083, "Different metrics: do not compare the two score panels. Scores are not grammar accuracy.", fontsize=11, color=muted)
    fig.text(.065, .055, "Unmatched edits and changes to reference-accepted text are evaluation proxies; valid alternatives may be missed.", fontsize=10, color=muted)
    measured_date = datetime.fromisoformat(receipt["measured_at_utc"]).astimezone(ZoneInfo("America/Los_Angeles")).date()
    source_commit = jfleg["code"]["baseCommit"][:7]
    fig.text(.065, .029, f"Source: public-benchmark-results.json · engine {source_commit} · {measured_date} (Pacific) · JFLEG / CWEB: CC BY-NC-SA 4.0", fontsize=10, color=muted)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "benchmark-overview"
    files = {}
    for extension in ("png", "svg"):
        path = output_dir / f"{stem}.{extension}"
        metadata = {"Date": None} if extension == "svg" else None
        fig.savefig(path, dpi=150, facecolor=background, metadata=metadata)
        if extension == "svg":
            namespace = "http://www.w3.org/2000/svg"
            ET.register_namespace("", namespace)
            tree = ET.parse(path)
            root = tree.getroot()
            root.set("role", "img")
            root.set("aria-labelledby", "benchmark-title benchmark-description")
            title = ET.Element(f"{{{namespace}}}title", {"id": "benchmark-title"})
            title.text = "Gamma EH natural writing benchmark: limited neural benefit and web-text overcorrection"
            description = ET.Element(f"{{{namespace}}}desc", {"id": "benchmark-description"})
            description.text = (
                f"JFLEG GLEU: unchanged {100*jfleg['scores']['unchanged']['gleu']:.2f}, neural {100*jfleg['scores']['model']['gleu']:.2f}, "
                f"rules and combined {100*jfleg['scores']['combined']['gleu']:.2f}. "
                f"CWEB F0.5: unchanged zero, neural {100*cweb['scores']['model']['f0_5']:.2f}, combined {100*combined['f0_5']:.2f}. "
                f"Combined CWEB edits: {tp} matched, {fp} unmatched. {changed} of {accepted} reference-accepted sources changed. "
                "Different metrics must not be compared across datasets. Results do not establish general reliability.")
            root.insert(0, description)
            root.insert(0, title)
            tree.write(path, encoding="utf-8", xml_declaration=True)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    plt.close(fig)
    manifest = {"schema": 1, "source_receipt": receipt_path.name,
                "source_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {name: version(name) for name in ("matplotlib", "numpy")}, "files": files}
    (output_dir / "benchmark-figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/public-benchmark-results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output_dir), indent=2))
