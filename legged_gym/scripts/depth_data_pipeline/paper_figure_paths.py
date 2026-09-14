"""Figure layout and one-time relocation; no inference or dataset dependencies."""
from pathlib import Path
import html
import json
import shutil
from urllib.parse import quote


def figure_path(output, filename):
    category = ("depth_and_trajectories" if str(filename).startswith(
        ("depth_geometry_", "transition_depth_", "timeline_", "closed_loop_timeline")) else "results")
    return Path(output) / "figures" / category / filename


def write_figure_index(output, files):
    output = Path(output).resolve()
    sections = []
    for category, title in (("results", "High-level results"),
                            ("depth_and_trajectories", "Depth images and comparative trajectories")):
        cards = []
        for filename in files:
            path = Path(filename).resolve()
            if path.suffix != ".png" or path.parent.name != category:
                continue
            relative = path.relative_to(output).as_posix()
            png, pdf = quote(relative), quote(str(Path(relative).with_suffix(".pdf")))
            cards.append(f'<figure><a href="{png}"><img loading="lazy" width="360" src="{png}"></a>'
                         f'<figcaption>{html.escape(path.name)} · <a href="{pdf}">PDF</a></figcaption></figure>')
        sections.append(f'<h2>{title}</h2>' + ''.join(cards))
    index = output / "figure_index.html"
    index.write_text('<!doctype html><meta charset="utf-8"><title>Paper figures</title>'
                     '<h1>Paper figures</h1><p>Click a thumbnail for the full PNG or PDF.</p>' + ''.join(sections))
    return str(index)


def relocate_existing_figures(output):
    """Move only root PNG/PDF files; repair figure links, with metadata backups.

    Intended for completed runs. Destination collisions abort before any moves.
    Bundles, CSV metrics, checkpoints, and existing figure contents are untouched.
    """
    output = Path(output).resolve()
    sources = sorted([*output.glob("*.png"), *output.glob("*.pdf")])
    moves = [(source, figure_path(output, source.name)) for source in sources]
    for source, destination in moves:
        if source.is_symlink() or destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing ambiguous/overwriting move: {source} -> {destination}")
    # Keep existing metadata/galleries recoverable without backing up big figures.
    metadata = [p for p in (output / "manifest.json", output / "figure_manifest.json",
                            output / "training_cost_manifest.json") if p.is_file()]
    documents = {p: json.loads(p.read_text()) for p in metadata}
    for path in [*metadata, output / "figure_index.html"]:
        backup = path.with_name(path.name + ".before_figure_layout.bak")
        if moves and path.is_file() and not backup.exists():
            shutil.copy2(path, backup)
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        source.rename(destination)
        after = destination.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)

    files = sorted([*output.glob("figures/*/*.png"), *output.glob("figures/*/*.pdf")])
    by_name = {path.name: path for path in files}
    def update(value):
        if isinstance(value, dict):
            return {key: update(item) for key, item in value.items()}
        if isinstance(value, list):
            return [update(item) for item in value]
        if isinstance(value, str):
            path = Path(value)
            if path.name in by_name:
                return str(by_name[path.name])  # Also relocates saved /paper/... Docker links.
            if path.name == "figure_index.html":
                return str(output / "figure_index.html")
        return value
    for path, document in documents.items():
        document = update(document)
        # depth_examples' stem is a filename prefix rather than a full PNG path.
        report = document.get("figures", document)
        if isinstance(report, dict):
            for example in report.get("depth_examples", []):
                stem = Path(example["figure_stem"]).name
                example["figure_stem"] = str(figure_path(Path(), stem))
        path.write_text(json.dumps(document, indent=2) + "\n")
    index = write_figure_index(output, files)
    report = {"output": str(output), "moved_files": len(moves), "browse_index": index,
              "counts": {category: sum(p.parent.name == category for p in files)
                         for category in ("results", "depth_and_trajectories")}}
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize figures in a completed output directory without regenerating them")
    parser.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.outputs:
        print(json.dumps(relocate_existing_figures(directory), indent=2))
