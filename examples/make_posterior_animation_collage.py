#!/usr/bin/env python3
"""Combine posterior sample frames from four maintained examples into one GIF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "pcat_posterior_samples.gif"


@dataclass(frozen=True)
class Panel:
    """Describe one posterior-frame sequence and its reader-facing label."""

    label: str
    pattern: str


PANELS = (
    Panel("Gaussian mixture | model intensity", "gmix_demo/visuals/post/fram/thiscntpmodl_*.png"),
    Panel("Point sources | residual counts", "chan_demo/visuals/post/fram/thiscntpresien02evt0_*.png"),
    Panel("Strong lens | model counts", "hst_lens/visuals/post/fram/thiscntpmodl_*.png"),
    Panel("Spectral lines | model and data", "voigt-profile/visuals/post/fram/thisscatcntpevt0_*.png"),
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Use a widely available font and retain a portable fallback."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _frame_paths(panel: Panel, examples_root: Path) -> list[Path]:
    paths = sorted(examples_root.glob(panel.pattern))
    if len(paths) < 2:
        raise RuntimeError(
            f"{panel.label} requires at least two posterior frames matching "
            f"{examples_root / panel.pattern}. Run python examples/run_examples.py first."
        )
    return paths


def _sample_index(frame_index: int, output_count: int, input_count: int) -> int:
    """Spread each available sequence over the common animation timeline."""
    if output_count == 1:
        return input_count - 1
    return round(frame_index * (input_count - 1) / (output_count - 1))


def make_collage(
    output_path: Path = DEFAULT_OUTPUT,
    examples_root: Path = ROOT,
    frame_count: int = 10,
    duration_ms: int = 500,
    panel_size: int = 480,
) -> Path:
    """Write a synchronized 2-by-2 animation of posterior sample frames."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least two.")

    sequences = [_frame_paths(panel, examples_root) for panel in PANELS]
    margin = 20
    title_height = 66
    label_height = 42
    canvas_size = (
        2 * panel_size + 3 * margin,
        title_height + 2 * (panel_size + label_height) + 3 * margin,
    )
    title_font = _font(28, bold=True)
    label_font = _font(20, bold=True)
    counter_font = _font(18)
    frames: list[Image.Image] = []

    for frame_index in range(frame_count):
        canvas = Image.new("RGB", canvas_size, "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (margin, margin),
            "PCAT posterior samples",
            fill="black",
            font=title_font,
        )
        counter = f"draw {frame_index + 1:02d} / {frame_count:02d}"
        counter_box = draw.textbbox((0, 0), counter, font=counter_font)
        draw.text(
            (canvas.width - margin - (counter_box[2] - counter_box[0]), margin + 6),
            counter,
            fill="#3f4b4b",
            font=counter_font,
        )

        for panel_index, (panel, paths) in enumerate(zip(PANELS, sequences)):
            row, column = divmod(panel_index, 2)
            x = margin + column * (panel_size + margin)
            y = title_height + margin + row * (panel_size + label_height + margin)
            source_index = _sample_index(frame_index, frame_count, len(paths))
            source_path = paths[source_index]
            print(f"Reading from {source_path}...")
            with Image.open(source_path) as source:
                panel_image = source.convert("RGB").resize(
                    (panel_size, panel_size), Image.Resampling.LANCZOS
                )
            canvas.paste(panel_image, (x, y))
            draw.rectangle(
                (x, y, x + panel_size - 1, y + panel_size - 1),
                outline="#c4cccc",
                width=2,
            )
            draw.text(
                (x, y + panel_size + 8),
                panel.label,
                fill="black",
                font=label_font,
            )
        frames.append(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-count", type=int, default=10)
    parser.add_argument("--duration-ms", type=int, default=500)
    arguments = parser.parse_args()
    make_collage(
        output_path=arguments.output,
        frame_count=arguments.frame_count,
        duration_ms=arguments.duration_ms,
    )


if __name__ == "__main__":
    main()