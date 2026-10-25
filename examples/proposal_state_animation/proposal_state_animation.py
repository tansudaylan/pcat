#!/usr/bin/env python3
"""Generate retained-state and every-proposal animations from one PCAT run.

The data are a seeded simulated spectrum containing two Voigt emission lines.
PCAT fits one to three lines with Gaussian, differential-evolution, and
Student-t within-model steps plus birth, death, split, merge, and jump proposals.
The short run demonstrates output semantics rather than convergence.
"""

from tdpy.verbosity import print

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pcat import sampling
from pcat.demo import load_example_namespace
from pcat.plotting import _quantize_shared_palette


EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name


def summarize_jacobian_acceptance(worker):
    """Return exact split/merge counterfactuals for uncensored proposals."""
    proposal = np.asarray(worker.listpostindxproptype).reshape(-1).astype(int)
    log_acceptance = np.asarray(worker.listpostaccplprb).reshape(-1)
    jacobian = np.asarray(worker.listpostljcb).reshape(-1)
    accepted = np.asarray(worker.listpostboolpropaccp).reshape(-1).astype(bool)
    supported = np.asarray(worker.listpostboolpropfilt).reshape(-1).astype(bool)
    if not all(array.size == proposal.size for array in (log_acceptance, jacobian, accepted, supported)):
        raise ValueError("proposal arrays must have matching lengths")

    result = {}
    for move, name in ((3, "split"), (4, "merge")):
        selected = (
            (proposal == move) & supported & np.isfinite(log_acceptance)
            & np.isfinite(jacobian)
        )
        if not np.any(selected):
            raise ValueError(f"the run needs a finite, prior-valid {name} proposal")
        actual = np.exp(np.minimum(log_acceptance[selected], 0.0))
        result[name] = {
            "actual": actual,
            "without_jacobian": np.exp(np.minimum(log_acceptance[selected] - jacobian[selected], 0.0)),
            "log_acceptance": log_acceptance[selected],
            "log_acceptance_without_jacobian": log_acceptance[selected] - jacobian[selected],
            "jacobian": jacobian[selected],
            "accepted": accepted[selected],
        }
    return result


def plot_jacobian_acceptance(summary, output_path):
    """Plot each split/merge probability against its no-Jacobian counterfactual."""
    colors = {"split": "#A51C30", "merge": "#007C78"}
    markers = {"split": "o", "merge": "^"}
    figure, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.0), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.7)},
    )
    shift_axis, probability_axis = axes
    positions = np.arange(2, dtype=float)
    for index, name in enumerate(("split", "merge")):
        values = summary[name]
        log_shift = (values["log_acceptance_without_jacobian"] - values["log_acceptance"]) / np.log(10.0)
        shift_axis.scatter(
            np.full(log_shift.size, positions[index]), log_shift,
            color=colors[name], marker=markers[name], s=30,
            alpha=0.8, linewidths=0, label=f"{name} (n={log_shift.size})",
        )

        baseline_logratio = np.linspace(-5.0, 1.0, 301)
        median_jacobian = float(np.median(values["jacobian"]))
        probability_axis.plot(
            baseline_logratio,
            np.exp(np.minimum(0.0, baseline_logratio + median_jacobian)),
            color=colors[name], lw=1.5,
            label=f"{name}, with J",
        )
        probability_axis.plot(
            baseline_logratio, np.exp(np.minimum(0.0, baseline_logratio)),
            color=colors[name], lw=1.0, ls="--", alpha=0.7,
            label=f"{name}, without J",
        )

    shift_axis.axhline(0.0, color="0.5", lw=0.8, ls="--")
    shift_axis.set(
        xticks=positions, xticklabels=("Split", "Merge"),
        ylabel="Change in log10 acceptance ratio without J [dex]",
    )
    shift_axis.legend(loc="best", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    probability_axis.set(
        xlabel="Log acceptance ratio before Jacobian",
        ylabel="Acceptance probability",
        ylim=(0.0, 1.03),
        title="Median log|J|: split %.2f, merge %.2f" % (
            np.median(summary["split"]["jacobian"]),
            np.median(summary["merge"]["jacobian"]),
        ),
    )
    probability_axis.legend(loc="upper left", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    for axis in axes:
        axis.grid(False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def simulate_parameterization_jumps(number_jumps=40, amplitude=2.0, seed=814):
    """Compare same-dimensional linear/log parameterization jumps with and without |J|."""
    if number_jumps < 1 or not 1.0 <= amplitude <= np.e:
        raise ValueError("number_jumps must be positive and amplitude must lie in [1, e]")

    random = np.random.default_rng(seed)
    states_with_jacobian = [0]
    states_without_jacobian = [0]
    attempts = []
    log_amplitude = float(np.log(amplitude))
    for _ in range(number_jumps):
        current = states_with_jacobian[-1]
        if current == 0:
            log_target_ratio = log_amplitude
            log_jacobian = -log_amplitude
        else:
            log_target_ratio = -log_amplitude
            log_jacobian = log_amplitude
        log_ratio_with_jacobian = log_target_ratio + log_jacobian
        probability_with_jacobian = float(np.exp(min(0.0, log_ratio_with_jacobian)))
        uniform = float(random.random())
        accepted_with_jacobian = uniform < probability_with_jacobian
        states_with_jacobian.append(1 - current if accepted_with_jacobian else current)
        current_without_jacobian = states_without_jacobian[-1]
        log_ratio_without_jacobian = log_amplitude if current_without_jacobian == 0 else -log_amplitude
        probability_without_jacobian = float(np.exp(min(0.0, log_ratio_without_jacobian)))
        accepted_without_jacobian = uniform < probability_without_jacobian
        attempts.append({
            "from": current,
            "to": 1 - current,
            "log_target_ratio": log_target_ratio,
            "log_jacobian": log_jacobian,
            "probability_with_jacobian": probability_with_jacobian,
            "accepted_with_jacobian": accepted_with_jacobian,
            "from_without_jacobian": current_without_jacobian,
            "log_ratio_without_jacobian": log_ratio_without_jacobian,
            "probability_without_jacobian": probability_without_jacobian,
            "accepted_without_jacobian": accepted_without_jacobian,
            "uniform": uniform,
        })
        states_without_jacobian.append(
            1 - current_without_jacobian if accepted_without_jacobian else current_without_jacobian
        )

    return {
        "amplitude": float(amplitude),
        "log_amplitude": log_amplitude,
        "attempts": attempts,
        "states_with_jacobian": np.asarray(states_with_jacobian, dtype=int),
        "states_without_jacobian": np.asarray(states_without_jacobian, dtype=int),
    }


def _parameterization_font(size, bold=False):
    names = (
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf")
        if bold else
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_parameterization_frame(summary, frame_index):
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), "#F5F7F6")
    draw = ImageDraw.Draw(image)
    title_font = _parameterization_font(30, bold=True)
    heading_font = _parameterization_font(21, bold=True)
    body_font = _parameterization_font(18)
    small_font = _parameterization_font(15)
    ink, muted = "#1E2A2A", "#526262"
    linear_color, log_color, correct_color, counterfactual_color = (
        "#A51C30", "#087F72", "#0072B2", "#D17A00"
    )
    draw.text((48, 28), "A Jacobian-corrected jump between two parameterizations", fill=ink,
              font=title_font)
    draw.text((50, 75), "Two one-parameter models encode the same physical amplitude A", fill=muted,
              font=body_font)

    draw.rounded_rectangle((44, 120, 1236, 310), radius=12, fill="white", outline="#BCC8C7", width=2)
    draw.text((82, 145), "Model 1: linear amplitude A", fill=linear_color, font=heading_font)
    draw.text((700, 145), "Model 2: log amplitude eta = log(A)", fill=log_color, font=heading_font)
    draw.line((145, 225, 535, 225), fill=linear_color, width=5)
    draw.line((745, 225, 1135, 225), fill=log_color, width=5)
    for x, label in ((145, "1"), (535, "e"), (745, "0"), (1135, "1")):
        draw.line((x, 214, x, 237), fill=ink, width=2)
        draw.text((x - 6, 243), label, fill=muted, font=small_font)
    linear_fraction = (summary["amplitude"] - 1.0) / (np.e - 1.0)
    log_fraction = summary["log_amplitude"]
    linear_x = round(145 + 390 * linear_fraction)
    log_x = round(745 + 390 * log_fraction)
    draw.ellipse((linear_x - 11, 214, linear_x + 11, 236), fill=linear_color, outline="white", width=2)
    draw.ellipse((log_x - 11, 214, log_x + 11, 236), fill=log_color, outline="white", width=2)
    draw.text((linear_x - 45, 183), f"A={summary['amplitude']:.2f}", fill=linear_color, font=small_font)
    draw.text((log_x - 65, 183), f"eta={summary['log_amplitude']:.2f}", fill=log_color, font=small_font)
    if frame_index > 0:
        attempt = summary["attempts"][frame_index - 1]
        direction = "linear -> log" if attempt["from"] == 0 else "log -> linear"
        arrow_color = correct_color if attempt["accepted_with_jacobian"] else counterfactual_color
        draw.line((570, 220, 710, 220), fill=arrow_color, width=4)
        draw.polygon(((710, 220), (695, 211), (695, 229)), fill=arrow_color)
        draw.text((574, 190), direction, fill=arrow_color, font=small_font)

    draw.rounded_rectangle((44, 330, 1236, 566), radius=12, fill="white", outline="#BCC8C7", width=2)
    draw.text((78, 350), "Model index after each proposed jump", fill=ink, font=heading_font)
    left, right = 270, 1170
    total = len(summary["attempts"])
    rows = (("With Jacobian", summary["states_with_jacobian"], correct_color),
            ("Without Jacobian", summary["states_without_jacobian"], counterfactual_color))
    for row_index, (label, states, color) in enumerate(rows):
        y_linear = 425 + row_index * 92
        y_log = y_linear + 35
        draw.text((78, y_linear - 9), label, fill=color, font=body_font)
        draw.text((left - 42, y_linear - 8), "linear", fill=muted, font=small_font)
        draw.text((left - 42, y_log - 8), "log", fill=muted, font=small_font)
        draw.line((left, y_linear, right, y_linear), fill="#D8DFDE", width=1)
        draw.line((left, y_log, right, y_log), fill="#D8DFDE", width=1)
        points = []
        for index, state in enumerate(states[:frame_index + 1]):
            x = left + (right - left) * index / max(total, 1)
            y = y_log if state == 1 else y_linear
            points.append((round(x), y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.text((left, 535), "Jump attempt", fill=muted, font=small_font)

    if frame_index == 0:
        footer = "Initial state: linear model | A remains unchanged by each coordinate-map jump"
    else:
        attempt = summary["attempts"][frame_index - 1]
        accepted_with = sum(item["accepted_with_jacobian"] for item in summary["attempts"][:frame_index])
        accepted_without = sum(item["accepted_without_jacobian"] for item in summary["attempts"][:frame_index])
        counter_width = len(str(total))
        from_model = "linear" if attempt["from_without_jacobian"] == 0 else "log"
        footer = (
            f"Jump {frame_index:0{counter_width}d}/{total:0{counter_width}d}: "
            f"with-J log target ratio {attempt['log_target_ratio']:+.3f}, "
            f"log|J| {attempt['log_jacobian']:+.3f}, alpha {attempt['probability_with_jacobian']:.3f}\n"
            f"No-J chain from {from_model:<6}: "
            f"log target ratio {attempt['log_ratio_without_jacobian']:+.3f}, "
            f"alpha {attempt['probability_without_jacobian']:.3f}"
            f" | accepted so far: with J {accepted_with:0{counter_width}d}/{frame_index:0{counter_width}d}; "
            f"without J {accepted_without:0{counter_width}d}/{frame_index:0{counter_width}d}"
        )
    draw.multiline_text((54, 598), footer, fill=ink, font=body_font, spacing=8)
    draw.text((54, 680), "Equal model priors; p(A) is uniform on [1,e], so p(eta) is proportional to exp(eta).",
              fill=muted, font=small_font)
    return image


def make_parameterization_jump_animation(output_path, number_jumps=40, amplitude=2.0, seed=814):
    """Animate identical cross-parameterization jumps with and without the Jacobian."""
    summary = simulate_parameterization_jumps(number_jumps, amplitude, seed)
    frames = [_draw_parameterization_frame(summary, frame) for frame in range(number_jumps + 1)]
    frames = _quantize_shared_palette(frames)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output_path, summary


def summarize_proposal_types(worker):
    """Count attempts for every categorical proposal type in a worker chain."""
    proposal_types = np.asarray(worker.listpostindxproptype).reshape(-1).astype(int)
    labels = getattr(worker, 'lablproptype', None)
    if labels is None:
        from pcat.main import _retr_proposal_type_labels

        names = getattr(worker, 'nameproptype', None)
        if names is None:
            names = ('with', 'brth', 'deth', 'splt', 'merg', 'jump')
        labels = _retr_proposal_type_labels(names)
    labels = [str(label) for label in np.asarray(labels).reshape(-1)]
    return {
        label: int(np.count_nonzero(proposal_types == index))
        for index, label in enumerate(labels)
    }


def run_example(number_sweeps=120, *, time_tag=None):
    """Run the simulated Voigt problem and return animations and Jacobian plot."""
    from pcat.main import retr_pathrun

    time_tag = time_tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if datetime.strptime(time_tag, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S") != time_tag:
        raise ValueError("time_tag must use YYYYMMDD_HHMMSS")
    if number_sweeps < 24:
        raise ValueError("At least 24 sweeps are required")
    run_name = f"{RUN_NAME}_{time_tag}"
    run_root = Path(retr_pathrun(EXAMPLE_PATH, run_name))
    if (run_root / "data" / "outp").exists() or (run_root / "visuals").exists():
        raise FileExistsError(f"Refusing to overwrite proposal example {run_root}")
    if list((EXAMPLE_PATH / "visuals").glob(f"*_{time_tag}.*")):
        raise FileExistsError("Refusing to overwrite earlier proposal figures")
    namespace = load_example_namespace(
        "voigt_spectral_line_catalog/voigt_spectral_line_catalog.py"
    )
    configuration, variations, _ = namespace["build_configurations"]()
    configuration.update(variations["nomi"])
    configuration.update(
        strgcnfg=run_name,
        pathbase=str(EXAMPLE_PATH),
        truenumbelempop0=2,
        fittminmnumbelempop0=1,
        fittmaxmnumbelempop0=3,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        probtran=0.8,
        probspmr=0.5,
        probjump=0.1,
        probdemc=0.15,
        probstud=0.15,
        numbswep=number_sweeps,
        numbburn=max(2, number_sweeps // 6),
        numbsamp=max(4, number_sweeps // 3),
        numbswepplot=max(1, number_sweeps // 6),
        boolcheckconv=False,
        boolmakeplot=True,
        boolmakeplotinit=False,
        boolmakeplotfram=True,
        boolmakeplotfinlpost=False,
        makeanim=True,
        boolmakeanimprop=True,
        booldiag=False,
        numbproc=1,
        typeverb=0,
    )
    sampling.sample(**configuration)
    from pcat.main import readfile

    worker = readfile(
        str(run_root / "data" / "outp" / run_name / "gdatmodi0000post")
    )
    jacobian_summary = summarize_jacobian_acceptance(worker)
    proposal_counts = summarize_proposal_types(worker)
    missing_proposals = [name for name, count in proposal_counts.items() if count == 0]
    if missing_proposals:
        raise RuntimeError(
            "The demonstration did not exercise these proposal types: "
            + ", ".join(missing_proposals)
            + ". Increase the run length to show the full proposal roster."
        )
    jacobian_path = plot_jacobian_acceptance(
        jacobian_summary, EXAMPLE_PATH / "visuals" / f"jacobian_acceptance_effect_{time_tag}.png"
    )
    parameterization_animation, parameterization_summary = make_parameterization_jump_animation(
        EXAMPLE_PATH / "visuals" / f"parameterization_jacobian_jumps_{time_tag}.gif"
    )

    animation_root = run_root / "visuals" / "post" / "anim"
    posterior_path = animation_root / "proposal_sequence.gif"
    proposal_path = animation_root / "proposal_candidates.gif"
    if not posterior_path.is_file() or not proposal_path.is_file():
        raise RuntimeError("PCAT did not produce both requested animations.")
    final_posterior = EXAMPLE_PATH / "visuals" / f"proposal_sequence_{time_tag}.gif"
    final_proposal = EXAMPLE_PATH / "visuals" / f"proposal_candidates_{time_tag}.gif"
    shutil.copy2(posterior_path, final_posterior)
    shutil.copy2(proposal_path, final_proposal)
    posterior_path, proposal_path = final_posterior, final_proposal
    accepted = np.asarray(worker.listpostboolpropaccp).reshape(-1).astype(bool)
    record = {"run_name": run_name, "time_tag": time_tag, "sweeps": number_sweeps,
              "proposal_counts": proposal_counts, "accepted_proposals": int(accepted.sum()),
              "rejected_proposals": int((~accepted).sum()),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sampler_source_sha256": hashlib.sha256(Path(readfile.__code__.co_filename).read_bytes()).hexdigest(),
              "assumptions": "Seeded simulated two-line spectrum; short run demonstrates trace semantics rather than convergence."}
    record_path = EXAMPLE_PATH / "visuals" / f"proposal_summary_{time_tag}.json"
    print(f"Writing to {record_path}...")
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return {
        "posterior_animation": posterior_path,
        "proposal_animation": proposal_path,
        "proposal_counts": proposal_counts,
        "jacobian_figure": jacobian_path,
        "parameterization_animation": parameterization_animation,
        "parameterization_summary": {
            "acceptance_with_jacobian": float(np.mean([
                item["accepted_with_jacobian"] for item in parameterization_summary["attempts"]
            ])),
            "acceptance_without_jacobian": float(np.mean([
                item["accepted_without_jacobian"] for item in parameterization_summary["attempts"]
            ])),
            "log_amplitude": parameterization_summary["log_amplitude"],
        },
        "jacobian_summary": {
            name: {
                "count": values["actual"].size,
                "mean_actual_probability": float(np.mean(values["actual"])),
                "mean_without_jacobian": float(np.mean(values["without_jacobian"])),
                "observed_acceptance_fraction": float(np.mean(values["accepted"])),
                "median_log_jacobian": float(np.median(values["jacobian"])),
            }
            for name, values in jacobian_summary.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=120)
    parser.add_argument("--time-tag")
    arguments = parser.parse_args()
    products = run_example(arguments.numbswep, time_tag=arguments.time_tag)
    for label in (
        "posterior_animation", "proposal_animation", "jacobian_figure",
        "parameterization_animation",
    ):
        print(f"{label}: {products[label]}")
    print(f"same-dimensional Jacobian comparison: {products['parameterization_summary']}")
    print(f"proposal attempts: {products['proposal_counts']}")
    for name, result in products["jacobian_summary"].items():
        print(f"{name}: {result}")


if __name__ == "__main__":
    main()