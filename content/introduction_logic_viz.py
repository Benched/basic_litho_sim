"""Notebook-local helpers for the introduction logic visualizations."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path


@dataclass(frozen=True)
class TwoBitAdderResult:
    sum0: bool
    carry0: bool
    half_sum1: bool
    sum1: bool
    carry1: bool


@dataclass(frozen=True)
class CircuitStyle:
    background: str = "#ffffff"
    text: str = "#172033"
    text_muted: str = "#5f6b7a"
    gate_fill: str = "#f7f9fc"
    gate_active_fill: str = "#e9f8ef"
    gate_edge: str = "#172033"
    signal_on: str = "#f5b700"
    signal_off: str = "#b7c0cc"
    node_on: str = "#ffd60a"
    node_off: str = "#eef1f5"
    input_on: str = "#2fbf71"


STYLE = CircuitStyle()


def logic_output(a: bool, b: bool, op: str) -> bool:
    if op == "AND":
        return a and b
    if op == "OR":
        return a or b
    if op == "XOR":
        return a ^ b
    raise ValueError(f"Unsupported logic operation: {op!r}")


def not_output(x: bool) -> bool:
    return not x


def binary_to_int(*binary: bool) -> int:
    total = 0
    for i, bit in enumerate(reversed(binary)):
        if bit:
            total += 2**i
    return total


def two_bit_adder(a1: bool, a0: bool, b1: bool, b0: bool) -> TwoBitAdderResult:
    sum0 = a0 ^ b0
    carry0 = a0 and b0
    half_sum1 = a1 ^ b1
    sum1 = half_sum1 ^ carry0
    carry1 = (a1 and b1) or (carry0 and half_sum1)
    return TwoBitAdderResult(sum0, carry0, half_sum1, sum1, carry1)


def _blank_figure(figsize: tuple[float, float], xlim: tuple[float, float], ylim: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(STYLE.background)
    ax.set_facecolor(STYLE.background)
    ax.axis("off")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return fig, ax


def _signal_color(state: bool) -> str:
    return STYLE.signal_on if state else STYLE.signal_off


def _node_color(state: bool) -> str:
    return STYLE.node_on if state else STYLE.node_off


def _wire(ax, points: list[tuple[float, float]], state: bool, lw: float = 3.0) -> None:
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color=_signal_color(state),
        lw=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1.35 if state else 1,
    )


def _draw_input_node(ax, x: float, y: float, label: str, state: bool) -> None:
    ax.add_patch(
        patches.Circle(
            (x, y),
            0.16,
            fc=STYLE.input_on if state else STYLE.node_off,
            ec=STYLE.gate_edge,
            lw=1.6,
            zorder=3,
        )
    )
    ax.text(x - 0.34, y, label, ha="right", va="center", color=STYLE.text, fontsize=11)
    ax.text(x, y, str(int(state)), ha="center", va="center", color=STYLE.text, fontsize=10)


def _draw_output_node(ax, x: float, y: float, label: str, state: bool) -> None:
    ax.add_patch(
        patches.Circle(
            (x, y),
            0.23,
            fc=_node_color(state),
            ec=STYLE.gate_edge,
            lw=2,
            zorder=3,
        )
    )
    if state:
        ax.add_patch(
            patches.Circle((x, y), 0.35, fc=STYLE.node_on, alpha=0.22, ec=None, zorder=2)
        )
    ax.text(x, y + 0.43, label, ha="center", va="center", color=STYLE.text, fontsize=11)
    ax.text(x, y, str(int(state)), ha="center", va="center", color=STYLE.text, fontsize=10)


def _output_input_x(x: float) -> float:
    return x - 0.23


def _draw_switch(ax, x: float, y: float, state: bool, label: str) -> None:
    ax.text(x - 0.45, y + 0.3, label, ha="left", va="center", color=STYLE.text, fontsize=12)
    ax.text(
        x - 0.45,
        y - 0.28,
        f"input {int(state)}",
        ha="left",
        va="center",
        color=STYLE.text_muted,
        fontsize=9,
    )
    ax.add_patch(patches.Circle((x + 0.1, y), 0.055, fc=STYLE.gate_edge, ec="none", zorder=3))
    ax.add_patch(patches.Circle((x + 0.72, y), 0.055, fc=STYLE.gate_edge, ec="none", zorder=3))
    if state:
        _wire(ax, [(x + 0.1, y), (x + 0.72, y)], True)
    else:
        _wire(ax, [(x + 0.1, y), (x + 0.45, y + 0.28)], False)


def _draw_gate(ax, x: float, y: float, kind: str, active: bool = False, scale: float = 1.0) -> None:
    kind = kind.upper()
    width = 0.9 * scale
    height = 0.62 * scale
    fill = STYLE.gate_active_fill if active else STYLE.gate_fill

    if kind == "AND":
        verts = [
            (x - width / 2, y - height / 2),
            (x - width / 2, y + height / 2),
            (x, y + height / 2),
            (x + width / 2, y + height / 2),
            (x + width / 2, y),
            (x + width / 2, y - height / 2),
            (x, y - height / 2),
            (x - width / 2, y - height / 2),
        ]
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CLOSEPOLY,
        ]
        patch = patches.PathPatch(Path(verts, codes), fc=fill, ec=STYLE.gate_edge, lw=2, zorder=2)
        ax.add_patch(patch)
    elif kind in {"OR", "XOR"}:
        verts = [
            (x - width / 2, y - height / 2),
            (x - width / 8, y - height / 2),
            (x + width / 2, y),
            (x - width / 8, y + height / 2),
            (x - width / 2, y + height / 2),
            (x - width / 4, y),
            (x - width / 2, y - height / 2),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]
        patch = patches.PathPatch(Path(verts, codes), fc=fill, ec=STYLE.gate_edge, lw=2, zorder=2)
        ax.add_patch(patch)
        if kind == "XOR":
            extra = Path(
                [
                    (x - width / 2 - 0.12 * scale, y - height / 2),
                    (x - width / 4 - 0.12 * scale, y),
                    (x - width / 2 - 0.12 * scale, y + height / 2),
                ],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )
            ax.add_patch(patches.PathPatch(extra, fc="none", ec=STYLE.gate_edge, lw=2, zorder=2))
    else:
        raise ValueError(f"Unsupported gate kind: {kind!r}")

    ax.text(x + 0.02, y, kind, ha="center", va="center", fontsize=9.5, color=STYLE.text, zorder=4)


def _gate_input_x(x: float, scale: float = 1.0) -> float:
    return x - (0.9 * scale) / 2 + 0.05 * scale


def _gate_output_x(x: float, scale: float = 1.0) -> float:
    return x + (0.9 * scale) / 2 - 0.02 * scale


def _not_input_x(x: float, scale: float = 1.0) -> float:
    return x - (0.9 * scale) / 2 + 0.04 * scale


def _not_output_x(x: float, scale: float = 1.0) -> float:
    return x + (0.9 * scale) / 2 + 0.09 * scale


def _draw_not_gate(ax, x: float, y: float, active: bool = False, scale: float = 1.0) -> None:
    width = 0.9 * scale
    height = 0.72 * scale
    fill = STYLE.gate_active_fill if active else STYLE.gate_fill
    triangle = patches.Polygon(
        [
            (x - width / 2, y - height / 2),
            (x - width / 2, y + height / 2),
            (x + width / 3, y),
        ],
        closed=True,
        fc=fill,
        ec=STYLE.gate_edge,
        lw=2,
        zorder=2,
    )
    ax.add_patch(triangle)
    ax.add_patch(
        patches.Circle((x + width / 2, y), 0.09 * scale, fc=STYLE.background, ec=STYLE.gate_edge, lw=2, zorder=3)
    )
    ax.text(x - 0.08 * scale, y, "NOT", ha="center", va="center", fontsize=9.5, color=STYLE.text, zorder=4)


def draw_logic_gate_diagram(a: bool, b: bool, op: str):
    result = logic_output(a, b, op)
    fig, ax = _blank_figure((7.0, 3.0), (0, 7.2), (0, 3.0))
    gate_x, gate_y, gate_scale = 3.75, 1.5, 1.25
    gate_in = _gate_input_x(gate_x, gate_scale)
    gate_out = _gate_output_x(gate_x, gate_scale)

    _draw_switch(ax, 0.6, 2.15, a, "A")
    _draw_switch(ax, 0.6, 0.95, b, "B")
    _wire(ax, [(1.35, 2.15), (2.75, 2.15), (2.75, 1.7), (gate_in, 1.7)], a)
    _wire(ax, [(1.35, 0.95), (2.75, 0.95), (2.75, 1.3), (gate_in, 1.3)], b)
    _draw_gate(ax, gate_x, gate_y, op, result, scale=gate_scale)
    output_x = 6.15
    _wire(ax, [(gate_out, gate_y), (_output_input_x(output_x), gate_y)], result)
    _draw_output_node(ax, output_x, 1.5, "Output", result)

    plt.close(fig)
    return fig


def draw_inverter_diagram(x: bool):
    y = not_output(x)
    fig, ax = _blank_figure((7.0, 2.6), (0, 7.0), (0, 2.4))
    gate_x, gate_y, gate_scale = 3.55, 1.2, 1.35

    _draw_switch(ax, 0.75, 1.2, x, "Input")
    _wire(ax, [(1.5, 1.2), (_not_input_x(gate_x, gate_scale), 1.2)], x)
    _draw_not_gate(ax, gate_x, gate_y, y, scale=gate_scale)
    output_x = 6.15
    _wire(ax, [(_not_output_x(gate_x, gate_scale), 1.2), (_output_input_x(output_x), 1.2)], y)
    _draw_output_node(ax, output_x, 1.2, "Output", y)

    plt.close(fig)
    return fig


def draw_two_bit_adder_diagram(a1: bool, a0: bool, b1: bool, b0: bool):
    result = two_bit_adder(a1, a0, b1, b0)
    carry_from_inputs = a1 and b1
    carry_from_carry = result.carry0 and result.half_sum1

    fig, ax = _blank_figure((9.2, 5.4), (0, 9.2), (0, 5.4))
    left_gate_x = 2.25
    mid_gate_x = 4.35
    final_gate_x = 6.45
    left_gate_in = _gate_input_x(left_gate_x)
    left_gate_out = _gate_output_x(left_gate_x)
    mid_gate_in = _gate_input_x(mid_gate_x)
    mid_gate_out = _gate_output_x(mid_gate_x)
    final_gate_in = _gate_input_x(final_gate_x)
    final_gate_out = _gate_output_x(final_gate_x)
    sum1_lane_x = 3.35
    carry1_lane_x = 3.62

    _draw_group_label(ax, 1.9, 5.05, "bit 0 half adder")
    _draw_group_label(ax, 4.35, 5.05, "bit 1 with carry")
    _draw_group_label(ax, 6.5, 5.05, "final carry")

    inputs = {
        "A0": (0.65, 4.55, a0),
        "B0": (0.65, 3.95, b0),
        "A1": (0.65, 2.75, a1),
        "B1": (0.65, 2.15, b1),
    }
    for label, (x, y, state) in inputs.items():
        _draw_input_node(ax, x, y, label, state)

    _wire(ax, [(0.82, 4.55), (1.38, 4.55), (1.38, 4.38), (left_gate_in, 4.38)], a0)
    _wire(ax, [(0.82, 3.95), (1.38, 3.95), (1.38, 4.12), (left_gate_in, 4.12)], b0)
    _draw_gate(ax, left_gate_x, 4.25, "XOR", result.sum0)
    output_x = 8.2
    _wire(ax, [(left_gate_out, 4.25), (_output_input_x(output_x), 4.25)], result.sum0)
    _draw_output_node(ax, output_x, 4.25, "O0", result.sum0)

    _wire(ax, [(0.82, 4.55), (1.08, 4.55), (1.08, 3.48), (left_gate_in, 3.48)], a0)
    _wire(ax, [(0.82, 3.95), (1.22, 3.95), (1.22, 3.22), (left_gate_in, 3.22)], b0)
    _draw_gate(ax, left_gate_x, 3.35, "AND", result.carry0)
    _wire(ax, [(left_gate_out, 3.35), (sum1_lane_x, 3.35), (sum1_lane_x, 3.12), (mid_gate_in, 3.12)], result.carry0)
    _draw_small_label(ax, 3.03, 3.55, "carry")

    _wire(ax, [(0.82, 2.75), (1.38, 2.75), (1.38, 2.68), (left_gate_in, 2.68)], a1)
    _wire(ax, [(0.82, 2.15), (1.38, 2.15), (1.38, 2.42), (left_gate_in, 2.42)], b1)
    _draw_gate(ax, left_gate_x, 2.55, "XOR", result.half_sum1)
    _wire(ax, [(left_gate_out, 2.55), (sum1_lane_x, 2.55), (sum1_lane_x, 2.88), (mid_gate_in, 2.88)], result.half_sum1)

    _draw_gate(ax, mid_gate_x, 3.0, "XOR", result.sum1)
    _wire(ax, [(mid_gate_out, 3.0), (_output_input_x(output_x), 3.0)], result.sum1)
    _draw_output_node(ax, output_x, 3.0, "O1", result.sum1)

    _wire(ax, [(left_gate_out, 3.35), (carry1_lane_x, 3.35), (carry1_lane_x, 1.82), (mid_gate_in, 1.82)], result.carry0)
    _wire(ax, [(left_gate_out, 2.55), (3.2, 2.55), (3.2, 1.58), (mid_gate_in, 1.58)], result.half_sum1)
    _draw_gate(ax, mid_gate_x, 1.7, "AND", carry_from_carry)

    _wire(ax, [(0.82, 2.75), (1.08, 2.75), (1.08, 0.98), (mid_gate_in, 0.98)], a1)
    _wire(ax, [(0.82, 2.15), (1.22, 2.15), (1.22, 0.72), (mid_gate_in, 0.72)], b1)
    _draw_gate(ax, mid_gate_x, 0.85, "AND", carry_from_inputs)

    _wire(ax, [(mid_gate_out, 1.7), (5.6, 1.7), (5.6, 1.48), (final_gate_in, 1.48)], carry_from_carry)
    _wire(ax, [(mid_gate_out, 0.85), (5.6, 0.85), (5.6, 1.22), (final_gate_in, 1.22)], carry_from_inputs)
    _draw_gate(ax, final_gate_x, 1.35, "OR", result.carry1)
    _wire(ax, [(final_gate_out, 1.35), (_output_input_x(output_x), 1.35)], result.carry1)
    _draw_output_node(ax, output_x, 1.35, "O2", result.carry1)

    plt.close(fig)
    return fig


def _draw_group_label(ax, x: float, y: float, label: str) -> None:
    ax.text(
        x,
        y,
        label.upper(),
        ha="center",
        va="center",
        color=STYLE.text_muted,
        fontsize=8.5,
        fontweight="bold",
    )


def _draw_small_label(ax, x: float, y: float, label: str) -> None:
    ax.text(x, y, label, ha="center", va="center", color=STYLE.text_muted, fontsize=8.5)
