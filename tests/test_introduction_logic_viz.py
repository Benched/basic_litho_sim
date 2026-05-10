from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from content.introduction_logic_viz import (
    binary_to_int,
    draw_inverter_diagram,
    draw_logic_gate_diagram,
    draw_two_bit_adder_diagram,
    logic_output,
    not_output,
    two_bit_adder,
)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("AND", [False, False, False, True]),
        ("OR", [False, True, True, True]),
        ("XOR", [False, True, True, False]),
    ],
)
def test_logic_output_truth_tables(op: str, expected: list[bool]) -> None:
    inputs = [(False, False), (False, True), (True, False), (True, True)]
    assert [logic_output(a, b, op) for a, b in inputs] == expected


def test_not_output_inverts_value() -> None:
    assert not_output(False) is True
    assert not_output(True) is False


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ((False, False), 0),
        ((False, True), 1),
        ((True, False), 2),
        ((True, True), 3),
        ((True, False, True), 5),
    ],
)
def test_binary_to_int(bits: tuple[bool, ...], expected: int) -> None:
    assert binary_to_int(*bits) == expected


@pytest.mark.parametrize("a", [False, True])
@pytest.mark.parametrize("b", [False, True])
@pytest.mark.parametrize("c", [False, True])
@pytest.mark.parametrize("d", [False, True])
def test_two_bit_adder_matches_integer_addition(a: bool, b: bool, c: bool, d: bool) -> None:
    result = two_bit_adder(a, b, c, d)
    left = binary_to_int(a, b)
    right = binary_to_int(c, d)
    output = binary_to_int(result.carry1, result.sum1, result.sum0)
    assert output == left + right


def test_visual_helpers_return_matplotlib_figures() -> None:
    figures = [
        draw_logic_gate_diagram(True, False, "XOR"),
        draw_inverter_diagram(False),
        draw_two_bit_adder_diagram(True, False, False, True),
    ]

    assert all(isinstance(figure, Figure) for figure in figures)
