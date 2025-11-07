import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    return mo, patches, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chip producton process

    ### The purpose of these pages
    These notebooks are designed to help you get familiar with the basics of how computer chips are made, with a special focus on the optics involved in that process.

    We’ll use visualizations, and interactive simulations where you can turn the knobs yourself to see how light is used to build the technology inside our phones, cars, and computers.

    ### Where to find what?

    This first notebook gives a big-picture introduction:

    - What a chip is
    - The main steps in how chips are fabricated
    - Why optics is absolutely central to making them

    After that, the following notebooks will zoom in on the optics: what are the challanges in using light to etch patterns smaller than bacteria onto silicon, and why some of the machines that do this cost hundreds of billions of euros.

    If you’re mainly here for the optics, you can skip ahead past the fabrication overview
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # What Does a Chip Do?

    Chips are everywhere in modern life. Mobile phones and laptops are the obvious examples, but even everyday objects like washing machines, cameras, and cars depend on them.
    After the COVID-19 pandemic, for instance, a global chip shortage made headlines because car production stalled. That’s no surprise when you realize that a typical car today contains more than a thousand chips — controlling everything from unlocking doors and measuring tire pressure to adjusting headlights and running the infotainment system.

    So what do all these chips actually do?
    At their core, chips handle three kinds of tasks:

    * Storing data
    * Processing information — such as performing calculations and logical decisions.
    * Coordinating operations — controlling how data moves and when things happen.

    Every modern chip, no matter how complex, builds on these same ideas. Before looking at the fabrication process we will first look more closely at what the first two steps for a chip look like.

    ### The language and logic of chips
    Most chips share a common internal language: binary — a language of zeros and ones.
    Each “bit” can be either 0 (off) or 1 (on). With only two symbols, binary can represent anything: numbers, letters, colors, even videos. In the upcomming sections we briefly look into binary as wel as the operations we can do with it that correspond to real-world tasks.
    """
    )
    return


@app.cell
def _():
    binary_a = bin(ord('a'))[2:]; 
    binary_a = '0'*(8-len(binary_a)) + binary_a
    return (binary_a,)


@app.cell
def _(binary_a, mo):
    mo.md(f"""One example of using binary is the letter 'a' as used in a notebook like this one. In binary it is represented as: {binary_a}""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Counting in binary follows a simple pattern:

    | Decimal |	Binary | power of two |
    |:---|:---:|---:|
    |0	 |0    |2^0  |
    |1	 |1    |     |
    |2	 |10   |2^1  |
    |3	 |11   |     |
    |4	 |100  |2^2  |
    |5	 |101  |     |
    |6	 |110  |     |
    |7	 |111  |     |
    |8   |1000 |2^3  |

    Just as our familiar decimal system (base 10) counts in powers of ten, binary (base 2) counts in powers of two.

    For example, in decimal
    $$111 = 1 \cdot 10^2 + 1 \cdot 10^1 + 1 \cdot 10^0 = 111$$
    while in 111 in binary becomes:
    $$111 = 1 \cdot 2^2 + 1 \cdot 2^1 + 1 \cdot 2^0 = 7$$

    Inside a chip, these 0s and 1s are represented by tiny electrical charges and signals — millions or billions of microscopic switches flipping on and off.
    By combining many of these switches, we can build circuits that add numbers, store information, and make decisions — everything that forms the foundation of modern computing.

    Below you can try it out and convert numbers to binary and vice versa.
    """
    )
    return


@app.cell
def _(mo):
    number_to_convert = mo.ui.number(start=0, value=42)
    return (number_to_convert,)


@app.cell
def _(mo, number_to_convert):
    mo.hstack((
        mo.md("* Convert a positive integer into binary: "),
        number_to_convert,
        mo.md(f"equals: {bin(number_to_convert.value)[2:]}")), justify="start")
    return


@app.cell
def _(mo):
    last_valid, set_valid = mo.state("1010")
    return last_valid, set_valid


@app.cell
def _(last_valid, mo, set_valid):
    def validate_binary(new_value):
        if all(ch in "01" for ch in new_value):
            set_valid(new_value)
        else:
            set_valid(last_valid())

    def render_binary_input():
        return mo.ui.text(
            label="* Convert binary input to an integer: ",
            value=last_valid(),
            on_change=validate_binary
        )

    binary_text_input = render_binary_input()
    return binary_text_input, validate_binary


@app.cell
def _(binary_text_input, mo, validate_binary):
    try:
        binary_converted_to_integer = int(binary_text_input.value, 2)
    except:
        validate_binary(binary_text_input.value)

    mo.hstack([
        binary_text_input, mo.md(f"equals: {binary_converted_to_integer}")
    ], justify="start")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Now that we have binary as a language, we still need to do something with it for a chip to be useful. It turns out we can reduce all the complex operations we want to perform to a couple of very simple ones, such as the AND, OR and XOR operations:

    - given two bits of input the AND operator gives an output that is 1 if the first bit AND the second bit is 1, otherwise the outpur is zero.
    - given two bits of input the OR operator gives an out that is 1 is the first bit OR the second bit is 1, where both bits being 1 also counts. However, if both bits are 0 the output will be 0.
    - Finally the XOR (eXclusive OR) gives 1 as an output if the first bit is 1 or the second bit is 1, but not both. A common table format for this is:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### AND operator
    | input 1 | input 2 | output |
    |---|---|---|
    | 0 | 0 | 0 |
    | 0 | 1 | 0 |
    | 1 | 0 | 0 |
    | 1 | 1 | 1 |
    """
    )
    return


app._unparsable_cell(
    r"""
    ### OR operator
    | input 1 | input 2 | output |
    |---|---|---|
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 1 |
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### XOR operator
    | input 1 | input 2 | output |
    |---|---|---|
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 0 |
    """
    )
    return


@app.cell
def _(mo):

    # --- Inputs ---
    A = mo.ui.checkbox(label="Switch A (checked = 1, unchecked = 0)", value=False)
    B = mo.ui.checkbox(label="Switch B (checked = 1, unchecked = 0)", value=False)

    operation = mo.ui.dropdown(
        options=["AND", "OR", "XOR"],
        value="AND",
        label="Gate Type"
    )

    # --- Logic function ---
    def logic_output(a: bool, b: bool, op: str) -> bool:
        if op == "AND": return a and b
        if op == "OR": return a or b
        if op == "XOR": return a ^ b
    return A, B, logic_output, operation


@app.cell
def _(mo):
    mo.md(r"""Here is a visualization with two switches and the logical operator controlling the lightbulb. Switch closed means it is a 1, switch open means it is a zero""")
    return


@app.cell
def _(A, B, logic_output, mo, operation, patches, plt):

    # --- Reactive diagram ---
    def logic_diagram():
        a, b, op = A.value, B.value, operation.value
        result = logic_output(a, b, op)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")

        # Draw switches
        def draw_switch(x, y, state, label):
            ax.text(x - 0.1, y + 0.15, label, fontsize=12, ha='right')
            ax.plot([x, x + 0.3], [y, y], color="black", lw=2)
            if state:
                ax.plot([x + 0.3, x + 0.6], [y, y], color="black", lw=2)
            else:
                ax.plot([x + 0.3, x + 0.6], [y + 0.15, y], color="black", lw=2)

        draw_switch(0.1, 0.7, a, "A")
        draw_switch(0.1, 0.3, b, "B")

        # Wires to gate
        ax.plot([0.7, 1.0], [0.7, 0.7], color="black", lw=2)
        ax.plot([0.7, 1.0], [0.3, 0.3], color="black", lw=2)

        # Gate box
        gate_box = patches.FancyBboxPatch(
            (1.0, 0.25), 0.5, 0.6,
            boxstyle="round,pad=0.05", fc="lightgray"
        )
        ax.add_patch(gate_box)
        ax.text(1.25, 0.55, op, fontsize=14, ha="center", va="center")

        # Output wire
        ax.plot([1.5, 2.0], [0.55, 0.55], color="black", lw=2)

        # Light bulb
        bulb_color = "gold" if result else "lightgray"
        bulb = patches.Circle((2.2, 0.55), 0.15, fc=bulb_color, ec="black", lw=1.5)
        ax.add_patch(bulb)
        ax.text(2.2, 0.85, "Output", ha="center", fontsize=10)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1)

        plt.close(fig)
        return fig

    # --- Layout ---
    mo.vstack([
        mo.hstack([A, B, operation]),
        logic_diagram()
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### NOT
    In the discussion above one very important, and somewhat counterintuitive, operator is missing. The NOT:
    """
    )
    return


@app.cell
def _(mo):


    switch = mo.ui.checkbox(label="Switch (checked = 1, unchecked = 0)", value=False)

    def not_output(x: bool) -> bool:
        return (not x)

    def status():
        x = switch.value
        y = not_output(x)
        symbol = "💡" if y else "⚫"
        return mo.md(f"### Output: {symbol} → `{int(y)}`  (input `{int(x)}`)")
    return not_output, switch


@app.cell
def _(not_output, patches, plt, switch):
    def inverter_diagram():
        x = switch.value
        y = not_output(x)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")

        # Helpers
        active_in = "black" if x else "lightgray"
        active_out = "gold" if y else "lightgray"

        # Draw a simple switch symbol (left)
        def draw_switch(x0, y0, state, label):
            ax.text(x0 - 0.1, y0 + 0.15, label, fontsize=12, ha="right")
            # base
            ax.plot([x0, x0 + 0.3], [y0, y0], color="black", lw=2)
            # blade (closed if state=True)
            if state:
                ax.plot([x0 + 0.3, x0 + 0.6], [y0, y0], color="black", lw=2)
            else:
                ax.plot([x0 + 0.3, x0 + 0.6], [y0 + 0.15, y0], color="black", lw=2)

        draw_switch(0.1, 0.55, x, "Input")

        # Input wire to inverter
        ax.plot([0.7, 1.0], [0.55, 0.55], color=active_in, lw=3)

        # Inverter symbol (triangle + bubble)
        tri = patches.Polygon([[1.0, 0.25], [1.0, 0.85], [1.45, 0.55]],
                              closed=True, fc="lightgray", ec="black", lw=1.5)
        ax.add_patch(tri)
        bubble = patches.Circle((1.55, 0.55), 0.05, fc="white", ec="black", lw=1.5)
        ax.add_patch(bubble)
        ax.text(1.225, 0.55, "NOT", ha="center", va="center", fontsize=11)

        # Output wire
        ax.plot([1.60, 2.00], [0.55, 0.55], color=active_out, lw=3)

        # Light bulb (with soft glow when on)
        bulb_center = (2.20, 0.55)
        bulb = patches.Circle(bulb_center, 0.15, fc=("gold" if y else "lightgray"),
                              ec="black", lw=1.5)
        ax.add_patch(bulb)
        if y:
            glow = patches.Circle(bulb_center, 0.28, fc="gold", alpha=0.25, ec=None)
            ax.add_patch(glow)
        ax.text(2.20, 0.85, "Output", ha="center", fontsize=10)

        # Bounds
        ax.set_xlim(0.0, 2.6)
        ax.set_ylim(0.0, 1.1)

        plt.close(fig)
        return fig

    return (inverter_diagram,)


@app.cell
def _(inverter_diagram, mo, switch):
    mo.vstack([
        switch,
        inverter_diagram()
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The NOT operator inverts the signal, given a 0 it gives a 1, and given a 1 it gives 0 as output. The truthtable is as follows:

    | Input bit | output bit |
    |---|---|
    | 0 | 1 |
    | 1 | 0 |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The operators above are so useful beccause you can build any *boolean operator*. That means that for any number of bits, we can get any binary outcome by combining these operators.
    In fact, AND and NOT are already enough. The OR and XOR operators we can make out of AND and NOT:

    A OR B = NOT ((NOT A) AND (NOT B))

    and

    A XOR B = (NOT (A AND B)) AND (NOT ((NOT A) AND (NOT B)))
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    One way to check that this works is by working out the truth table, the first columns are the inidvidual input bits. In this case A and B. Then we compute the inbetween  steps, such as NOT A and NOT B. In the last column we get the full expression, which, in this case is exactly the same column we also had for the OR operator.

    | A | B | NOT A | NOT B | (NOT A) AND (NOT B) |   NOT ((NOT A) AND (NOT B))
    |---|---|---|---|---|---|
    | 0 | 0 | 1 | 1 | 1 | 0 |
    | 0 | 1 | 1 | 0 | 0 | 1 |
    | 1 | 0 | 0 | 1 | 0 | 1 |
    | 1 | 1 | 0 | 0 | 0 | 1 |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    For more reading:

    - https://en.wikipedia.org/wiki/Boolean_function

    - https://en.wikipedia.org/wiki/Functional_completeness

    For this text, let's have a quick look at why it makes sense that these operations are useful. Let try to build addition (for small numbers), you can click on the four buttons with 0's to change them to 1's
    """
    )
    return


@app.cell
def _(mo):
    get_A0, set_A0 = mo.state(True)
    get_A1, set_A1 = mo.state(True)
    get_B0, set_B0 = mo.state(True)
    get_B1, set_B1 = mo.state(True)
    return get_A0, get_A1, get_B0, get_B1, set_A0, set_A1, set_B0, set_B1


@app.cell
def _(get_A0, get_A1, get_B0, get_B1, set_A0, set_A1, set_B0, set_B1):
    def update_A0(v):
        set_A0(not get_A0())

    def update_A1(v):
        set_A1(not get_A1())

    def update_B0(v):
        set_B0(not get_B0())

    def update_B1(v):
        set_B1(not get_B1())
    return update_A0, update_A1, update_B0, update_B1


@app.cell
def _(
    get_A0,
    get_A1,
    get_B0,
    get_B1,
    mo,
    update_A0,
    update_A1,
    update_B0,
    update_B1,
):
    bit_A0 = mo.ui.button(value = get_A0(), on_change=update_A0, label=f"{int(get_A0())}")
    bit_A1 = mo.ui.button(value = get_A1(), on_change=update_A1, label=f"{int(get_A1())}")
    bit_B0 = mo.ui.button(value = get_B0(), on_change=update_B0, label=f"{int(get_B0())}")
    bit_B1 = mo.ui.button(value = get_B1(), on_change=update_B1, label=f"{int(get_B1())}")
    return bit_A0, bit_A1, bit_B0, bit_B1


@app.cell
def _():
    def binary_to_int(*binary):
        total = 0
        for i, b in enumerate(reversed(binary)):
            if b:
                total += 2**i
        return total

    def XOR(a,b): return a ^ b
    def AND(a,b): return a & b
    def OR(a,b):  return a | b
    return AND, OR, XOR, binary_to_int


@app.cell
def _(
    AND,
    OR,
    XOR,
    binary_to_int,
    bit_A0,
    bit_A1,
    bit_B0,
    bit_B1,
    get_A0,
    get_A1,
    get_B0,
    get_B1,
    mo,
    patches,
    plt,
):
    def adder_diagram():
        # Calculations
        a1,a0,b1,b0 = get_A1(), get_A0(), get_B1(), get_B0()
        s0 = XOR(a0,b0)
        c0 = AND(a0,b0)
        xor1 = XOR(a1,b1)
        s1 = XOR(xor1,c0)
        c1 = OR(AND(a1,b1), AND(c0,xor1))
        # Draw
        fig, ax = plt.subplots(figsize=(8,6))
        ax.axis("off")

        def gate(x,y,label,color="lightgray"):
            box = patches.FancyBboxPatch((x-0.3,y-0.2),0.6,0.4,boxstyle="round,pad=0.05",fc=color,ec="black",lw=1.5)
            ax.add_patch(box)
            ax.text(x,y,label,ha="center",va="center",fontsize=10)

        def bulb(x,y,state):
            col="gold" if state else "lightgray"
            c=patches.Circle((x,y),0.15,fc=col,ec="black")
            ax.add_patch(c)
            if state:
                ax.add_patch(patches.Circle((x,y),0.25,fc="gold",alpha=0.25))

        # Output bit 1
        ax.text(0.2,2.2,"A₀",ha="center", color="lightgreen" if a0 else "black") 
        ax.text(0.2,1.8,"B₀",ha="center", color="lightgreen" if b0 else "black")
        ax.plot([0.3,0.64],[2.2,2.2],color="black",lw=2)
        ax.plot([0.3,0.64],[1.8,1.8],color="black",lw=2)
        ax.plot([1.36, 4.13],[2, 2],color="black",lw=2)
        gate(1.0,2.0,"XOR","lightgreen" if s0 else "lightgray")
        bulb(4.3,2.0,s0); ax.text(4.32,2.2,"O0",ha="center")

        # output bit 2
        ax.text(0.2,1.3,"A₀",ha="center", color="lightgreen" if a0 else "black")
        ax.text(0.2,0.9,"B₀",ha="center", color="lightgreen" if b0 else "black")
        ax.text(0.2,0.5,"A₁",ha="center", color="lightgreen" if a1 else "black")
        ax.text(0.2,0.1,"B₁",ha="center", color="lightgreen" if b1 else "black")
        gate(1.0,1.1,"AND","lightgreen" if c0 else "lightgray")
        ax.plot([0.3,0.64],[1.3, 1.3],color="black",lw=2)
        ax.plot([0.3,0.64],[0.9, 0.9],color="black",lw=2)
        gate(1.0,0.3,"XOR","lightgreen" if xor1 else "lightgray")
        ax.plot([0.3,0.64],[0.5, 0.5],color="black",lw=2)
        ax.plot([0.3,0.64],[0.1, 0.1],color="black",lw=2)
        gate(2.0, 0.7,"XOR","lightgreen" if s1 else "lightgray")
        ax.plot([1.36,1.64],[0.9, 0.9],color="black",lw=2)
        ax.plot([1.36,1.64],[0.5, 0.5],color="black",lw=2)
        ax.plot([2.36,4.13],[0.7, 0.7],color="black",lw=2)
        bulb(4.3,0.7,s1); ax.text(4.32, 0.9,"O1",ha="center")

        # output bit 3
        ax.text(0.2,-0.4,"A₀",ha="center", color="lightgreen" if a0 else "black")
        ax.text(0.2,-0.8,"B₀",ha="center", color="lightgreen" if b0 else "black")
        ax.text(0.2,-1.2,"A₁",ha="center", color="lightgreen" if a1 else "black")
        ax.text(0.2,-1.6,"B₁",ha="center", color="lightgreen" if b1 else "black")
        ax.text(0.2,-2.0,"A₁",ha="center", color="lightgreen" if a1 else "black")
        ax.text(0.2,-2.4,"B₁",ha="center", color="lightgreen" if b1 else "black")
        gate(1.0,-0.6,"AND","lightgreen" if c0 else "lightgray")
        ax.plot([0.3,0.64],[-0.4, -0.4],color="black",lw=2)
        ax.plot([0.3,0.64],[-0.8, -0.8],color="black",lw=2)
        gate(1.0,-1.4,"XOR","lightgreen" if xor1 else "lightgray")
        ax.plot([0.3,0.64],[-1.2, -1.2],color="black",lw=2)
        ax.plot([0.3,0.64],[-1.6, -1.6],color="black",lw=2)
        gate(2.0, -1,"AND","lightgreen" if AND(c0,xor1) else "lightgray")
        ax.plot([1.36,1.64],[-0.8, -0.8],color="black",lw=2)
        ax.plot([1.36,1.64],[-1.2, -1.2],color="black",lw=2)
        ax.plot([2.36,2.64],[-1.0, -1.0],color="black",lw=2)
        gate(1.0, -2.2,"AND","lightgreen" if AND(a1,b1) else "lightgray")
        ax.plot([0.36,0.64],[-2.0, -2.0],color="black",lw=2)
        ax.plot([0.36,0.64],[-2.4, -2.4],color="black",lw=2)
        ax.plot([1.36,2.64],[-2.2, -2.2],color="black",lw=2)
        gate(3.0, -1.6,"OR","lightgreen" if c1 else "lightgray")
        ax.plot([2.64,2.64],[-1.0, -1.4],color="black",lw=2)
        ax.plot([2.64,2.64],[-2.2, -1.8],color="black",lw=2)
        ax.plot([3.36,4.13],[-1.6, -1.6],color="black",lw=2)
        bulb(4.3,-1.6, c1); ax.text(4.32, -1.4,"O2",ha="center")

        ax.set_xlim(0,4.6); ax.set_ylim(-2.5,2.6)
        plt.close(fig)
        return fig

    a1,a0,b1,b0 = get_A1(), get_A0(), get_B1(), get_B0()
    s0 = XOR(a0,b0)
    c0 = AND(a0,b0)
    xor1 = XOR(a1,b1)
    s1 = XOR(xor1,c0)
    c1 = OR(AND(a1,b1), AND(c0,xor1))

    mo.vstack([
        mo.hstack((bit_A1, bit_A0, "+", bit_B1, bit_B0, "=", int(c1), int(s1), int(s0), mo.md(f"corresponding to {binary_to_int(a1, a0)}+{binary_to_int(b1, b0)} = {binary_to_int(c1, s1, s0)}")), justify="start"), 
        mo.md("order: A1 A0 + B1 B0 = O2 O1 O0"),
        adder_diagram()
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""Note that already in such a simple diagram many things are repeated, so this is by no means the simplest (in the sense of number of operations) way to construct addition.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Summary and next steps:
    Chips work with a language consisting of bits. Anything a chip should do to such a collection of bits can be constructed from simple operations such as AND and NOT. What is missing is how these abstract ideas can be created in the real world on a piece of silicon!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Transistors
    The production of semiconductor chips generally splits into two broad categories: memory and logic.

    Memory chips are designed to store information by holding electric charges in tiny structures called memory cells. Each cell represents a bit — it can be charged (“1”) or uncharged (“0”). This principle underlies technologies such as DRAM (used as working memory in laptops and computers) and NAND flash (used in SSDs).

    Logic chips, on the other hand, are designed to process information. They perform calculations and decision-making using vast networks of transistors arranged to carry out logical operations — we’ll explore how those transistors work next.

    💡 Fun fact: The fastest memory in a computer — the cache located inside logic chips like CPUs — is also made entirely of transistors. It’s called SRAM (static RAM) and can access data much faster than external memory types.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The logical operations AND, OR, and NOT are a great way to think about how logic works in principle.
    On an actual chip, these operations are realized using physical components called transistors.

    A transistor is a tiny electronic device with three terminals. Two of them form a path for electric current, but by default that path is blocked. The third terminal, called the gate, controls whether current can flow.

    When a voltage is applied to the gate, it creates an electric field that allows current to pass between the other two terminals — called the source and the drain. In effect, the transistor acts like a voltage-controlled switch: with no voltage on the gate, the switch is off; with sufficient voltage, it turns on.

    In a transistor, the gate voltage carries the logical signal. The source serves as the reference for that signal, and the drain acts as the output whose voltage depends on whether the transistor conducts. Therefore the transistor itself is not an AND operator - as the source is not a logical input.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""There are multiple ways to depict transistors in circuit diagrams, below you find one such a depiction for the (NMOS) transistor we described above:""")
    return


@app.cell
def _(mo):
    mo.image(r"C:\Users\steve\Documents\python\basic_litho_sim\content\figures\npn_transistor.png")
    return


app._unparsable_cell(
    r"""
    Another type of transistor, called a PMOS, conducts when the voltage on its gate is low. In this device, the source is typically connected to the higher supply voltage, and current flows from source to drain when the transistor is on. The drain then serves as the output, and the gate voltage controls whether that path is open or closed.
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.image(r"C:\Users\steve\Documents\python\basic_litho_sim\content\figures\pnp_transistor.png")
    return


app._unparsable_cell(
    r"""
    Combining these two transistors in a the cingle circuit below we get the inverter:
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.image(r"C:\Users\steve\Documents\python\basic_litho_sim\content\figures\CMOS_Inverter.svg",caption="CMOS inverter source: https://commons.wikimedia.org/wiki/File:CMOS_Inverter.svg")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This circuit is a CMOS inverter, made from one PMOS and one NMOS transistor, both controlled by the same input voltage A.

    When A is high (1), the NMOS transistor turns on, connecting the output Q to Vss (ground, 0). The PMOS transistor remains off, so it has no effect. The result is Q = 0.

    When A is low (0), the situation reverses: the PMOS transistor turns on, connecting Q to Vdd (the supply voltage, 1), while the NMOS remains off. Thus A = 0 produces Q = 1 — the output is the logical inverse of the input.

    Notice that at any time, only one transistor conducts. This means there is no direct current path between Vdd and Vss when the circuit is stable, minimizing power consumption — a key advantage of CMOS (C for Combined) logic.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### NAND
    There is one final basic logical operator we should mention: The NAND = NOT AND. The diagram matches its name:

    | A | B | O |
    |---|---|---|
    | 0 | 0 | 1 |
    | 1 | 0 | 1 |
    | 0 | 1 | 1 |
    | 1 | 1 | 0 |

    Recall that AND and NOT together can generate all logical operations, but you need both. The NAND can do this on its own.
    NOT A = NAND(A, A)
    which we can then use for:
    AND(A, B) = NOT(NAND(A, B)).
    """
    )
    return


app._unparsable_cell(
    r"""
    The second very nice propery, we can implement it in CMOS logic with only 4 transistors (compared to 6 for AND)
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.image(r"https://upload.wikimedia.org/wikipedia/commons/e/e2/CMOS_NAND.svg")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    In the NAND circuit, we now see four transistors — two NMOS and two PMOS.
    The two NMOS transistors conduct only when both inputs A and B are high (1). In that case, both PMOS transistors are off, and the output Out is connected to Vss (0).

    In all other cases, at least one input (A or B) is low (0). This turns on at least one PMOS transistor, connecting the output to Vdd (1). The NMOS path to ground is then interrupted, so Vss is not connected to the output.

    The resulting behavior matches exactly the truth table of a NAND gate: the output is 0 only when both inputs are 1.

    So we see that all the logical operations we would like can be build out of transistors. What remains is how we can make a chips with all those resistors.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Design and fabrication
    When you look at today’s leading chip companies, names like NVIDIA, Intel, and TSMC often come up. But these companies don’t all do the same thing.

    Aside from the types of chips they produce — GPUs in the case of NVIDIA and CPUs for Intel — the key difference lies in which part of the production chain they handle.

    NVIDIA doesn’t physically manufacture its chips. Instead, it focuses on chip design, a highly complex process that can involve more than 100 billion transistors in a single modern processor. Designing such chips is extremely challenging and is where much of the value of the final product is created.

    The manufacturing of NVIDIA’s chips is carried out by specialized companies known as foundries, such as TSMC. TSMC produces most of the world’s leading-edge chips and has unmatched expertise in semiconductor fabrication. For the most advanced process technologies, only TSMC and Samsung are currently capable of production — with TSMC typically achieving higher yields and lower costs.

    This division between chip design and chip manufacturing is known as the fabless–foundry model, and it’s the dominant structure in today’s semiconductor industry. Fabless companies create the chip designs, while foundries handle the fabrication.

    The strength of this split lies in specialization: both design and manufacturing are so complex that focusing on one enables companies to excel. Intel, historically both a designer and manufacturer (an integrated device manufacturer, or IDM), has recently fallen behind TSMC and Samsung in process technology but is now investing heavily to catch up and enter the foundry business.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The economic picture and Moore's law.
    As we saw above, only a few companies are now able to compete at the very high end of chip production due to the enormous complexity of both design and manufacturing.

    In the semiconductor world, Moore’s law is a famous “law” that states that the number of transistors on a chip roughly doubles every two years. This law — or rather, an empirical observation — was first made by Gordon Moore in 1965, although the original doubling rate was slightly faster. The main way to sustain this trend has been to shrink the size of transistors.

    Continuing this scaling means ever-smaller transistors, more expensive machines, longer and more intricate manufacturing steps, and increasingly sophisticated designs. Yet it has been worthwhile because the benefits go well beyond simply increasing transistor count:

    1. Higher performance: Smaller transistors can switch faster, allowing processors to run at higher clock speeds and perform more calculations per second.
    2. Lower power consumption: Shrinking transistors reduces the energy needed for each switching event, leading to lower heat generation and better power efficiency.
    3. Lower cost per transistor: More transistors can fit on the same silicon wafer, reducing the cost per transistor.

    These improvements have enabled compact and power-efficient devices such as smartphones, and they continue to drive advances in AI hardware, where energy efficiency directly affects the cost of training and inference.

    In recent years, however, some of these benefits have become less automatic. Clock speeds have largely stopped increasing since the mid-2000s, and cost per transistor is no longer dropping as consistently because of the extreme complexity of modern fabrication. For this reason, many in the industry now debate whether Moore’s law is slowing — or even ending. For now, the industry technology roadmap forecast a continuation of shrinking the size of transistors, in part by using new transistor designs. At the same time by making use of other solutions, such as smart integration of more functionalities on chips under the name "more than Moore".
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Actual transistors, semiconductors and Integrated Circuits
    We’ve seen that transistors are the fundamental building blocks of chips, and that by combining many of them, we can perform any operation we like on binary data. In this section, we’ll take a closer look at what a physical transistor actually is.

    Chips — or integrated circuits — are typically built on thin circular wafers made from monocrystalline silicon. Silicon is especially useful because it is a semiconductor: it conducts electricity better than an insulator, but not as freely as a metal such as copper.

    The key property of silicon is that its electrical behavior can be precisely tuned by introducing tiny amounts of certain other elements — typically phosphorus or boron. Pure silicon does not conduct electricity well because its electrons are tightly bound to atoms and have limited mobility.

    Adding phosphorus, which has one more electron than silicon, introduces extra electrons that can move freely — creating what’s known as n-type silicon (“n” for negative). Adding boron, which has one fewer electron, creates p-type silicon, where the missing electrons leave behind “holes” that act as positive charge carriers.

    You can think of it this way: when a hole is filled by a negatively charged electron, that electron leaves behind an empty space — another hole — effectively making it seem as though a positively charged particle is moving in the opposite direction.

    When regions of p-type and n-type silicon are placed next to each other, something interesting happens at their boundary. Electrons from the n-type side (where they are plentiful) tend to diffuse into the p-type side, where there are many holes. Similarly, holes from the p-type side move into the n-type region.

    As these charge carriers cross the boundary, they recombine — electrons fill holes — leaving behind fixed charged atoms on both sides: negatively charged acceptor ions in the p-type region and positively charged donor ions in the n-type region. This forms a thin area around the junction that is depleted of mobile charge carriers, known as the depletion region.

    The result is an internal electric field that points from the n-type side to the p-type side. This field acts like a barrier: it prevents further movement of electrons and holes across the junction unless an external voltage is applied.

    If we apply a voltage that makes the p-side positive relative to the n-side (called forward bias), the barrier becomes smaller, allowing current to flow easily — this is how a diode conducts. If we reverse the voltage, the barrier grows and almost no current flows — the diode becomes an insulator.

    This already gives us nontrivial behavior from semiconductors. To create a transistor, we build on this principle by arranging n-type and p-type regions to form the source and drain of the device, separated by a channel that is normally non-conductive.

    The input of the transistor — the gate — is not directly connected to the channel. Instead, when a voltage is applied to the gate, it creates an electric field that attracts charge carriers into the channel, making it conductive and allowing current to flow between the source and drain. In other words, the transistor turns on.
    """
    )
    return


if __name__ == "__main__":
    app.run()
