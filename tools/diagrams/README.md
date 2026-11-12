# Chunker diagrams

`make_chunk_diagrams.py` draws one animated SVG per chunker into
`docs/assets/images/chunkers/`. It **runs the real chunkers** on the example
text, so every chunk in a diagram is real output. These files are not part of
the package.

```bash
pip install -e ".[tokens]"                                   # the token diagram needs tiktoken
python tools/diagrams/make_chunk_diagrams.py                 # draw all diagrams
python tools/diagrams/make_chunk_diagrams.py word token      # draw only some
python tools/diagrams/make_chunk_diagrams.py --check         # fail if any is out of date
```

Run `--check` after changing a chunker. If it fails, the chunker's output
changed: redraw the diagrams and look at them before committing.

| File              | What it does                                                  |
|-------------------|---------------------------------------------------------------|
| `specs.py`        | the example text, and for each chunker its units and chunks   |
| `render.py`       | turns a spec into the animated SVG (timing, highlights)       |
| `svg_parts.py`    | drawing helpers in the docs' house style (boxes, cards, text) |
| `make_chunk_diagrams.py` | the command: draws, or checks with `--check`           |

Each diagram shows the text cut into the chunker's units. The chunks are then
built one at a time: an orange frame marks the units of the chunk being made,
filled orange marks the overlap repeated from the chunk before, and the chunk
appears in the list below. With "reduce motion" turned on, all chunks are shown
at once.

# Terminal demos of the evaluation cookbook

`make_terminal_demos.py` draws the animated terminals in
`docs/assets/images/eval-terminal/`. It **runs the real** `rag-titan evaluate`
and `inspect_report.py` on the cookbook starter data, so every line is real output.

```bash
python tools/diagrams/make_terminal_demos.py           # draw all demos
python tools/diagrams/make_terminal_demos.py --check   # fail if any is out of date
```

Run `--check` after changing the evaluator or the starter data. `terminal_svg.py`
draws the terminal (typing, line timing, colours) in the style of `cli.svg`.
