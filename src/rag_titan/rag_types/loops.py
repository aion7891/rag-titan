"""
Loop limits for the RAG types that search more than once (Corrective, Iterative).

A loop that asks a model "search again?" must never run away, so every loop:
    - has a hard cap on its rounds (max_steps);
    - stops when a search brings no chunk it did not already have (no progress);
    - can be stopped by the model ("I have what I need"), or because the
      evidence is already good (Corrective RAG's grades).

The reason it stopped is kept, so the trace can say "stopped: no new chunks
after step 2" (handbook #28: "needs loop cap and trace evaluation").

    limit = LoopLimit(max_steps=3)
    while limit.next_step():
        fresh = limit.new_chunks(search(...))
        if not fresh:
            break                      # limit.stop_reason == "no new chunks"
    print(limit.summary())             # {"steps": 2, "max_steps": 3, "stop_reason": ...}
"""

# why a loop stopped
MAX_STEPS = "max steps"
NO_NEW_CHUNKS = "no new chunks"
MODEL_DONE = "model said done"
EVIDENCE_GOOD = "evidence is good"
STOP_REASONS = [MAX_STEPS, NO_NEW_CHUNKS, MODEL_DONE, EVIDENCE_GOOD]


class LoopLimit:
    """
    Counts the rounds of a loop and remembers why it stopped.

    Args:
        max_steps: the most rounds allowed; 0 means the loop never runs.
        name: the setting's name, used in error messages (e.g. "max_retries").
    """

    def __init__(self, max_steps, name="max_steps"):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError(f"{name} must be a whole number of 0 or more, got {max_steps!r}")
        self.max_steps = max_steps
        self.steps = 0
        self.seen = set()
        self.stop_reason = None

    def next_step(self):
        """
        Start one more round. Returns False when the loop must stop: it was
        stopped already, or the cap is reached (stop_reason becomes "max steps").
        """
        if self.stop_reason is not None:
            return False
        if self.steps >= self.max_steps:
            self.stop_reason = MAX_STEPS
            return False
        self.steps += 1
        return True

    def new_chunks(self, chunks):
        """
        Return the chunks not seen before (by "id"), and remember them.

        When none are new, the loop made no progress: stop_reason becomes
        "no new chunks" and the caller should stop.
        """
        fresh = []
        for chunk in chunks:
            if chunk["id"] not in self.seen:
                self.seen.add(chunk["id"])
                fresh.append(chunk)

        if not fresh:
            self.stop(NO_NEW_CHUNKS)
        return fresh

    def remember(self, chunks):
        """Mark chunks as seen without counting a round (e.g. the first search)."""
        self.seen.update(chunk["id"] for chunk in chunks)

    def stop(self, reason):
        """Stop the loop for `reason`, one of STOP_REASONS. The first reason wins."""
        if reason not in STOP_REASONS:
            raise ValueError(f"Unknown stop reason {reason!r}; use one of {STOP_REASONS}")
        if self.stop_reason is None:
            self.stop_reason = reason

    def summary(self):
        """What the trace shows: rounds run, the cap, and why the loop stopped."""
        return {"steps": self.steps, "max_steps": self.max_steps,
                "stop_reason": self.stop_reason}
