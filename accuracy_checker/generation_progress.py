"""Non-mutating generation progress reporting."""

from __future__ import annotations

import logging
import time


class GenerationProgressProcessor:
    """Log decode progress without changing logits."""

    def __init__(self, prompt_tokens: int, every: int = 128,
                 logger: logging.Logger | None = None):
        self.prompt_tokens = prompt_tokens
        self.every = max(1, every)
        self.logger = logger or logging.getLogger(__name__)
        self.started = time.time()
        self.last_logged = -1

    def __call__(self, input_ids, scores):
        generated = max(0, input_ids.shape[-1] - self.prompt_tokens)
        if generated != self.last_logged and (generated == 1 or generated % self.every == 0):
            elapsed = max(time.time() - self.started, 1e-6)
            self.logger.info(
                "  [generate] %d tokens, %.2f token-step/s, elapsed %.1fs",
                generated, generated / elapsed, elapsed,
            )
            self.last_logged = generated
        return scores

