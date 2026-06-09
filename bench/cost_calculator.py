"""Pure, dependency-free cost math for LLM inference.

Vendored verbatim from the sibling ``mini-llm-gateway`` repo so cost numbers are
consistent across the portfolio. Everything here is a pure function of its
arguments. The only addition over the gateway copy is ``get_gpu_hourly_rate``,
a convenience wrapper around the ``GPU_HOURLY_RATE`` env var (the gateway keeps
that read in its ``cost_tracker`` module instead).

The central idea: an inference server costs money per wall-clock hour
(``gpu_hourly_rate``), and produces tokens at some throughput
(``tokens_per_sec``). Divide the two and you get the marginal cost of a token.
The headline portfolio metric is cost per *million* tokens, because that is the
unit the rest of the industry quotes.

Run the doctests with::

    python -m doctest cost_calculator.py -v
"""

import os

SECONDS_PER_HOUR = 3600
TOKENS_PER_MILLION = 1_000_000

# GPU_HOURLY_RATE is the swappable economic input. On M1 it stands in for a
# cloud GPU price (e.g. an A10G at ~$0.80/hr) or the amortized local machine.
DEFAULT_GPU_HOURLY_RATE = 0.80


def get_gpu_hourly_rate() -> float:
    """Read the configured GPU hourly rate from the environment.

    Benchmarks-repo convenience; the gateway reads this in cost_tracker.py.
    """
    return float(os.getenv("GPU_HOURLY_RATE", DEFAULT_GPU_HOURLY_RATE))


def cost_per_token(gpu_hourly_rate: float, tokens_per_sec: float) -> float:
    """USD cost of producing one token at a given throughput.

    ``cost_per_token = gpu_hourly_rate / (tokens_per_sec * 3600)``

    >>> round(cost_per_token(0.80, 100.0), 12)
    2.222222e-06
    >>> cost_per_token(0.80, 0)
    0.0
    """
    if tokens_per_sec <= 0:
        return 0.0
    return gpu_hourly_rate / (tokens_per_sec * SECONDS_PER_HOUR)


def cost_per_million_tokens(gpu_hourly_rate: float, tokens_per_sec: float) -> float:
    """USD cost per 1,000,000 tokens. The headline portfolio metric.

    >>> round(cost_per_million_tokens(0.80, 100.0), 6)
    2.222222
    >>> cost_per_million_tokens(0.80, 0)
    0.0
    """
    return cost_per_token(gpu_hourly_rate, tokens_per_sec) * TOKENS_PER_MILLION


def cost_per_request(gpu_hourly_rate: float, tokens_per_sec: float,
                     input_tokens: int, output_tokens: int) -> float:
    """USD cost of a single request given its token counts.

    Cost is driven by the total tokens the server had to move through it
    (prompt + completion) at the observed throughput.

    >>> round(cost_per_request(0.80, 100.0, 45, 128), 10)
    0.0003844444
    >>> cost_per_request(0.80, 0, 45, 128)
    0.0
    """
    total_tokens = input_tokens + output_tokens
    return cost_per_token(gpu_hourly_rate, tokens_per_sec) * total_tokens


def cost_from_gpu_seconds(gpu_hourly_rate: float, seconds: float) -> float:
    """USD cost of occupying the GPU for ``seconds`` of wall-clock time.

    This is the time-based view of cost: the server is billed per wall-clock
    hour regardless of what it is doing, so any phase that holds the GPU for
    ``seconds`` costs ``gpu_hourly_rate / 3600 * seconds``.

    >>> round(cost_from_gpu_seconds(0.80, 3600), 6)
    0.8
    >>> round(cost_from_gpu_seconds(0.80, 6.19), 8)
    0.00137556
    >>> cost_from_gpu_seconds(0.80, 0)
    0.0
    """
    if seconds <= 0:
        return 0.0
    return gpu_hourly_rate / SECONDS_PER_HOUR * seconds


def prefill_decode_cost_split(gpu_hourly_rate: float, ttft_s: float,
                              total_elapsed_s: float) -> dict:
    """Split one request's GPU cost into prefill vs decode dollars.

    A request holds the GPU for ``total_elapsed_s`` of wall-clock time. Time to
    first token (``ttft_s``) is the boundary: everything up to the first token
    is the prefill phase (process the whole prompt in one compute-bound pass);
    everything after is the decode phase (emit output tokens one at a time,
    memory-bound). Because the GPU is billed by time, the dollar split is just
    that time split scaled by the hourly rate.

    Returns a dict with prefill/decode/total USD and the prefill fraction. For
    typical multi-hundred-token generations prefill is a small slice of total
    cost; for short completions over long prompts it can dominate.

    >>> s = prefill_decode_cost_split(0.80, 0.188, 6.19)
    >>> round(s["prefill_cost_usd"], 8)
    4.178e-05
    >>> round(s["decode_cost_usd"], 8)
    0.00133378
    >>> round(s["total_cost_usd"], 8)
    0.00137556
    >>> round(s["prefill_fraction"], 4)
    0.0304
    >>> prefill_decode_cost_split(0.80, 0, 0)["prefill_fraction"]
    0.0
    """
    ttft_s = max(0.0, ttft_s)
    total_elapsed_s = max(0.0, total_elapsed_s)
    # Decode time is whatever wall-clock remains after the first token lands.
    # Clamp at 0 in case of measurement noise where ttft >= elapsed.
    decode_s = max(0.0, total_elapsed_s - ttft_s)
    prefill_cost = cost_from_gpu_seconds(gpu_hourly_rate, ttft_s)
    decode_cost = cost_from_gpu_seconds(gpu_hourly_rate, decode_s)
    total_cost = prefill_cost + decode_cost
    fraction = prefill_cost / total_cost if total_cost > 0 else 0.0
    return {
        "prefill_cost_usd": prefill_cost,
        "decode_cost_usd": decode_cost,
        "total_cost_usd": total_cost,
        "prefill_fraction": fraction,
    }


def effective_cost_per_token(nominal_cost_per_token: float, utilization: float) -> float:
    """Adjust nominal cost for real-world GPU utilization.

    ``effective = nominal / utilization`` (utilization in ``(0, 1]``).

    Idle GPU time still costs money, so the real cost of a token is higher than
    the theoretical cost when the GPU is not fully saturated.

    >>> effective_cost_per_token(2e-06, 1.0)
    2e-06
    >>> effective_cost_per_token(2e-06, 0.5)
    4e-06
    >>> effective_cost_per_token(2e-06, 0)
    0.0
    """
    if utilization <= 0:
        return 0.0
    return nominal_cost_per_token / utilization


if __name__ == "__main__":
    import doctest

    failures, _ = doctest.testmod(verbose=False)
    if not failures:
        print("All cost_calculator doctests passed.")
