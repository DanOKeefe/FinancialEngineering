"""payout_lab — turn a probability view into an option order ticket.

Pipeline:  chain -> implied density q -> your density p -> Kelly payoff
g = (W/Z) p/q -> replication with listed calls -> priced ticket.
"""
from .providers import ChainSnapshot, MassiveProvider, SyntheticProvider, get_provider
from .implied import (ImpliedDistribution, bs_call_price, implied_distribution,
                      stitched_quotes)
from .views import geometric_blend, historical_view
from .kelly import (Ticket, expected_log_growth, log_optimal_payout,
                    replicate, summarize)

__all__ = [
    "ChainSnapshot", "MassiveProvider", "SyntheticProvider", "get_provider",
    "ImpliedDistribution", "bs_call_price", "implied_distribution", "stitched_quotes",
    "geometric_blend", "historical_view",
    "Ticket", "expected_log_growth", "log_optimal_payout", "replicate",
    "summarize",
]
