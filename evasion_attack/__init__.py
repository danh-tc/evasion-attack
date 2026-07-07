"""OA-TRaPA: Object-Aware Temporal Random Parameter Pruning attack on object detectors.

Combines OSFD's backbone feature suppress/amplify loss with RaPA's random
parameter pruning, replacing RaPA's per-iteration multi-mask averaging with a
single mask per iteration plus momentum as a temporal ensemble.
See PLAN_EXPERIMENTS.md for the staged experiment design (methods A-F).
"""
