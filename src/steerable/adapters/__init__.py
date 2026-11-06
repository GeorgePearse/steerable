"""Adapters bind a specific training loop's objects to steerable keys and ship a default policy table.

Each adapter imports its upstream package lazily, so this package imports without torch,
pytorch_lightning, rfdetr or D-FINE installed. See docs/adapters.md for the per-knob tables and
for writing an adapter for another DETR-style loop.
"""
