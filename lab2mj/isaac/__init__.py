"""Isaac-side reference / parity dumpers (GPU box with Isaac Sim + Isaac Lab).

Run as modules from any venv that has lab2mj installed, e.g.
``python -m lab2mj.isaac.dump_reference --task <id> ...``. Nothing here is imported
by the MuJoCo-side runtime; the package stays import-free so ``lab2mj`` itself never
touches Isaac.
"""
