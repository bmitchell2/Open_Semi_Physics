"""Regenerate the forming-gas-anneal and contact-resistivity figures."""
import sys
from semiconductor_lib.figures_fga_contacts import build_all

if __name__ == "__main__":
    out = build_all(sys.argv[1] if len(sys.argv) > 1 else "figs", png=True)
    print(out)
