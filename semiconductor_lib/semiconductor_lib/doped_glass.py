"""
Doped silicate glasses (PSG, BPSG): composition bookkeeping.
Backs the Semiconductor Notes page "Doped Silicate Glasses (PSG and BPSG)".
"""
_M_B, _M_P = 10.811, 30.974
_M_B2O3, _M_P2O5, _M_SIO2 = 69.62, 141.94, 60.08


def bpsg_network_composition(wt_B, wt_P):
    """
    Convert BPSG composition in wt% of elemental B and P (the usual FTIR/XRF
    reporting convention) into network-former fractions and a mean
    connectivity (bridging oxygens per network-forming atom), counting Si as
    4-connected and both trigonal B (BO3) and P (PO4 with one terminal P=O)
    as 3-connected.

    Returns dict(x_B, x_P, x_Si, connectivity). This is an upper bound on
    network disruption: in borophosphosilicate glass part of the B and P
    pair into tetrahedral BO4-/PO4+ units (BPO4-like), which restores some
    connectivity. Conversion basis: per 100 g of glass, B -> B2O3,
    P -> P2O5, balance SiO2.
    """
    nB = wt_B / _M_B
    nP = wt_P / _M_P
    m_rest = 100.0 - (nB / 2) * _M_B2O3 - (nP / 2) * _M_P2O5
    nSi = m_rest / _M_SIO2
    tot = nB + nP + nSi
    conn = (4 * nSi + 3 * nB + 3 * nP) / tot
    return dict(x_B=nB / tot, x_P=nP / tot, x_Si=nSi / tot, connectivity=conn)
