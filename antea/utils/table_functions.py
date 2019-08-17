from invisible_cities.reco.corrections import Correction
from invisible_cities.io.dst_io   import load_dst


def load_rpos(filename, group = "Radius",
                        node  = "f100bins"):
    dst = load_dst(filename, group, node)
    return Correction((dst.Sigma_phi.values,), dst.Rpos.values, dst.Uncertainty.values)
