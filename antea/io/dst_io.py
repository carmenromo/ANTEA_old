import numpy  as np

from invisible_cities.reco.corrections import Correction
from invisible_cities.io.dst_io        import load_dst

def load_zr_corrections(filename, *,
                        group = "Corrections",
                        node  = "ZRcorrections",
                        **kwargs):
    dst  = load_dst(filename, group, node)
    z, r = np.unique(dst.z.values), np.unique(dst.r.values)
    f, u = dst.factor.values, dst.uncertainty.values

    return Correction((z, r),
                      f.reshape(z.size, r.size),
                      u.reshape(z.size, r.size),
                      **kwargs)
