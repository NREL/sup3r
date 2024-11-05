"""tests for using vortex to perform bias correction"""

import calendar
import os

from rex import Resource

from sup3r.bias.bias_calc_vortex import VortexMeanPrepper
from sup3r.utilities.pytest.helpers import make_fake_tif

in_heights = [10, 100, 120, 140]
out_heights = [10, 40, 80, 100, 120, 160, 200]


def test_vortex_prepper(tmpdir_factory):
    """Smoke test for vortex mean prepper."""

    td = tmpdir_factory.mktemp('tmp')
    vortex_pattern = os.path.join(td, "{month}/{month}_{height}m.tif")
    for m in [calendar.month_name[i] for i in range(1, 13)]:
        os.makedirs(f'{td}/{m}')
        for h in in_heights:
            out_file = vortex_pattern.format(month=m, height=h)
            make_fake_tif(shape=(100, 100), outfile=out_file)
    vortex_out_file = os.path.join(td, 'vortex_means.h5')

    VortexMeanPrepper.run(
        vortex_pattern,
        in_heights=in_heights,
        out_heights=out_heights,
        fp_out=vortex_out_file,
        overwrite=True,
    )
    assert os.path.exists(vortex_out_file)

    with Resource(vortex_out_file) as res:
        for h in out_heights:
            assert f'windspeed_{h}m' in res
