"""Measure cold Numba compilation separately from a second warm synthetic transform."""
import json
import os
import subprocess
import sys
import tempfile
from time import perf_counter


def probe():
    import numpy as np
    from numba.core.event import install_timer
    from .rocket_detector import RUN
    compile_events=[]
    with install_timer("numba:compile",compile_events.append):
        start=perf_counter()
        from aeon.transformations.collection.convolution_based import MiniRocket
        x=np.random.default_rng(42).normal(size=(12,3,3200)).astype("float32")
        m=MiniRocket(n_kernels=10000,random_state=42,n_jobs=4)
        m.fit(x)
        z=m.transform(x)
        cold=perf_counter()-start
    start=perf_counter()
    np.testing.assert_array_equal(z,m.transform(x))
    stats=dict(synthetic_windows=12,actual_features=z.shape[1],cold_import_fit_transform_s=cold,
               numba_compiler_s=sum(compile_events),warm_transform_and_check_s=perf_counter()-start,
               scope="synthetic inputs, fresh temporary Numba cache; excluded from fit/search timing")
    (RUN/"compilation.json").write_text(json.dumps(stats,indent=2))
    print(stats)


if __name__=="__main__":
    if sys.argv[1:]==["--probe"]:
        probe()
    else:
        with tempfile.TemporaryDirectory() as cache:
            subprocess.run([sys.executable,"-m","src.rocket_warmup","--probe"],check=True,
                           env={**os.environ,"NUMBA_CACHE_DIR":cache})
