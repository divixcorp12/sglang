"""Promoted FP32 scratch observer; new identities, unchanged sealed v4 sources."""
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"performance-v4"))
spec=importlib.util.spec_from_file_location("scratch_prior_v4",ROOT/"performance-v4/observe.py")
v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
sys.path.insert(0,str(HERE))
import softmax_candidates
from policy import extend
prior_identity=v4.v3.v2.fast.observer.identity
def identity(*args,**kwargs):
    return extend(prior_identity(*args,**kwargs))
v4.v3.v2.fast.observer.identity=identity
if __name__=="__main__":
    v4.v3.v2.cadence(sys.argv)
    if (ROOT/"full-scoring-v8/calibration.hold.json").exists():
        raise RuntimeError("Calibration explicitly held")
    with v4.expert_pool.install(),v4.pinned_staging.install(),softmax_candidates.install("fp32_inplace"),v4.v3.v2.fast.install(v4.v3.v2.fast.observer.sequential_pipeline,v4.v3.v2.fast.event),v4.v3.v2.install_saliency(),v4.v3.writers():
        raise SystemExit(v4.v3.v2.fast.observer.main())
