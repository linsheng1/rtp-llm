"""Source-pinned thin probe adapter; all base44 admission checks retained."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve();BASE=HERE.with_name('c3_hc031_capture44_core_v1.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='0c1e0e8cb420b1a34c5a94b59afb129945c3c0024d2c0673a121a2ad25fe422d'
B1='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c'
ENV='1bb3d2ba9082c9ce17c2990ff9a8448855d68c88d793d89b8d25538d8355f6b2'
MANIFESTS={'10.44.12.175':'8614842e06e19bb5be272326ef990ff42263a837745fdbb12a22489e8f1e6ba5','10.44.61.71':'179428dbcd30ca64ce914ad0dad5f1b6d2bf5aa43682ffca8c5d87680f7c0115'}
MODEL_HELPER='1d06ade4ce4fc1368fd99a1a7fc78e79dab608ba6a7c85ccff083e66bda0713d'
spec=importlib.util.spec_from_file_location('_calibrated_core44_for45',BASE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.__file__=str(HERE);m.MODEL_HELPER=MODEL_HELPER
globals().update({k:v for k,v in vars(m).items() if not k.startswith('__')})
if __name__=='__main__':raise SystemExit(main())
