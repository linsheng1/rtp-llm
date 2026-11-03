"""Source-pinned thin client adapter; base44 actual input/EOF checks retained."""
import hashlib,importlib.util
from pathlib import Path
import c3_hc031_capture45_core_v1 as core45
HERE=Path(__file__).resolve();BASE=HERE.with_name('c3_hc031_capture44_client_v1.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='a487a8a2042447bbdba637d9878d5da8ea013a08b7a0ac73b9db4d75bdbfa68a'
SOURCE_SHA='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa'
PINS={'token_fixture.py': 'a1ed5084218ca8053788cc036d387a04b4cf723bfa60863c91bde206b784eb15', 'business_token_fixture.py': '41a752d933e99413bd0195da9e668d649c9b4a4e0462069f8cc85270c0c3b427', 'precision_compare.py': 'a258eb6864eb3df49920050d45c266902a9dec11111fb0433bf8af59c86e2885', 'c3_hc031_capture45_core_v1.py': '0ec7db23425938e7f5418290ffe77f3298054ee96aae6fcb03a6eea25e8bc7b6'}
spec=importlib.util.spec_from_file_location('_calibrated_client44_for45',BASE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.__file__=str(HERE);m.m=core45;m.PINS=PINS
globals().update({k:v for k,v in vars(m).items() if not k.startswith('__')})
if __name__=='__main__':raise SystemExit(main())
