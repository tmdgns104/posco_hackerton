"""Read-only provenance / consistency review. Never unpickles unverified model files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .surface_data import ROOT


def hash_file(path: Path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def audit(root=ROOT):
    root=Path(root)
    checks=[]
    def check(relative, expected=None, description=''):
        f=root/relative
        got=hash_file(f) if f.is_file() else None
        passed=got is not None and (expected is None or got==expected)
        checks.append(dict(item=relative,description=description,status='pass' if passed else 'fail',
                           actual_sha256=got,expected_sha256=expected))
        return passed
    for run in ('surface_v1','surface_g1'):
        manifest_path=root/'experiments'/run/'manifest.json'
        if not manifest_path.exists():
            check('experiments/'+run+'/manifest.json','missing'); continue
        manifest=json.loads(manifest_path.read_text(encoding='utf8'))
        for file,digest in manifest.get('files',{}).items():
            check(f'experiments/{run}/{file}',digest,'sealed experiment artifact')
        for file,digest in manifest.get('code_sha256',{}).items():
            check(f'src/{file}',digest,'sealed source code')
    base=json.loads((root/'experiments/surface_v1/protocol.json').read_text(encoding='utf8'))
    check('experiments/surface_v1/models/resnet18-f37072fd.pth',base['weight_sha256'],'frozen encoder')
    cnn=root/'experiments/cnn_v1'; m=json.loads((cnn/'manifest.json').read_text(encoding='utf8'))
    check('experiments/cnn_v1/protocol.json',m['protocol_sha256'])
    for model in m['models']:
        if model['model'] in ('C0','C1'):
            check(f"experiments/cnn_v1/models/{model['fold']}_{model['model']}.pt",model['model_sha256'],'CNN fold model')
    check('experiments/cnn_v1/predictions.csv',m['predictions_sha256'])
    for file,digest in m['report_hashes'].items():
        check('experiments/cnn_v1/'+file,digest)
    selection=root/'experiments/inspection_binary/selection.json'
    binary=None
    if selection.is_file():
        binary=json.loads(selection.read_text(encoding='utf8'))
        for file,digest in binary['frozen_files'].items():
            check('experiments/inspection_binary/'+file,digest,'KSDD2 experimental record')
        check('experiments/surface_v1/models/resnet18-f37072fd.pth',binary['encoder_sha256'])
    else:
        check('experiments/inspection_binary/selection.json','missing')
    fold_map=json.loads((root/'artifacts/fold_map.json').read_text(encoding='utf8'))
    for f in sorted(set(fold_map.values())):
        check(f'artifacts/fold_{f}.joblib',None,'legacy per-fold SSFD checkpoint, no original digest commitment')
    if sorted(map(int,fold_map))!=list(range(1,17)):
        checks.append(dict(item='artifacts/fold_map.json',description='expected 16 groups',status='fail'))
    sel=json.loads((root/'experiments/surface_g1/selection.json').read_text(encoding='utf8'))
    g_status=json.loads((root/'experiments/surface_g1/test_status.json').read_text(encoding='utf8'))
    cnn_verdict=json.loads((root/'experiments/cnn_v1/verdict.json').read_text(encoding='utf8'))
    surface_v=json.loads((root/'experiments/surface_v1/selection.json').read_text(encoding='utf8'))
    report=dict(checks=checks,n_checked=len(checks),n_failed=sum(x['status']=='fail' for x in checks),
                model_adoption={'G1':bool(g_status['adopt']), 'V1':surface_v['selected']=='V1',
                                'V2':surface_v['selected']=='V2',
                                'K1':bool(binary and binary.get('eligible')), 
                                'C0':bool(cnn_verdict['C0']['adopt']), 'C1':bool(cnn_verdict['C1']['adopt'])},
                important_notes=[
                    'K1 failed validation selection; increased defect misses despite fewer false positives.',
                    'G1 passed existing predefined gates; it did not improve clean-image accuracy.',
                    'V1 and V2 were not selected; CNN C0 and C1 did not satisfy the original gates.',
                    'K0/K1 official source images/masks are not included; cannot independently infer their validation set.',
                    'NEU-CLS contains only defective classes; do not treat it as a normal/defect detector.',
                    'Adaptive search uses validation-only images. Its discovery score is not an independent test score.',
                    'Legacy fold artifacts have no signed original checksum commitment; hashes recorded for reference only.'])
    return report

if __name__=='__main__':
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
