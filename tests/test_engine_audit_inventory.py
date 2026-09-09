"""The audit must account for every engine and remain tied to reviewed source."""
import hashlib
import json
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]


def test_every_engine_and_public_method_has_an_explicit_review():
    audit=json.loads((ROOT/'model/engine_audit.json').read_text())
    actual={p.relative_to(ROOT).as_posix() for p in (ROOT/'src').glob('*_engine.py')}
    assert {row['source'] for row in audit['engines']}==actual
    assert len(audit['engines'])==len(actual)
    for row in audit['engines']:
        tree=ast.parse((ROOT/row['source']).read_text())
        methods={n.name for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and not n.name.startswith('_')}
        assert set(row['methods'])==methods
        assert row['review_status']=='reviewed_with_limits'
        assert row['findings'] and row['verification'] and row['remaining_validation']
        for path in row['interfaces']+row['renderers']+row['verification']:
            assert (ROOT/path).is_file(),path


def test_review_hashes_and_analytical_component_inventory_are_current():
    audit=json.loads((ROOT/'model/engine_audit.json').read_text())
    update_path=ROOT/'model/engine_audit_review_updates.json'
    updates=json.loads(update_path.read_text()) if update_path.exists() else {}
    expected_hashes={**audit['reviewed_file_sha256'], **updates.get('reviewed_file_sha256', {})}
    # An incremental update may only replace an already inventoried reviewed path;
    # it cannot silently expand the analytical audit scope.
    assert set(updates.get('reviewed_file_sha256', {})) <= set(audit['reviewed_file_sha256'])
    mismatches={}
    for path,expected in expected_hashes.items():
        actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        if actual!=expected:
            mismatches[path]={'expected':expected,'actual':actual}
    assert not mismatches, 'Review must be updated after changing analytical files:\n'+json.dumps(mismatches,indent=2,sort_keys=True)
    recorded={p for row in audit['engines'] for p in row['renderers']}
    analytical_components=set()
    for directory in ['charts','diagnostics']:
        analytical_components.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'web/components'/directory).glob('*.tsx'))
    assert analytical_components <= recorded


def test_human_audit_report_matches_canonical_inventory():
    import importlib.util
    spec=importlib.util.spec_from_file_location('engine_audit_report',ROOT/'scripts/render_engine_audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert (ROOT/'docs/validation/ENGINE_RENDERER_AUDIT.md').read_text()==module.render()
