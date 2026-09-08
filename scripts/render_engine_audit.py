"""Render the reviewed engine inventory; never silently change review evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    audit = json.loads((ROOT / 'model/engine_audit.json').read_text())
    link = lambda path: f'[{path}](../../{path})'
    lines = [
        '# Engine and renderer accuracy audit', '',
        f'Review date: {audit["review_date"]}. Baseline: `{audit["baseline_commit"]}`.', '',
        '**All 20 engine modules have been reviewed. Enterprise predictive validity is not established.**', '',
        audit['scope_note'], '',
        'The canonical inventory is [model/engine_audit.json](../../model/engine_audit.json). '
        'It records public methods, engine consumers, interfaces, current/dormant presentation paths, '
        'verification references, limitations and hashes of the reviewed files. CI rejects missing engines or stale review hashes.', '',
        '## Coverage and findings', '',
        '| Engine | Supported interpretation | Finding / correction | Verification |',
        '|---|---|---|---|',
    ]
    for row in audit['engines']:
        lines.append(f'| {link(row["source"])} | {row["runtime_state"]} | {" ".join(row["findings"])} | {", ".join(link(p) for p in row["verification"])} |')
    lines += ['', '## Renderer and interface trace', '',
              'Reachability is based on static imports from Next page/layout entrypoints. Legacy or dormant components are explicitly included; a component existing on disk does not imply a live product route.', '',
              '| Engine | API / schema or ingestion interface | Presentation consumers |', '|---|---|---|']
    for row in audit['engines']:
        renderers = ', '.join(f'{link(p)} ({row["renderer_state"][p]})' for p in row['renderers']) or 'No dedicated routed renderer; availability/JSON API only.'
        lines.append(f'| {row["id"]} | {", ".join(link(p) for p in row["interfaces"])} | {renderers} |')
    lines += ['', 'Shared boundaries: '+', '.join(link(p) for p in audit['shared_consumers'])+'.', '',
              '## Verification evidence', '',
              'The new engine/API/export tests use manually specified expected counts, probabilities, distances, and missing-data outcomes. '
              'Clustering is checked with two perfectly separated synthetic groups using label-permutation-invariant agreement. '
              'React tests render actual components against controlled query evidence, checking numbers, units and unavailable states. '
              'They are not screenshot, browser-interaction, accessibility or visual-layout certification.', '',
              'The pinned IBM synthetic fixture and Waltons survival dataset are rerun with nine assertions. '
              'The benchmark includes an employee holdout, baseline comparison, three shuffled-label rejection controls and independently calculated Kaplan–Meier steps/RMST. '
              'See [the benchmark protocol and limits](ANALYTICS_VALIDATION.md). '
              'SHAP shape checks use controlled arrays; they do not claim a successful real-SHAP runtime benchmark.', '',
              'SHAP output units vary with the estimator and explainer mode; the renderer must reconcile baseline plus contributions with the output in matching units. '
              'Reference: [SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).', '',
              '## Remaining validation', '', '| Engine | Required evidence before stronger claims |', '|---|---|']
    for row in audit['engines']:
        lines.append(f'| {row["id"]} | {" ".join(row["remaining_validation"])} |')
    lines += ['', '## Release interpretation', '',
              'The working Windows launcher, frozen configuration lookup, packaging dependencies and smoke assertions are preserved. '
              'Source and renderer tests cannot certify packaged execution: Windows x64, macOS ARM64 and Linux x64 must each pass the Local Desktop Build on the published head.', '',
              'Review rating: **Adequate for the documented arithmetic/availability contracts; Needs attention for enterprise validation.** '
              'The missing evidence is source-verified organizational data, prospective outcomes, validated instruments and representative retrieval/text labels. '
              'The consequence is that exploratory composites, observational differences and synthetic benchmark scores cannot establish reliable future employee outcomes. '
              'Treatment: keep the documented availability/claim boundaries, then validate each intended use prospectively with predefined acceptance criteria. '
              'Do not average these different evidence states into a single accuracy or enterprise-readiness score.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    (ROOT/'docs/validation/ENGINE_RENDERER_AUDIT.md').write_text(render())
