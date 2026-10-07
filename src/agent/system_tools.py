"""Full-system read adapters for the PeopleOS agent.

These adapters are deliberately in-process. They use the already activated
workspace runtime, so an agent cannot bypass dataset provenance, the app lock or
the API's authorization boundary by making a second HTTP request. Every result
is reduced to aggregate-safe context before it reaches the evidence bundle.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional

from src.agent.access import profile_frame, redact_for_agent
from src.agent.evidence import EvidenceItem, EvidenceKind, ToolResult, ToolResultStatus
from src.agent.tools import AgentToolDescriptor, ToolContext


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


class DataProfileTool:
    """Scan all active records for schema and measurement coverage only."""

    tool_id = 'system.data_profile'
    description = 'Scan the complete active dataset schema, missingness and field coverage without returning raw records.'
    descriptor = AgentToolDescriptor(
        tool_id=tool_id,
        description=description,
        data_scope='schema',
        api_routes=('/api/upload/status', '/api/analytics/summary'),
        availability='conditional',
    )

    def __init__(self, state: Any):
        self.state = state

    def is_available(self) -> bool:
        frame = getattr(self.state, 'raw_df', None)
        return frame is not None and not frame.empty

    def availability_reason(self) -> str:
        return 'No active workforce snapshot is loaded.' if not self.is_available() else ''

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        frame = getattr(self.state, 'raw_df', None)
        if frame is None or frame.empty:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.PARTIAL,
                summary='No active workforce snapshot is available to profile.',
                warnings=['Load and verify a workforce dataset before asking the agent to inspect its fields.'],
                duration_ms=_elapsed_ms(started),
            )
        try:
            profile = profile_frame(frame)
            evidence = EvidenceItem(
                kind=EvidenceKind.OBSERVED,
                claim=f"The active workforce snapshot was scanned across {profile['records_scanned']} records and {profile['column_count']} fields.",
                source_tool=self.tool_id,
                value=profile,
                metric='dataset_profile',
                confidence=1.0,
                dataset_version=context.dataset_version,
                metadata={
                    'confidence_basis': 'complete_in_process_schema_scan',
                    'raw_records_excluded': True,
                    'free_text_excluded': True,
                },
            )
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS,
                summary='Complete active snapshot profile calculated without exposing row values.',
                evidence=[evidence],
                duration_ms=_elapsed_ms(started),
                metadata={'records_scanned': profile['records_scanned'], 'protected_field_count': profile['protected_field_count']},
            )
        except Exception as exc:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.FAILED,
                summary='Active snapshot profiling failed.',
                error=type(exc).__name__,
                duration_ms=_elapsed_ms(started),
            )


class RuntimeStatusTool:
    """Expose runtime capability and provenance state to the agent."""

    tool_id = 'system.runtime_status'
    description = 'Read the active workspace, dataset provenance, engine readiness and model/search state.'
    descriptor = AgentToolDescriptor(
        tool_id=tool_id,
        description=description,
        data_scope='control_plane',
        api_routes=('/api/status', '/api/platform/health', '/api/intelligence/capabilities'),
        availability='available',
    )

    def __init__(self, state: Any):
        self.state = state

    def is_available(self) -> bool:
        return True

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        provenance = getattr(self.state, 'runtime_provenance', None) or {}
        engine_names = (
            'analytics_engine', 'compensation_engine', 'experience_engine', 'fairness_engine',
            'quality_of_hire_engine', 'sentiment_engine', 'structural_engine', 'succession_engine',
            'survival_engine', 'team_dynamics_engine', 'scenario_engine', 'vector_engine',
        )
        available_engines = [name.removesuffix('_engine') for name in engine_names if getattr(self.state, name, None) is not None]
        model_provenance = getattr(self.state, 'model_provenance', None)
        vector_engine = getattr(self.state, 'vector_engine', None)
        vector_ready = getattr(vector_engine, 'is_initialized', False)
        if callable(vector_ready):
            vector_ready = vector_ready()
        value = {
            'workspace_id': context.workspace_id or provenance.get('workspace_id', 'local'),
            'dataset_id': provenance.get('dataset_id'),
            'dataset_version': context.dataset_version or provenance.get('dataset_version'),
            'generation': provenance.get('generation'),
            'source_rows': provenance.get('source_rows'),
            'current_rows': provenance.get('current_rows'),
            'active_rows': provenance.get('active_rows'),
            'reporting_currency': provenance.get('reporting_currency'),
            'features_enabled': dict(getattr(self.state, 'features_enabled', {}) or {}),
            'active_model': bool(model_provenance),
            'model_ready': bool(model_provenance),
            'semantic_search_prepared': bool(vector_ready),
            'available_engines': available_engines,
        }
        evidence = EvidenceItem(
            kind=EvidenceKind.OBSERVED,
            claim='PeopleOS runtime and active-snapshot status were read before interpretation.',
            source_tool=self.tool_id,
            value=value,
            metric='runtime_capabilities',
            confidence=1.0,
            dataset_version=context.dataset_version,
            metadata={'confidence_basis': 'runtime_control_plane_state'},
        )
        return ToolResult(
            tool_id=self.tool_id,
            status=ToolResultStatus.SUCCESS,
            summary='Runtime capability and provenance state read.',
            evidence=[evidence],
            duration_ms=_elapsed_ms(started),
        )


class EngineSnapshotTool:
    """Call one passive engine summary and remove row-level material."""

    def __init__(
        self,
        state: Any,
        *,
        tool_id: str,
        description: str,
        engine_id: str,
        engine_attr: str,
        method_name: str,
        api_routes: Iterable[str],
        limitations: Iterable[str] = (),
    ):
        self.state = state
        self.tool_id = tool_id
        self.description = description
        self.engine_id = engine_id
        self.engine_attr = engine_attr
        self.method_name = method_name
        self.limitations = tuple(limitations)
        self.descriptor = AgentToolDescriptor(
            tool_id=tool_id,
            description=description,
            engine=engine_id,
            api_routes=tuple(api_routes),
            availability='conditional',
        )

    def _engine(self) -> Any:
        return getattr(self.state, self.engine_attr, None)

    def is_available(self) -> bool:
        engine = self._engine()
        return engine is not None and callable(getattr(engine, self.method_name, None))

    def availability_reason(self) -> str:
        if self.is_available():
            return ''
        return f'{self.engine_id} is not active or does not expose a passive summary in this runtime.'

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        engine = self._engine()
        method = getattr(engine, self.method_name, None) if engine is not None else None
        if not callable(method):
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.PARTIAL,
                summary=f'{self.engine_id.replace("_", " ").title()} is unavailable in the active runtime.',
                warnings=[self.availability_reason()],
                duration_ms=_elapsed_ms(started),
            )
        try:
            raw = method()
            safe_value, redaction_count = redact_for_agent(raw)
            if safe_value in (None, {}, []):
                return ToolResult(
                    tool_id=self.tool_id,
                    status=ToolResultStatus.PARTIAL,
                    summary=f'{self.engine_id.replace("_", " ").title()} returned no usable evidence for this snapshot.',
                    warnings=list(self.limitations),
                    duration_ms=_elapsed_ms(started),
                    metadata={'redacted_items': redaction_count, 'engine': self.engine_id},
                )
            evidence = EvidenceItem(
                kind=EvidenceKind.ASSUMED if self.engine_id in {'model_lab', 'forecasting', 'clustering'} else EvidenceKind.DERIVED,
                claim=f"{self.engine_id.replace('_', ' ').title()} returned an aggregate summary for the active snapshot.",
                source_tool=self.tool_id,
                value=safe_value,
                metric=f'engine_{self.engine_id}_summary',
                confidence=1.0,
                dataset_version=context.dataset_version,
                metadata={
                    'engine': self.engine_id,
                    'api_routes': list(self.descriptor.api_routes),
                    'redacted_items': redaction_count,
                    'raw_records_excluded': True,
                    'confidence_basis': 'engine_output_with_agent_redaction',
                },
            )
            warnings = list(self.limitations)
            if redaction_count:
                warnings.append('Employee-level identifiers, free text or row-level records were withheld from agent context.')
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS,
                summary=f'{self.engine_id.replace("_", " ").title()} aggregate summary completed.',
                evidence=[evidence],
                warnings=warnings,
                duration_ms=_elapsed_ms(started),
                metadata={'engine': self.engine_id, 'redacted_items': redaction_count},
            )
        except Exception as exc:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.FAILED,
                summary=f'{self.engine_id.replace("_", " ").title()} summary failed safely.',
                warnings=list(self.limitations),
                error=type(exc).__name__,
                duration_ms=_elapsed_ms(started),
            )


class ScenarioLibraryTool(EngineSnapshotTool):
    """Read scenario templates without simulating or changing a scenario."""

    def __init__(self, state: Any):
        super().__init__(
            state,
            tool_id='workforce.scenario_library',
            description='Read saved scenario templates and planning context without running or changing a scenario.',
            engine_id='scenario',
            engine_attr='scenario_engine',
            method_name='get_scenario_templates',
            api_routes=('/api/scenario/templates', '/api/scenario/history/recent'),
            limitations=(
                'Scenario results are assumption-based planning comparisons, not forecasts or approvals.',
            ),
        )


class CachedNLPTool:
    """Use only an already-computed NLP result; never trigger text inference implicitly."""

    tool_id = 'workforce.nlp'
    description = 'Read an already-computed aggregate NLP result; text inference is an explicit user action.'
    descriptor = AgentToolDescriptor(
        tool_id=tool_id,
        description=description,
        engine='nlp',
        api_routes=('/api/nlp/analysis',),
        availability='conditional',
    )

    def __init__(self, state: Any):
        self.state = state

    def is_available(self) -> bool:
        return isinstance(getattr(self.state, 'nlp_results', None), dict)

    def availability_reason(self) -> str:
        return 'No cached NLP result exists; run the explicit text analysis from the NLP surface first.' if not self.is_available() else ''

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        raw = getattr(self.state, 'nlp_results', None)
        if not isinstance(raw, dict):
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.PARTIAL,
                summary='No cached NLP result is available.',
                warnings=[self.availability_reason()],
                duration_ms=_elapsed_ms(started),
            )
        value, redaction_count = redact_for_agent(raw)
        evidence = EvidenceItem(
            kind=EvidenceKind.ASSUMED,
            claim='A previously completed aggregate NLP summary is available for this snapshot.',
            source_tool=self.tool_id,
            value=value,
            metric='nlp_summary',
            confidence=1.0,
            dataset_version=context.dataset_version,
            metadata={'redacted_items': redaction_count, 'text_inference_not_triggered': True},
        )
        return ToolResult(
            tool_id=self.tool_id,
            status=ToolResultStatus.PARTIAL,
            summary='Cached NLP summary read without starting new text inference.',
            evidence=[evidence],
            warnings=['Generated themes and sentiment labels require a reviewed text corpus; they are not workforce facts or future-outcome predictions.'],
            duration_ms=_elapsed_ms(started),
        )


class UnavailableReadTool:
    """Keep an engine/API boundary visible without reopening unsafe behavior."""

    def __init__(self, *, tool_id: str, description: str, engine_id: str, api_routes: Iterable[str], reason: str):
        self.tool_id = tool_id
        self.description = description
        self.engine_id = engine_id
        self.reason = reason
        self.descriptor = AgentToolDescriptor(
            tool_id=tool_id,
            description=description,
            engine=engine_id,
            api_routes=tuple(api_routes),
            availability='unavailable',
            agent_callable=True,
        )

    def is_available(self) -> bool:
        return False

    def availability_reason(self) -> str:
        return self.reason

    def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(
            tool_id=self.tool_id,
            status=ToolResultStatus.PARTIAL,
            summary=f'{self.engine_id.replace("_", " ").title()} is unavailable for agent use.',
            warnings=[self.reason],
        )


def build_system_tools(state: Any) -> list[Any]:
    """Build the supplemental tool catalog for one active workspace runtime."""
    return [
        RuntimeStatusTool(state),
        DataProfileTool(state),
        EngineSnapshotTool(
            state, tool_id='workforce.analytics_detail',
            description='Read the complete safe descriptive analytics summary and aggregate distributions.',
            engine_id='analytics', engine_attr='analytics_engine', method_name='get_summary_statistics',
            api_routes=('/api/analytics/summary', '/api/analytics/departments', '/api/analytics/distributions', '/api/analytics/correlations', '/api/analytics/forecast', '/api/analytics/compare-groups'),
            limitations=('Correlations and forecasts are observational or exploratory and do not establish causes or future outcomes.',),
        ),
        EngineSnapshotTool(
            state, tool_id='workforce.compensation_detail',
            description='Read the complete safe compensation analysis, pay dispersion and pay-gap screening summary.',
            engine_id='compensation', engine_attr='compensation_engine', method_name='analyze_all',
            api_routes=('/api/compensation/summary', '/api/compensation/equity', '/api/compensation/gender-pay-gap', '/api/compensation/by-tenure', '/api/compensation/analysis'),
            limitations=('Dispersion and unadjusted gaps are descriptive screening measures, not legal or adjusted-equity determinations.',),
        ),
        EngineSnapshotTool(
            state, tool_id='workforce.quality_of_hire',
            description='Read aggregate hiring-source, cohort and quality-of-hire evidence with support counts.',
            engine_id='quality_of_hire', engine_attr='quality_of_hire_engine', method_name='analyze_all',
            api_routes=('/api/quality-of-hire/analysis', '/api/quality-of-hire/source-effectiveness', '/api/quality-of-hire/correlations', '/api/quality-of-hire/insights', '/api/quality-of-hire/cohort-analysis'),
            limitations=('Quality-of-hire composites and associations require mature, exposure-aligned validation before investment or selection decisions.',),
        ),
        EngineSnapshotTool(
            state, tool_id='workforce.sentiment',
            description='Read aggregate survey, eNPS, onboarding and early-warning summaries already available in the active runtime.',
            engine_id='sentiment', engine_attr='sentiment_engine', method_name='analyze_all',
            api_routes=('/api/sentiment/analysis', '/api/sentiment/enps', '/api/sentiment/enps/trends', '/api/sentiment/enps/drivers', '/api/sentiment/onboarding', '/api/sentiment/onboarding/health', '/api/sentiment/early-warnings'),
            limitations=('Survey response bias and warning labels do not establish future departure risk.',),
        ),
        CachedNLPTool(state),
        EngineSnapshotTool(
            state, tool_id='workforce.succession',
            description='Read aggregate succession pipeline, bench strength, gaps and recorded-assessment summaries.',
            engine_id='succession', engine_attr='succession_engine', method_name='analyze_all',
            api_routes=('/api/succession/bench-strength', '/api/succession/gaps', '/api/succession/9box/summary', '/api/succession/summary'),
            limitations=('Recorded assessment summaries are not validated successor recommendations or future performance predictions.',),
        ),
        EngineSnapshotTool(
            state, tool_id='workforce.survival',
            description='Read aggregate tenure and observed-departure survival summaries with cohort support limits.',
            engine_id='survival', engine_attr='survival_engine', method_name='analyze_all',
            api_routes=('/api/survival/analysis', '/api/survival/kaplan-meier', '/api/survival/hazard-over-time', '/api/survival/cohort-insights'),
            limitations=('Survival estimates require source-specific time-origin, censoring and independent validation.',),
        ),
        EngineSnapshotTool(
            state, tool_id='workforce.team_dynamics',
            description='Read aggregate team health, composition, diversity and collaboration indicators.',
            engine_id='team_dynamics', engine_attr='team_dynamics_engine', method_name='analyze_all',
            api_routes=('/api/team/health', '/api/team/diversity', '/api/team/analysis', '/api/team/comprehensive'),
            limitations=('Team-health, diversity and collaboration labels are configured descriptive constructs, not validated outcomes.',),
        ),
        ScenarioLibraryTool(state),
        UnavailableReadTool(
            tool_id='workforce.predictive_detail',
            description='Read aggregate predictive-model diagnostics only when a governed model is active.',
            engine_id='ml',
            api_routes=('/api/predictions/model-metrics', '/api/predictions/feature-importance', '/api/predictions/risk', '/api/model-lab/validation', '/api/model-lab/sensitivity', '/api/model-lab/refinement-plan'),
            reason='No governed predictive model is active in the current workspace; PeopleOS will not substitute observed attrition for prediction.',
        ),
        UnavailableReadTool(
            tool_id='workforce.semantic_search',
            description='Read semantic-search matches after the active dataset index has been explicitly prepared.',
            engine_id='vector',
            api_routes=('/api/search', '/api/search/status'),
            reason='Semantic search is not prepared for the active snapshot; prepare an index explicitly before using retrieval.',
        ),
        UnavailableReadTool(
            tool_id='workforce.network',
            description='Read validated collaboration-network aggregates when observed relationship events are available.',
            engine_id='network',
            api_routes=('/api/network/summary',),
            reason='Observed collaboration relationship data and validated semantics are unavailable; influence and isolation rankings remain closed.',
        ),
        UnavailableReadTool(
            tool_id='workforce.causal',
            description='Read causal estimates only after an approved identification design is configured.',
            engine_id='causal',
            api_routes=('/api/causal/impact', '/api/causal/recommendations'),
            reason='Causal intervention analysis is disabled until identification, overlap and sensitivity requirements are configured.',
        ),
        UnavailableReadTool(
            tool_id='workforce.clustering',
            description='Read aggregate clustering results only after an explicit exploratory clustering run.',
            engine_id='clustering',
            api_routes=('/api/analytics/clusters',),
            reason='Clustering is exploratory and not active in the read-only runtime; employee cluster membership is never exposed.',
        ),
        UnavailableReadTool(
            tool_id='workforce.forecasting',
            description='Read a forecast only when a user supplies a metric, horizon and complete dated history.',
            engine_id='forecasting',
            api_routes=('/api/analytics/forecast',),
            reason='Forecasting needs an explicit metric and a complete dated history; the agent will not invent a horizon or history.',
        ),
    ]


def system_api_read_surfaces() -> list[dict[str, Any]]:
    """Document every governed read family represented by the agent catalog."""
    surfaces = []
    for tool in build_system_tools(None):
        surfaces.append(tool.descriptor.as_dict())
    return surfaces
