"""Upload and governed import-review route handlers."""

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.dependencies import AppState, get_app_state
from src.platform.local_dataset_store import remove_dataset_artifact, save_dataset_artifact
from src.column_mapping_llm import mapping_data_scope, suggest_column_mappings
from src.data_loader import DataLoader, DataValidationError, GOLDEN_SCHEMA
from src.platform.runtime_loader import apply_pay_declarations, load_dataset, pay_basis_is_confirmed
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK, runtime_mutation
from src.platform.workspace import DatasetState, ModelState, WorkspaceStore

router = APIRouter(prefix="/api/upload", tags=["upload"])
_store = WorkspaceStore()


@router.get("/template")
async def download_template():
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "templates", "peopleos_template.csv"
    )
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template file not found")
    return FileResponse(template_path, media_type="text/csv", filename="peopleos_golden_schema_template.csv")


class UploadResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    success: bool
    message: str
    rows_loaded: int
    columns: list[str]
    features_enabled: Dict[str, bool]
    workspace_id: str = "local"
    dataset_id: Optional[str] = None
    dataset_version: Optional[int] = None
    deferred: Dict[str, bool] = Field(default_factory=dict)
    mapping_report: Optional[Dict[str, Any]] = None


class ColumnMappingPreview(BaseModel):
    source: str
    target: Optional[str] = None
    method: str
    confidence: float = Field(ge=0.0, le=1.0)
    required: bool = False
    status: str
    sample_values: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class LLMMappingPreview(BaseModel):
    requested: bool
    available: bool
    used: bool
    reason: Optional[str] = None
    data_scope: str = mapping_data_scope()


class UploadPreviewResponse(BaseModel):
    success: bool
    filename: str
    rows_detected: int
    source_columns: list[str]
    available_fields: list[str]
    mappings: list[ColumnMappingPreview]
    missing_required_fields: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    features_enabled: Dict[str, bool] = Field(default_factory=dict)
    can_activate: bool
    requires_review: bool
    llm: LLMMappingPreview


class DatabaseStatusResponse(BaseModel):
    has_data: bool
    employee_count: int
    features_enabled: Dict[str, bool]
    workspace_id: str = "local"
    active_dataset_id: Optional[str] = None
    dataset_version: Optional[int] = None
    source_name: Optional[str] = None
    reporting_currency: Optional[str] = None


def _parse_column_mapping(raw_mapping: Optional[str]) -> Optional[dict[str, Optional[str]]]:
    if raw_mapping is None or not raw_mapping.strip():
        return None
    try:
        payload = json.loads(raw_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError('The import mapping could not be read. Review the column choices and try again.') from exc
    if not isinstance(payload, dict):
        raise ValueError('The import mapping must be an object of source columns and PeopleOS fields.')
    parsed: dict[str, Optional[str]] = {}
    for source, target in payload.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError('Every mapped source column needs a name.')
        if target is not None and not isinstance(target, str):
            raise ValueError(f"The mapping for '{source}' is invalid.")
        parsed[source] = target.strip() if isinstance(target, str) and target.strip() else None
    return parsed


def _sample_values(frame: pd.DataFrame, source_column: Any) -> list[str]:
    values: list[str] = []
    try:
        series = frame[source_column]
    except (KeyError, IndexError):
        return values
    for value in series.head(3).tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            values.append(text[:60])
    return values


def _mapping_preview_rows(
    raw_frame: pd.DataFrame,
    loader: DataLoader,
    *,
    llm_reasons: Optional[dict[str, str]] = None,
) -> list[ColumnMappingPreview]:
    details = loader.mapping_details
    rows: list[ColumnMappingPreview] = []
    for source_column in raw_frame.columns:
        source = str(source_column)
        detail = details.get(source, {
            'source': source,
            'target': None,
            'method': 'unmapped',
            'confidence': 0.0,
            'required': False,
            'status': 'unmapped',
        })
        rows.append(ColumnMappingPreview(
            source=source,
            target=detail.get('target'),
            method=str(detail.get('method', 'unmapped')),
            confidence=float(detail.get('confidence', 0.0)),
            required=bool(detail.get('required', False)),
            status=str(detail.get('status', 'unmapped')),
            sample_values=_sample_values(raw_frame, source_column),
            reason=(llm_reasons or {}).get(source),
        ))
    return rows


def _run_column_mapping(raw_frame: pd.DataFrame, explicit_mapping: Optional[dict[str, Optional[str]]], mapping_methods: Optional[dict[str, str]]) -> tuple[DataLoader, Optional[pd.DataFrame], Optional[str]]:
    loader = DataLoader()
    try:
        mapped = loader._map_columns(
            raw_frame.copy(),
            explicit_mapping=explicit_mapping,
            mapping_methods=mapping_methods,
        )
    except Exception as exc:
        return loader, None, str(exc)
    return loader, mapped, None


def _build_upload_preview(
    path: str,
    *,
    salary_basis: Optional[str],
    salary_currency: Optional[str],
    explicit_mapping: Optional[dict[str, Optional[str]]],
    use_llm: bool,
    filename: str,
) -> UploadPreviewResponse:
    reader = DataLoader()
    raw_frame = reader._read_frame(path)
    source_columns = [str(column) for column in raw_frame.columns]

    base_mapping_loader, _, mapping_error = _run_column_mapping(raw_frame, explicit_mapping, None)
    effective_mapping = explicit_mapping
    effective_methods: Optional[dict[str, str]] = None
    llm_result: dict[str, Any] = {
        'available': False,
        'used': False,
        'reason': None,
    }

    # Local AI is an explicit assist after deterministic suggestions. It can fill
    # unresolved fields but never overrides a deterministic mapping silently.
    if explicit_mapping is None and mapping_error is None and use_llm:
        llm_result = suggest_column_mappings(raw_frame, existing_mapping=base_mapping_loader.column_mapping)
        suggestions = dict(llm_result.get('mappings') or {})
        if suggestions:
            effective_mapping = {**base_mapping_loader.column_mapping, **suggestions}
            effective_methods = {
                source: str(detail.get('method', 'user_confirmed'))
                for source, detail in base_mapping_loader.mapping_details.items()
                if detail.get('target')
            }
            effective_methods.update({source: 'llm' for source in suggestions})

    if mapping_error is not None:
        final_mapping_loader = base_mapping_loader
        validated_frame: Optional[pd.DataFrame] = None
        blocking = [mapping_error]
    else:
        final_mapping_loader, _, final_mapping_error = _run_column_mapping(raw_frame, effective_mapping, effective_methods)
        if final_mapping_error is not None:
            validated_frame = None
            blocking = [final_mapping_error]
        else:
            validation_loader = DataLoader()
            validated_frame = None
            blocking = []
            try:
                validated_frame = validation_loader.load(
                    path,
                    column_mapping=effective_mapping,
                    mapping_methods=effective_methods,
                )
                # Validation may apply a conservative mapping a second time when
                # no explicit mapping was needed; use its warnings and flags.
                final_mapping_loader = validation_loader
            except Exception as exc:
                blocking = [str(exc)]
                # Row-count and file-shape gates can fail before DataLoader gets
                # to mapping. Keep the mapping review visible in that case.
                if validation_loader.mapping_details:
                    final_mapping_loader = validation_loader
                else:
                    final_mapping_loader.validation_warnings.extend(base_mapping_loader.validation_warnings)
                    final_mapping_loader.features_enabled = dict(base_mapping_loader.features_enabled)

    llm_reasons = {
        str(item.get('source')): str(item.get('reason'))
        for item in (llm_result.get('details') or [])
        if item.get('source') and item.get('reason')
    }
    # Keep the suggested explanation visible even though the actual target is
    # validated by DataLoader before it appears in the review table.
    mappings = _mapping_preview_rows(raw_frame, final_mapping_loader, llm_reasons=llm_reasons)

    if validated_frame is not None:
        try:
            validated_frame = apply_pay_declarations(
                validated_frame,
                salary_basis=salary_basis,
                salary_currency=salary_currency,
            )
        except Exception as exc:
            blocking.append(str(exc))
            validated_frame = None

    mapped_fields = {
        str(item.target)
        for item in mappings
        if item.target
    }
    missing_required = [field for field in GOLDEN_SCHEMA['required'] if field not in mapped_fields]
    if missing_required and not any('missing' in issue.lower() for issue in blocking):
        blocking.append(
            'PeopleOS still needs these required fields: ' + ', '.join(missing_required)
        )

    warnings: list[str] = []
    for warning in final_mapping_loader.validation_warnings:
        if warning not in warnings:
            warnings.append(warning)

    features_enabled = dict(final_mapping_loader.features_enabled)
    features_enabled['compensation'] = bool(validated_frame is not None and pay_basis_is_confirmed(validated_frame))
    can_activate = validated_frame is not None and not blocking and not missing_required
    requires_review = bool(
        use_llm
        or any(item.status in {'needs_review', 'unmapped'} for item in mappings)
        or any(item.method == 'similarity' for item in mappings)
    )
    if use_llm and not llm_result.get('available') and not llm_result.get('reason'):
        llm_result['reason'] = 'Local AI is not ready; deterministic mapping is still available.'

    return UploadPreviewResponse(
        success=True,
        filename=filename,
        rows_detected=len(raw_frame),
        source_columns=source_columns,
        available_fields=GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional'],
        mappings=mappings,
        missing_required_fields=missing_required,
        blocking_issues=list(dict.fromkeys(blocking)),
        warnings=warnings,
        features_enabled=features_enabled,
        can_activate=can_activate,
        requires_review=requires_review,
        llm=LLMMappingPreview(
            requested=use_llm,
            available=bool(llm_result.get('available')),
            used=bool(llm_result.get('used')),
            reason=llm_result.get('reason'),
        ),
    )


@runtime_mutation
def _register_loaded_dataset(state: AppState, source_name: str, content_hash: str, workspace_id: str = "local"):
    source_frame = state.historical_df if state.historical_df is not None else state.raw_df
    if source_frame is None or source_frame.empty:
        raise ValueError("Loaded dataset has no rows to persist")
    dataset_id = f"ds_{uuid4().hex}"
    path = save_dataset_artifact(dataset_id, source_frame)
    try:
        activated = _store.register_active_dataset(
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            source_name=source_name,
            content_hash=content_hash,
            row_count=len(state.raw_df),
            columns=list(state.raw_df.columns),
            quality={
                "missing_cells": int(state.raw_df.isna().sum().sum()),
                "duplicate_rows": int(state.raw_df.duplicated().sum()),
                'artifact_sha256': _store.hash_bytes(path.read_bytes()),
                'current_fingerprint': state.runtime_provenance['current_fingerprint'],
            },
        )
    except Exception:
        remove_dataset_artifact(dataset_id)
        raise
    state.runtime_provenance = {**state.runtime_provenance, 'workspace_id': workspace_id, 'dataset_id': dataset_id, 'dataset_version': activated.version, 'source_name': source_name}
    return activated


def _clear_active_lifecycle(workspace_id: str = "local") -> None:
    """Clear active selections without deleting version/audit history."""
    workspace = _store.ensure_workspace(workspace_id, "Local workspace")
    for dataset in workspace.datasets:
        if dataset.dataset_id == workspace.active_dataset_id and dataset.state == DatasetState.ACTIVE:
            dataset.state = DatasetState.SUPERSEDED
    for model in workspace.models:
        if model.model_id == workspace.active_model_id and model.state == ModelState.ACTIVE:
            model.state = ModelState.RETIRED
    workspace.active_dataset_id = None
    workspace.active_model_id = None
    _store._replace_workspace(workspace)


def _prepare_upload_file(content: bytes, ext: str) -> tuple[str, list[str]]:
    """Return a DataLoader-compatible temporary file and every path to clean up.

    Modern Excel workbooks are a user-facing convenience. They are converted to a
    temporary CSV while the original workbook bytes remain the provenance/content
    hash. Only the first worksheet is imported and empty workbooks fail closed.
    """
    cleanup: list[str] = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as raw:
        raw.write(content)
        raw_path = raw.name
    cleanup.append(raw_path)
    if ext != "xlsx":
        return raw_path, cleanup

    try:
        frame = pd.read_excel(raw_path, sheet_name=0, dtype=str, engine="openpyxl")
    except Exception as exc:
        raise ValueError("PeopleOS could not read this Excel workbook. Use a standard .xlsx file with employee data on the first worksheet.") from exc
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The first Excel worksheet is empty. Put the workforce table on the first worksheet and try again.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8", newline="") as converted:
        frame.to_csv(converted, index=False)
        csv_path = converted.name
    cleanup.append(csv_path)
    return csv_path, cleanup


@router.post("/preview", response_model=UploadPreviewResponse)
async def preview_upload(
    file: UploadFile = File(...),
    salary_basis: Optional[str] = Form(None),
    salary_currency: Optional[str] = Form(None),
    column_mapping: Optional[str] = Form(None),
    use_llm: bool = Form(False),
) -> UploadPreviewResponse:
    """Prepare a non-mutating import review before any dataset is activated."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"csv", "json", "xlsx"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use Excel (.xlsx), CSV or JSON.")
    try:
        parsed_mapping = _parse_column_mapping(column_mapping)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cleanup_paths: list[str] = []
    try:
        content = await file.read()
        tmp_path, cleanup_paths = _prepare_upload_file(content, ext)
        return await asyncio.to_thread(
            _build_upload_preview,
            tmp_path,
            salary_basis=salary_basis,
            salary_currency=salary_currency,
            explicit_mapping=parsed_mapping,
            use_llm=use_llm,
            filename=file.filename,
        )
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for path in cleanup_paths:
            if os.path.exists(path):
                os.unlink(path)


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), state: AppState = Depends(get_app_state),
                      salary_basis: Optional[str] = Form(None), salary_currency: Optional[str] = Form(None),
                      column_mapping: Optional[str] = Form(None)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"csv", "json", "xlsx"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use Excel (.xlsx), CSV or JSON.")
    try:
        parsed_mapping = _parse_column_mapping(column_mapping)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cleanup_paths: list[str] = []
    try:
        content = await file.read()
        with RUNTIME_MUTATION_LOCK:
            tmp_path, cleanup_paths = _prepare_upload_file(content, ext)
            candidate = SimpleNamespace(**state.__dict__)
            result = load_dataset(
                candidate,
                tmp_path,
                file.filename,
                salary_basis=salary_basis,
                salary_currency=salary_currency,
                column_mapping=parsed_mapping,
            )
            dataset = _register_loaded_dataset(candidate, file.filename, _store.hash_bytes(content))
            state.__dict__.update(candidate.__dict__)
            return UploadResponse(
                success=True,
                message=f"Your workforce is ready. {result['rows_loaded']} employee records were activated as dataset v{dataset.version}. " + candidate.runtime_provenance['pay_basis_message'],
                rows_loaded=result['rows_loaded'],
                columns=result['columns'],
                features_enabled=result['features_enabled'],
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.version,
                deferred=result.get('deferred', {}),
                mapping_report=result.get('report'),
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        for path in cleanup_paths:
            if os.path.exists(path):
                os.unlink(path)


@router.get("/status", response_model=DatabaseStatusResponse)
async def get_database_status(state: AppState = Depends(get_app_state)) -> DatabaseStatusResponse:
    if not state.has_data():
        loaded = state.load_from_database()
        if not loaded:
            workspace = _store.ensure_workspace("local", "Local workspace")
            return DatabaseStatusResponse(
                has_data=False,
                employee_count=0,
                features_enabled=state.features_enabled,
                active_dataset_id=workspace.active_dataset_id,
            )
    workspace = _store.ensure_workspace("local", "Local workspace")
    active_dataset = next(
        (
            dataset
            for dataset in workspace.datasets
            if dataset.dataset_id == workspace.active_dataset_id
            and dataset.state == DatasetState.ACTIVE
        ),
        None,
    )
    provenance = state.runtime_provenance or {}
    return DatabaseStatusResponse(
        has_data=True,
        employee_count=len(state.raw_df) if state.raw_df is not None else 0,
        features_enabled=state.features_enabled,
        active_dataset_id=workspace.active_dataset_id,
        dataset_version=active_dataset.version if active_dataset else provenance.get('dataset_version'),
        source_name=active_dataset.source_name if active_dataset else provenance.get('source_name'),
        reporting_currency=provenance.get('reporting_currency'),
    )


@router.post("/load-sample", response_model=UploadResponse)
async def load_sample_data(state: AppState = Depends(get_app_state)) -> UploadResponse:
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "sample_hr_data.csv"
    )
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample data file not found")
    try:
        with RUNTIME_MUTATION_LOCK:
            content = open(sample_path, "rb").read()
            candidate = SimpleNamespace(**state.__dict__)
            result = load_dataset(candidate, sample_path, "sample_hr_data.csv", salary_basis='annual', salary_currency='USD')
            dataset = _register_loaded_dataset(candidate, "sample_hr_data.csv", _store.hash_bytes(content))
            state.__dict__.update(candidate.__dict__)
            return UploadResponse(
                success=True,
                message=f"Your sample workforce is ready. {result['rows_loaded']} fictional employee records were activated as dataset v{dataset.version}.",
                rows_loaded=result['rows_loaded'],
                columns=result['columns'],
                features_enabled=result['features_enabled'],
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.version,
                deferred=result.get('deferred', {}),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/reset")
async def reset_data(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    with RUNTIME_MUTATION_LOCK:
        previous = _store.get_workspace('local')
        _clear_active_lifecycle('local')
        try:
            state.reset()
        except Exception:
            _store._replace_workspace(previous)
            raise
    return {
        "success": True,
        "message": "The active workforce was cleared. Previous dataset and model versions remain in local history.",
    }
