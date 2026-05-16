"""Analysis routes: trigger agent and poll status."""

import json
import traceback

from fastapi import APIRouter, BackgroundTasks

from api.models import AnalysisRequest, AnalysisStatus
from logic.analysis_agent import run_analysis_agent
from tools.s3_utils import get_s3_client, get_secret

router = APIRouter()


def _write_status(tenant: str, status: str, progress: str = None, error: str = None) -> None:
    """Write analysis_status.json to S3."""
    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")
    payload = {"tenant": tenant, "status": status, "progress": progress, "error": error}
    s3.put_object(
        Bucket=bucket,
        Key=f"{tenant}/analysis_status.json",
        Body=json.dumps(payload),
        ContentType="application/json",
    )


def _run_analysis_background(request: AnalysisRequest) -> None:
    """Background task: run agent, save dashboard, update status."""
    tenant = request.tenant
    try:
        _write_status(tenant, "running", progress="Loading your data...")

        s3 = get_s3_client()
        bucket = get_secret("S3_DOCS_BUCKET")
        resp = s3.get_object(Bucket=bucket, Key=f"{tenant}/schema.json")
        schemas = json.loads(resp["Body"].read())

        _write_status(tenant, "running", progress="Exploring data structure...")

        # Monkey-patch the agent to emit progress updates
        import logic.analysis_agent as agent_module
        original_handle_sql = agent_module._handle_sql_query
        original_handle_python = agent_module._handle_python_exec
        original_handle_chart = agent_module._handle_generate_chart_spec

        sql_count = [0]
        chart_count = [0]

        def tracked_sql(args, t):
            sql_count[0] += 1
            _write_status(tenant, "running", progress=f"Running analysis query {sql_count[0]}...")
            return original_handle_sql(args, t)

        def tracked_python(args, df_dict):
            _write_status(tenant, "running", progress="Computing statistics...")
            return original_handle_python(args, df_dict)

        def tracked_chart(args, charts, counter):
            chart_count[0] += 1
            _write_status(tenant, "running", progress=f"Building chart {chart_count[0]}...")
            return original_handle_chart(args, charts, counter)

        agent_module._handle_sql_query = tracked_sql
        agent_module._handle_python_exec = tracked_python
        agent_module._handle_generate_chart_spec = tracked_chart

        _write_status(tenant, "running", progress="AI agent analyzing your data...")

        dashboard = run_analysis_agent(
            discovery=request.discovery,
            schemas=schemas,
            tenant=tenant,
            model=request.model,
        )

        # Restore originals
        agent_module._handle_sql_query = original_handle_sql
        agent_module._handle_python_exec = original_handle_python
        agent_module._handle_generate_chart_spec = original_handle_chart

        _write_status(tenant, "running", progress="Saving dashboard...")

        s3.put_object(
            Bucket=bucket,
            Key=f"{tenant}/dashboard.json",
            Body=dashboard.model_dump_json(),
            ContentType="application/json",
        )

        _write_status(tenant, "complete")

    except Exception as e:
        _write_status(tenant, "error", error=f"{e}\n{traceback.format_exc()}")


@router.post("/analyze", response_model=AnalysisStatus)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """Kick off the analysis agent as a background task."""
    _write_status(request.tenant, "pending")
    background_tasks.add_task(_run_analysis_background, request)
    return AnalysisStatus(tenant=request.tenant, status="pending")


@router.get("/analyze/status/{tenant}", response_model=AnalysisStatus)
async def get_analysis_status(tenant: str):
    """Poll the current analysis status."""
    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{tenant}/analysis_status.json")
        data = json.loads(resp["Body"].read())
        return AnalysisStatus(**data)
    except Exception:
        return AnalysisStatus(tenant=tenant, status="pending", progress="No analysis started")
