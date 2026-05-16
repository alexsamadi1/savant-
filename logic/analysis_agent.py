"""Core analysis agent: 4-tool GPT loop that produces a DashboardConfig."""

import json
import os
import re
import sqlite3
import traceback
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI

from api.models import ChartSpec, DashboardConfig, DiscoveryRequest
from logic.data_loader import download_sqlite_from_s3, execute_sql_on_tenant_data
from logic.doc_ingestor import load_faiss_index
from tools.s3_utils import get_secret

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run a SELECT query against the tenant's SQLite database containing uploaded CSV/Excel data and a 'documents' table with document metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SQLite SELECT query.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief explanation of what this query is checking.",
                    },
                },
                "required": ["query", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Semantic search over uploaded PDF/DOCX documents. Use this to find specific passages, quotes, or qualitative evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute Python code for calculations using pandas and numpy. A dict 'df_dict' mapping table names to DataFrames is available. Assign your result to 'output'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must assign result to the variable 'output'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief explanation of what this code computes.",
                    },
                },
                "required": ["code", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart_spec",
            "description": "Record a chart specification for the dashboard. Call this for every key finding that should be visualised.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Chart title."},
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "pie", "scatter", "area"],
                        "description": "Chart type.",
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Array of data point objects for Recharts.",
                    },
                    "x_key": {"type": "string", "description": "Key for x-axis."},
                    "y_key": {"type": "string", "description": "Key for y-axis."},
                    "insight": {
                        "type": "string",
                        "description": "One-sentence insight this chart reveals.",
                    },
                },
                "required": ["title", "chart_type", "data", "x_key", "y_key", "insight"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution handlers
# ---------------------------------------------------------------------------


def _handle_sql_query(args: Dict, tenant: str) -> str:
    """Execute a SQL SELECT and return results as JSON."""
    result = execute_sql_on_tenant_data(args["query"], tenant)
    return json.dumps(result, default=str)


def _handle_rag_search(args: Dict, tenant: str) -> str:
    """Run similarity search against tenant's FAISS index."""
    index = load_faiss_index(tenant)
    if index is None:
        return json.dumps({"message": "No documents uploaded"})
    docs = index.similarity_search(args["query"], k=3)
    results = [
        {"content": doc.page_content, "source": doc.metadata.get("filename", "unknown")}
        for doc in docs
    ]
    return json.dumps(results, default=str)


def _handle_python_exec(args: Dict, df_dict: Dict[str, pd.DataFrame]) -> str:
    """Execute Python code in a sandboxed namespace."""
    local_vars: Dict = {
        "df_dict": df_dict,
        "pd": pd,
        "np": np,
        "output": {},
    }
    try:
        exec(args["code"], {"__builtins__": __builtins__, "pd": pd, "np": np}, local_vars)
        return json.dumps(local_vars["output"], default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def _handle_generate_chart_spec(args: Dict, charts: List[ChartSpec], chart_counter: int) -> str:
    """Record a chart spec and return confirmation."""
    try:
        # Handle data as string (GPT sometimes serializes it)
        data = args.get("data", [])
        if isinstance(data, str):
            data = json.loads(data)

        spec = ChartSpec(
            id=f"chart_{chart_counter}",
            title=args.get("title", f"Chart {chart_counter}"),
            type=args.get("chart_type", "bar"),
            data=data,
            x_key=args.get("x_key", "x"),
            y_key=args.get("y_key", "y"),
            color=args.get("color"),
            insight=args.get("insight", ""),
        )
        charts.append(spec)
        return json.dumps({"status": "chart_spec_recorded"})
    except Exception as e:
        return json.dumps({"status": "chart_spec_recorded", "warning": str(e)})


# ---------------------------------------------------------------------------
# DataFrame loader
# ---------------------------------------------------------------------------


def _load_tenant_dataframes(tenant: str) -> Dict[str, pd.DataFrame]:
    """Download tenant SQLite and load every table into a DataFrame dict."""
    db_path = download_sqlite_from_s3(tenant)
    if db_path is None:
        return {}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        df_dict: Dict[str, pd.DataFrame] = {}
        for table in tables:
            df_dict[table] = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        conn.close()
        return df_dict
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 12


def run_analysis_agent(
    discovery: DiscoveryRequest,
    schemas: List[Dict],
    tenant: str,
    model: str = "gpt-4o",
) -> DashboardConfig:
    """Run the analysis agent loop and return a complete DashboardConfig.

    Args:
        discovery: Client discovery form data.
        schemas: Table schema descriptions from get_schema_description().
        tenant: Tenant identifier.
        model: OpenAI model to use.

    Returns:
        A fully populated DashboardConfig.
    """
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    # Load tenant data for python_exec tool
    df_dict = _load_tenant_dataframes(tenant)

    # Trim schema data to reduce prompt size
    for schema in schemas:
        for col in schema.get("columns", []):
            col["sample_values"] = col["sample_values"][:1]
            col.pop("min", None)
            col.pop("max", None)
            col.pop("mean", None)

    # Build system prompt
    questions_block = "\n".join(f"  - {q}" for q in discovery.key_questions)
    schemas_block = json.dumps(schemas, indent=2, default=str)

    system_prompt = (
        "You are Savant, an expert data analyst and management consultant.\n\n"
        f"Company: {discovery.company_name}\n"
        f"Industry: {discovery.industry}\n"
        f"Problem statement: {discovery.problem_statement}\n"
        f"Key questions:\n{questions_block}\n\n"
        f"Available data schemas:\n{schemas_block}\n\n"
        "Instructions:\n"
        "1. Explore the data using sql_query to understand table structure and row counts before any analysis.\n"
        "2. For any rate or percentage calculation, always verify by running a second sql_query that cross-checks the result from a different angle.\n"
        "3. When calculating group attrition rates, always use: COUNT(attrited in group) / COUNT(total in group). Never invert this.\n"
        "4. Before calling generate_chart_spec, verify the data array is populated with actual query results — never call generate_chart_spec with empty data.\n"
        "5. Use python_exec for any correlation or statistical calculation — never estimate correlations manually.\n"
        "6. For every key finding, call generate_chart_spec with the actual data rows from your sql_query results populated in the data array.\n"
        "7. When your analysis is complete, send a FINAL message (no tool calls) containing a JSON object with exactly these keys: executive_summary, metrics, recommendations, data_sources_used."
    )

    messages: List[Dict] = [{"role": "system", "content": system_prompt}]
    charts: List[ChartSpec] = []
    chart_counter = 0

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # Append the assistant message to history
        messages.append(assistant_msg.model_dump())

        # If no tool calls, this is the final response
        if not assistant_msg.tool_calls:
            break

        # Process each tool call
        for tool_call in assistant_msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            try:
                if name == "sql_query":
                    result = _handle_sql_query(args, tenant)
                elif name == "rag_search":
                    result = _handle_rag_search(args, tenant)
                elif name == "python_exec":
                    result = _handle_python_exec(args, df_dict)
                elif name == "generate_chart_spec":
                    chart_counter += 1
                    result = _handle_generate_chart_spec(args, charts, chart_counter)
                else:
                    result = json.dumps({"error": f"Unknown tool: {name}"})
            except Exception as e:
                result = json.dumps({"error": str(e)})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Parse the final message into DashboardConfig
    raw_content = assistant_msg.content or ""

    if not raw_content.strip():
        # Agent hit iteration limit without producing final JSON
        # Build a minimal dashboard from whatever charts were collected
        raw_content = json.dumps({
            "executive_summary": "Analysis completed. See metrics and charts for findings.",
            "metrics": [],
            "recommendations": [],
            "data_sources_used": [s.get("table_name", "unknown") for s in schemas]
        })

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)

    # Coerce metric values to strings (GPT sometimes returns dicts/numbers)
    metrics = parsed.get("metrics", [])
    cleaned_metrics = []
    for m in metrics:
        if isinstance(m, dict):
            if not isinstance(m.get("value"), str):
                m["value"] = str(m["value"]) if m.get("value") is not None else ""
            if "id" not in m:
                m["id"] = f"metric_{len(cleaned_metrics)}"
            cleaned_metrics.append(m)
    parsed["metrics"] = cleaned_metrics

    # Ensure recommendation priorities are ints
    recs = parsed.get("recommendations", [])
    for r in recs:
        if isinstance(r.get("priority"), str):
            try:
                r["priority"] = int(r["priority"])
            except ValueError:
                r["priority"] = 2

    return DashboardConfig(
        tenant=tenant,
        company_name=discovery.company_name,
        problem_statement=discovery.problem_statement,
        generated_at=datetime.now().isoformat(),
        executive_summary=parsed.get("executive_summary", ""),
        metrics=metrics,
        charts=[c.model_dump() for c in charts],
        recommendations=recs,
        data_sources_used=parsed.get("data_sources_used", []),
    )
