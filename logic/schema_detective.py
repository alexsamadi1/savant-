"""GPT-powered schema analysis: suggest metrics, queries, and flag data quality issues."""

import json
from typing import Dict, List

from openai import OpenAI

from tools.s3_utils import get_secret


def detect_schema_and_suggest_questions(
    schemas: List[Dict], problem_statement: str
) -> Dict:
    """Analyze table schemas and return suggested metrics, queries, and data quality flags.

    Args:
        schemas: List of schema dicts from get_schema_description().
        problem_statement: The client's stated problem or objective.

    Returns:
        Dict with keys: suggested_metrics, suggested_queries, data_quality_flags.
    """
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    system_prompt = (
        "You are a data analyst. Given table schemas and a business problem, "
        "return a JSON object with exactly three keys:\n"
        "- \"suggested_metrics\": list of metric strings the data could answer\n"
        "- \"suggested_queries\": list of SQL SELECT queries (SQLite syntax) to explore the data\n"
        "- \"data_quality_flags\": list of potential data quality issues spotted in the schema\n"
        "Return ONLY valid JSON. No markdown fences."
    )

    user_prompt = (
        f"Problem statement: {problem_statement}\n\n"
        f"Table schemas:\n{json.dumps(schemas, indent=2, default=str)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    return json.loads(response.choices[0].message.content)
