"""
Tripletex AI Accounting Agent — NM i AI 2026
Uses Gemini to interpret Norwegian accounting prompts and execute Tripletex API calls.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path

import requests
import vertexai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vertexai.generative_models import GenerativeModel

# --- Config ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "ainm26osl-705")
REGION = os.environ.get("GCP_REGION", "europe-north1")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# --- Init ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel(MODEL_NAME)


SYSTEM_PROMPT = """You are an AI accounting agent for Tripletex (Norwegian accounting software).
You receive a task prompt in Norwegian (or other languages) and must execute it using the Tripletex REST API.

You have access to these Tripletex API endpoints (all via base_url):
- GET/POST/PUT /employee — manage employees
- GET/POST/PUT /customer — manage customers
- GET/POST /product — manage products
- GET/POST /invoice — create and query invoices
- GET/POST /order — manage orders (needed before invoices)
- GET/POST/PUT/DELETE /travelExpense — travel expense reports
- GET/POST /project — manage projects
- GET/POST /department — manage departments
- GET /ledger/account — query chart of accounts
- GET/POST/DELETE /ledger/voucher — manage vouchers

Authentication: Basic Auth with username "0" and the session_token as password.

IMPORTANT RULES:
- Analyze the prompt carefully. Extract entity names, values, and relationships.
- Return a JSON array of API calls to execute IN ORDER.
- For POST/PUT, include the JSON body.
- Use ?fields=* on GET requests to see all available fields.
- When creating invoices, you usually need to create an order first.
- Some fields are required — check error messages and retry if needed.
- Keep it minimal — fewer API calls = better efficiency score.

Respond ONLY with a valid JSON array. Each element:
{
  "method": "GET" | "POST" | "PUT" | "DELETE",
  "path": "/endpoint/path",
  "params": {"key": "value"},  // for GET query params
  "body": {}  // for POST/PUT
}

If you need to reference an ID from a previous call's response, use the placeholder "$PREV_N_ID" where N is the 0-based index of the previous call. For example "$PREV_0_ID" means the id from the first call's response.

Example — "Opprett en ansatt med navn Ola Nordmann, e-post ola@example.org":
[
  {
    "method": "POST",
    "path": "/employee",
    "body": {
      "firstName": "Ola",
      "lastName": "Nordmann",
      "email": "ola@example.org"
    }
  }
]

Example — "Opprett en kunde Test AS og lag en faktura":
[
  {
    "method": "POST",
    "path": "/customer",
    "body": {"name": "Test AS", "isCustomer": true}
  },
  {
    "method": "POST",
    "path": "/order",
    "body": {"customer": {"id": "$PREV_0_ID"}, "deliveryDate": "2026-03-20", "orderDate": "2026-03-20"}
  },
  {
    "method": "POST",
    "path": "/invoice",
    "body": {"invoiceDate": "2026-03-20", "invoiceDueDate": "2026-04-20", "orders": [{"id": "$PREV_1_ID"}]}
  }
]
"""


def resolve_placeholders(value, results):
    """Replace $PREV_N_ID placeholders with actual IDs from previous results."""
    if isinstance(value, str):
        pattern = r'\$PREV_(\d+)_ID'
        match = re.search(pattern, value)
        if match:
            idx = int(match.group(1))
            if idx < len(results) and results[idx]:
                prev_id = None
                if "value" in results[idx]:
                    prev_id = results[idx]["value"].get("id")
                elif "values" in results[idx] and results[idx]["values"]:
                    prev_id = results[idx]["values"][0].get("id")
                if prev_id is not None:
                    # If the entire string is just the placeholder, return the int
                    if value == match.group(0):
                        return prev_id
                    return value.replace(match.group(0), str(prev_id))
        return value
    elif isinstance(value, dict):
        return {k: resolve_placeholders(v, results) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_placeholders(v, results) for v in value]
    return value


def execute_api_calls(calls, base_url, session_token):
    """Execute a sequence of Tripletex API calls."""
    auth = ("0", session_token)
    results = []

    for i, call in enumerate(calls):
        method = call.get("method", "GET").upper()
        path = call.get("path", "")
        params = call.get("params", {})
        body = call.get("body", {})

        # Resolve placeholders from previous results
        path = resolve_placeholders(path, results)
        params = resolve_placeholders(params, results)
        body = resolve_placeholders(body, results)

        url = f"{base_url}{path}"
        logger.info(f"Call {i}: {method} {url}")

        try:
            if method == "GET":
                resp = requests.get(url, auth=auth, params=params, timeout=30)
            elif method == "POST":
                resp = requests.post(url, auth=auth, json=body, timeout=30)
            elif method == "PUT":
                resp = requests.put(url, auth=auth, json=body, timeout=30)
            elif method == "DELETE":
                resp = requests.delete(url, auth=auth, timeout=30)
            else:
                logger.warning(f"Unknown method: {method}")
                results.append(None)
                continue

            logger.info(f"  → {resp.status_code}")

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                logger.warning(f"  → Error: {error_text}")
                results.append({"error": resp.status_code, "detail": error_text})

                # If it failed, try to fix with Gemini
                if resp.status_code in (400, 422):
                    fix = try_fix_call(call, error_text, base_url, auth, results)
                    if fix:
                        results[-1] = fix
            else:
                try:
                    results.append(resp.json())
                except Exception:
                    results.append({"status_code": resp.status_code})

        except Exception as e:
            logger.error(f"  → Exception: {e}")
            results.append(None)

    return results


def try_fix_call(original_call, error_text, base_url, auth, results):
    """Ask Gemini to fix a failed API call based on the error message."""
    fix_prompt = f"""The following Tripletex API call failed:
{json.dumps(original_call, indent=2)}

Error response: {error_text}

Fix the API call and return ONLY a corrected JSON object (single call, not array).
Keep the same intent but fix the error. If fields are missing, add them with reasonable defaults.
"""
    try:
        response = model.generate_content(fix_prompt)
        text = response.text.strip()
        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        fixed_call = json.loads(text)
        method = fixed_call.get("method", "POST").upper()
        path = fixed_call.get("path", original_call.get("path", ""))
        body = resolve_placeholders(fixed_call.get("body", {}), results)
        url = f"{base_url}{path}"

        logger.info(f"  → Retry: {method} {url}")
        if method == "POST":
            resp = requests.post(url, auth=auth, json=body, timeout=30)
        elif method == "PUT":
            resp = requests.put(url, auth=auth, json=body, timeout=30)
        else:
            return None

        logger.info(f"  → Retry status: {resp.status_code}")
        if resp.status_code < 400:
            return resp.json()
    except Exception as e:
        logger.error(f"  → Fix failed: {e}")

    return None


def extract_file_content(files):
    """Extract text content from attached files."""
    file_descriptions = []
    for f in files:
        filename = f.get("filename", "unknown")
        mime = f.get("mime_type", "")
        try:
            data = base64.b64decode(f["content_base64"])
            if "pdf" in mime:
                file_descriptions.append(f"[Attached PDF: {filename}, {len(data)} bytes — extract data from this if needed]")
            elif "image" in mime:
                file_descriptions.append(f"[Attached image: {filename}, {len(data)} bytes]")
            else:
                try:
                    text = data.decode("utf-8")[:2000]
                    file_descriptions.append(f"[File: {filename}]\n{text}")
                except Exception:
                    file_descriptions.append(f"[Binary file: {filename}, {len(data)} bytes]")
        except Exception:
            file_descriptions.append(f"[Could not decode: {filename}]")
    return "\n".join(file_descriptions)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]
    base_url = creds["base_url"]
    session_token = creds["session_token"]

    logger.info(f"Received task: {prompt[:200]}...")

    # Build the full prompt for Gemini
    user_prompt = f"Task prompt:\n{prompt}"

    if files:
        file_info = extract_file_content(files)
        user_prompt += f"\n\nAttached files:\n{file_info}"

    # Ask Gemini to plan the API calls
    try:
        response = model.generate_content(
            [SYSTEM_PROMPT, user_prompt],
            generation_config={"temperature": 0.1, "max_output_tokens": 4096},
        )
        text = response.text.strip()

        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        calls = json.loads(text)
        if not isinstance(calls, list):
            calls = [calls]

        logger.info(f"Gemini planned {len(calls)} API calls")

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return JSONResponse({"status": "completed"})

    # Execute the API calls
    try:
        results = execute_api_calls(calls, base_url, session_token)
        logger.info(f"Executed {len(results)} calls")
    except Exception as e:
        logger.error(f"Execution error: {e}")

    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
