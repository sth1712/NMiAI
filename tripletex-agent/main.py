"""
Tripletex AI Accounting Agent — NM i AI 2026
Uses Gemini to interpret Norwegian accounting prompts and execute Tripletex API calls.
"""

import base64
import json
import logging
import os
import re

import requests as http_requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Init ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
client = None

@app.on_event("startup")
def startup():
    global client
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"Gemini client initialized (key: {GEMINI_API_KEY[:8]}...)")
    else:
        logger.warning("GEMINI_API_KEY not set — agent will return completed without action")


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

CRITICAL FIELD REQUIREMENTS (learned from testing):

EMPLOYEE (POST /employee):
- REQUIRED: firstName, lastName, email, userType, department.id
- department.id: First GET /department to find available departments, use the first one's id
- userType options:
  - "STANDARD" — regular user (cannot be given admin role)
  - "EXTENDED" — extended access user (CAN be given admin role)
  - "NO_ACCESS" — no login access
- If the prompt says "kontoadministrator", "administrator", "admin" or similar → use userType "EXTENDED"
- After creating an EXTENDED employee who should be admin, you MUST also call:
  1. GET /token/session/>whoAmI to get companyId
  2. POST /employee/entitlement with body: {"employee": {"id": NEW_EMPLOYEE_ID}, "entitlementId": 1, "customer": {"id": COMPANY_ID}}
  This assigns the ROLE_ADMINISTRATOR entitlement.
- If NO admin role is mentioned → use userType "STANDARD"

CUSTOMER (POST /customer):
- REQUIRED: name, isCustomer (set to true)
- Optional: email, organizationNumber, phoneNumber

PRODUCT (POST /product):
- REQUIRED: name
- Optional: number, priceExcludingVatCurrency, priceIncludingVatCurrency, costExcludingVatCurrency
- NOTE: Field names end with "Currency" — NOT "priceExcludingVat" but "priceExcludingVatCurrency"
- For VAT/MVA: use vatType with id. Standard 25% MVA = {"vatType": {"id": 3}}
- vatType IDs: 3 = 25% standard, 5 = 15% food, 31 = 12% transport, 6 = 0% exempt

ORDER (POST /order):
- REQUIRED: customer.id, deliveryDate, orderDate
- Get customer.id from a previous POST /customer or GET /customer

INVOICE (POST /invoice):
- REQUIRED: invoiceDate, invoiceDueDate, orders (array of {id})
- MUST create order first, then reference order id
- Flow: POST /customer → POST /order (with customer.id) → POST /invoice (with order.id)

TRAVEL EXPENSE (POST /travelExpense):
- REQUIRED: employee.id, title, startDate, endDate

PROJECT (POST /project):
- REQUIRED: name, number, projectManager.id (employee id)
- Optional: customer.id
- IMPORTANT: The employee used as projectManager MUST have project manager entitlement first!
  Before creating a project, give the employee project manager access:
  1. GET /token/session/>whoAmI to get companyId
  2. POST /employee/entitlement with body: {"employee": {"id": EMPLOYEE_ID}, "entitlementId": 10, "customer": {"id": COMPANY_ID}}
  This assigns AUTH_PROJECT_MANAGER. Without this, the project creation will fail with "prosjektleder har ikke fått tilgang".

DEPARTMENT (POST /department):
- REQUIRED: name
- Optional: departmentNumber

GENERAL RULES:
- Analyze the prompt carefully. Extract entity names, values, and relationships.
- Return a JSON array of API calls to execute IN ORDER.
- For POST/PUT, include the JSON body.
- Keep it minimal — fewer API calls = better efficiency score.
- For dates, use format "YYYY-MM-DD" and default to "2026-03-20" if not specified.
- When you need an existing entity's ID (like department for employee), GET it first.
- Prompts come in 7 languages (Norwegian, English, Spanish, Portuguese, Nynorsk, German, French).

Respond ONLY with a valid JSON array. No explanation, no markdown, just the JSON array.

Each element:
{
  "method": "GET" | "POST" | "PUT" | "DELETE",
  "path": "/endpoint/path",
  "params": {},
  "body": {}
}

If you need to reference an ID from a previous call's response, use "$PREV_N_ID" where N is the 0-based index. Example: "$PREV_0_ID" = id from first call's response.
For GET results with multiple values, "$PREV_N_ID" returns the first result's id.

Example — "Opprett en ansatt med navn Ola Nordmann, e-post ola@example.org":
[
  {
    "method": "GET",
    "path": "/department",
    "params": {"count": 1}
  },
  {
    "method": "POST",
    "path": "/employee",
    "body": {
      "firstName": "Ola",
      "lastName": "Nordmann",
      "email": "ola@example.org",
      "userType": "STANDARD",
      "department": {"id": "$PREV_0_ID"}
    }
  }
]

Example — "Opprett en ansatt med navn Kari Nordmann, kari@example.org. Hun skal være kontoadministrator.":
[
  {
    "method": "GET",
    "path": "/department",
    "params": {"count": 1}
  },
  {
    "method": "GET",
    "path": "/token/session/>whoAmI",
    "params": {}
  },
  {
    "method": "POST",
    "path": "/employee",
    "body": {
      "firstName": "Kari",
      "lastName": "Nordmann",
      "email": "kari@example.org",
      "userType": "EXTENDED",
      "department": {"id": "$PREV_0_ID"}
    }
  },
  {
    "method": "POST",
    "path": "/employee/entitlement",
    "body": {
      "employee": {"id": "$PREV_2_ID"},
      "entitlementId": 1,
      "customer": {"id": "$PREV_1_ID"}
    }
  }
]

Example — "Opprett en kunde Test AS med e-post test@test.no":
[
  {
    "method": "POST",
    "path": "/customer",
    "body": {
      "name": "Test AS",
      "email": "test@test.no",
      "isCustomer": true
    }
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
                    # Try id first, then companyId (for /token/session/>whoAmI)
                    prev_id = results[idx]["value"].get("id") or results[idx]["value"].get("companyId")
                elif "values" in results[idx] and results[idx]["values"]:
                    prev_id = results[idx]["values"][0].get("id")
                if prev_id is not None:
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

        path = resolve_placeholders(path, results)
        params = resolve_placeholders(params, results)
        body = resolve_placeholders(body, results)

        url = f"{base_url}{path}"
        logger.info(f"Call {i}: {method} {url}")
        if body:
            logger.info(f"  Body: {json.dumps(body)[:300]}")

        try:
            if method == "GET":
                resp = http_requests.get(url, auth=auth, params=params, timeout=30)
            elif method == "POST":
                resp = http_requests.post(url, auth=auth, json=body, timeout=30)
            elif method == "PUT":
                resp = http_requests.put(url, auth=auth, json=body, timeout=30)
            elif method == "DELETE":
                resp = http_requests.delete(url, auth=auth, timeout=30)
            else:
                results.append(None)
                continue

            logger.info(f"  → {resp.status_code}")

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                logger.warning(f"  → Error: {error_text}")
                results.append({"error": resp.status_code, "detail": error_text})

                # Try to fix with Gemini on validation errors
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

Fix the API call. Return ONLY a corrected JSON object (single call, not array). No explanation."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=fix_prompt,
            config={"temperature": 0.1, "max_output_tokens": 2048},
        )
        text = response.text.strip()
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
            resp = http_requests.post(url, auth=auth, json=body, timeout=30)
        elif method == "PUT":
            resp = http_requests.put(url, auth=auth, json=body, timeout=30)
        else:
            return None

        logger.info(f"  → Retry: {resp.status_code}")
        if resp.status_code < 400:
            return resp.json()
    except Exception as e:
        logger.error(f"  → Fix failed: {e}")
    return None


def extract_file_content(files):
    """Extract text content from attached files."""
    descriptions = []
    for f in files:
        filename = f.get("filename", "unknown")
        mime = f.get("mime_type", "")
        try:
            data = base64.b64decode(f["content_base64"])
            if "pdf" in mime:
                descriptions.append(f"[PDF: {filename}, {len(data)} bytes]")
            elif "image" in mime:
                descriptions.append(f"[Image: {filename}, {len(data)} bytes]")
            else:
                try:
                    text = data.decode("utf-8")[:2000]
                    descriptions.append(f"[File: {filename}]\n{text}")
                except Exception:
                    descriptions.append(f"[Binary: {filename}, {len(data)} bytes]")
        except Exception:
            descriptions.append(f"[Could not decode: {filename}]")
    return "\n".join(descriptions)


@app.get("/health")
@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/")
@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]
    base_url = creds["base_url"]
    session_token = creds["session_token"]

    logger.info(f"Task: {prompt[:200]}...")

    # Build prompt for Gemini
    user_prompt = f"Task prompt:\n{prompt}"
    if files:
        user_prompt += f"\n\nAttached files:\n{extract_file_content(files)}"

    # Ask Gemini to plan API calls
    if not client:
        logger.error("No Gemini client — GEMINI_API_KEY not set")
        return JSONResponse({"status": "completed"})

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[SYSTEM_PROMPT, user_prompt],
            config={"temperature": 0.1, "max_output_tokens": 4096},
        )
        text = response.text.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        calls = json.loads(text)
        if not isinstance(calls, list):
            calls = [calls]

        logger.info(f"Planned {len(calls)} API calls")

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return JSONResponse({"status": "completed"})

    # Execute
    try:
        results = execute_api_calls(calls, base_url, session_token)
        logger.info(f"Done — {len(results)} calls executed")
    except Exception as e:
        logger.error(f"Execution error: {e}")

    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
