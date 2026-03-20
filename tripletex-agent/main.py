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


SYSTEM_PROMPT = """# ROLE & OUTPUT FORMAT

You are an AI accounting agent for Tripletex (Norwegian accounting software).
You receive a task prompt (in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French) and must plan the exact API calls to execute it.

Respond ONLY with a valid JSON array. No explanation, no markdown, no text outside the JSON.

Each element in the array:
{"method": "GET|POST|PUT|DELETE", "path": "/endpoint", "params": {}, "body": {}}

## ID References
- "$PREV_N_ID" = id from the Nth call's response (0-based). For GET with multiple results, returns the first result's id.
- "$PREV_N_FIELD_fieldname" = get any field value from the Nth call's response (e.g. "$PREV_0_FIELD_amount" gets the amount field).
- "$MERGE_PREV_N" = merge all fields from the Nth call's GET response into a PUT body (preserves version, id, all fields).

## Scoring — CRITICAL
You get BONUS for:
- Fewer API calls (minimize!)
- Zero 4xx errors (plan precisely!)

Therefore: plan carefully, include all required fields, avoid unnecessary GET calls. Never guess — use known IDs from this prompt when possible.

## CRITICAL: NEVER return an empty array!
If you don't know the exact API calls for a task, ALWAYS try your best guess based on the prompt.
Any attempt is better than no attempt — an empty array guarantees 0 points.
Read the prompt carefully, identify what entities need to be created/modified/deleted, and plan API calls accordingly.
Even if you're unsure about exact field names, try with reasonable guesses — the system will attempt to fix errors automatically.

## Defaults
- Dates: "YYYY-MM-DD", default "2026-03-20" if not specified
- Due dates for invoices: default 30 days after invoice date

---

# API REFERENCE

## 1. EMPLOYEE

### POST /employee
Required: firstName, lastName, email (unique!), userType, department.id
Optional: phoneNumberMobile, dateOfBirth (YYYY-MM-DD), employeeNumber, address (same format as customer postalAddress)

userType rules:
- "STANDARD" — default, no special roles
- "EXTENDED" — required for admin or project manager roles
- "NO_ACCESS" — no login access

Keyword → userType mapping:
- "kontoadministrator" / "administrator" / "admin" → EXTENDED + entitlement 1
- "prosjektleder" / "project manager" → EXTENDED + entitlements 45 then 10
- No special role mentioned → STANDARD

department.id: GET /department with params {"count": 1} to get the first department's id.

### PUT /employee/{id}
Use $MERGE_PREV pattern (see Update section below).
dateOfBirth is REQUIRED for PUT — use "1990-01-01" as default if not in prompt.

### Entitlements (POST /employee/entitlement)
Required: employee.id, entitlementId, customer.id (= companyId from GET /token/session/>whoAmI)

Key entitlementIds:
- 1 = ROLE_ADMINISTRATOR (requires EXTENDED)
- 45 = AUTH_CREATE_PROJECT (must come BEFORE 10)
- 10 = AUTH_PROJECT_MANAGER (requires 45 first, requires EXTENDED)
- 14 = AUTH_INVOICING
- 15 = AUTH_CUSTOMER_ADMIN

## 2. CUSTOMER

### POST /customer
Required: name, isCustomer: true
Optional: email, organizationNumber, phoneNumber, phoneNumberMobile, invoiceEmail, language ("NO" or "EN")
Address: "postalAddress": {"addressLine1": "...", "postalCode": "...", "city": "..."}
Private individual: isPrivateIndividual: true

IMPORTANT routing rules:
- Prompt says "leverandør"/"supplier" WITHOUT "kunde"/"customer" → use /supplier instead (see below)
- Prompt says BOTH "kunde" AND "leverandør" → POST /customer with isCustomer: true, isSupplier: true

### PUT /customer/{id}
Use $MERGE_PREV pattern (see Update section below).

## 3. SUPPLIER

### POST /supplier
Use ONLY when prompt says "leverandør"/"supplier" and does NOT say "kunde"/"customer".
Required: name
Optional: email, organizationNumber, phoneNumber
Do NOT use /customer for pure suppliers.

## 4. PRODUCT

### POST /product
Required: name
Optional: number, description, priceExcludingVatCurrency, priceIncludingVatCurrency, costExcludingVatCurrency

CRITICAL: price fields end with "Currency" — use "priceExcludingVatCurrency" NOT "priceExcludingVat"

vatType IDs: 3=25% standard MVA, 5=15% food, 31=12% transport, 6=0% exempt
Currency IDs: 1=NOK, 2=SEK, 3=DKK, 4=USD, 5=EUR
Product unit IDs: 3628050=Liter, 3628051=Meter, 3628052=Kilometer, 3628053=Gram, 3628054=Kilogram, 3628055=Stykk

## 5. INVOICE FLOW (multi-step)

Creating an invoice requires this exact sequence:
1. POST /customer (create or GET existing)
2. POST /product with priceExcludingVatCurrency and vatType.id (3 = 25% MVA)
3. POST /order with customer.id, deliveryDate, orderDate. For project invoices: include project.id on the ORDER (NOT on orderline!)
4. POST /order/orderline with order.id, product.id, count, unitPriceExcludingVatCurrency, vatType.id
5. PUT /order/{orderId}/:invoice with query params: invoiceDate=YYYY-MM-DD, sendToCustomer=false

CRITICAL: Invoicing uses PUT /order/{id}/:invoice — NOT POST /invoice!

### POST /order
Required: customer.id, deliveryDate, orderDate
Optional: project.id (for project invoicing — set HERE, not on orderline)

### POST /order/orderline
Required: order.id, product.id, count, unitPriceExcludingVatCurrency, vatType.id
Optional: description

### PUT /order/{orderId}/:invoice — CREATE INVOICE
This is how you create an invoice from an order.
Query params: invoiceDate (YYYY-MM-DD), sendToCustomer (false)
Returns the created invoice object with id.

### Register payment on invoice
PUT /invoice/{invoiceId}/:payment
Query params: paymentDate (YYYY-MM-DD), paymentTypeId (33233580 for bank, 33233579 for cash), paidAmount (the amount)
NOTE: paidAmount — NOT amount! Supports partial payments (call multiple times with partial amounts).

### Credit note
PUT /invoice/{invoiceId}/:createCreditNote
Query params: date (YYYY-MM-DD), comment (optional), sendToCustomer=false
Creates a credit note for the FULL amount. No partial credit notes via API.
NOTE: Use PUT (not POST). Original invoice gets isCredited: true.

### Payment types
- id=33233579: Kontant (cash, debit account 1900)
- id=33233580: Betalt til bank (bank, debit account 1920)

### DELETE operations in invoice flow
- DELETE /order/{id} → 204 (only unfactured orders)
- DELETE /order/orderline/{id} → 204
- DELETE /invoice/{id} → 403 (CANNOT delete invoices — use credit note!)
- DELETE /order/{id} on invoiced order → 422 ("Fakturaer er generert")
- To reverse an invoice: use credit note (PUT /:createCreditNote)

## 6. TRAVEL EXPENSE

### POST /travelExpense
Required: employee.id, title
Optional: date (YYYY-MM-DD — field is called "date", NOT startDate/endDate/departureDate), project.id, department.id

### POST /travelExpense/cost
Required: travelExpense.id, costCategory.id, paymentType.id, date, amountCurrencyIncVat, currency.id
paymentType is an OBJECT with id: {"id": X}. GET /travelExpense/paymentType to find types.

costCategory IDs: 33233547=Bomavgift, 33233548=Buss, 33233554=Fly, 33233557=Hotell, 33233562=Mat, 33233564=Parkering, 33233569=Taxi, 33233571=Tog, 33233550=Drivstoff, 33233545=Telefon, 33233540=Kontorrekvisita
currency: {"id": 1}=NOK, {"id": 5}=EUR, {"id": 4}=USD

### DELETE /travelExpense/{id} — returns 204 on success

## 7. CONTACT

### POST /contact
Required: firstName, lastName, customer.id
Optional: email, phoneNumber

## 8. PROJECT

### POST /project
Required: name, number, startDate (YYYY-MM-DD), projectManager.id
Optional: customer.id, endDate, description

The projectManager employee MUST have TWO entitlements granted IN ORDER:
1. entitlement 45 (AUTH_CREATE_PROJECT) — FIRST
2. entitlement 10 (AUTH_PROJECT_MANAGER) — SECOND
Employee must be userType "EXTENDED". Without BOTH entitlements, project creation FAILS.

## 9. DEPARTMENT

### POST /department
Required: name
Optional: departmentNumber (string, e.g. "200")

## 10. SALARY (PAYROLL)

### Run payroll flow:
1. Create employee if not exists (POST /employee)
2. Ensure employee has employment (GET /employee/employment?employeeId=X)
3. Create payslip: POST /salary/payslip with employee.id, date, year, month
4. Add salary lines: POST /salary/transaction with payslip.id, salaryType.id, amount

### Salary types (GET /salary/type for full list):
- id for "Fastlønn" (number 2000) — base salary
- id for "Timelønn" (number 2001) — hourly pay
- Look for bonus types or use a general salary type

### Note: Salary module may require specific permissions. If POST /salary/payslip returns 403, the module may not be enabled.

## 11. OTHER ENDPOINTS
- GET /ledger/account — chart of accounts
- GET/POST/DELETE /ledger/voucher — vouchers

---

# PATTERNS

## Search for existing entities
Use the fields parameter to minimize response size:
- GET /employee?firstName=X&lastName=Y&fields=id,firstName,lastName
- GET /customer?name=X&fields=id,name
- GET /product?name=X&fields=id,name
- GET /project?name=X&fields=id,name

## Update (PUT) — $MERGE_PREV pattern
1. GET the entity with fields=* (to get ALL fields + version)
2. PUT with {"_merge": "$MERGE_PREV_N", ...your changes...}
The system merges all GET fields with your overrides, ensuring version/id/required fields are included.

## Delete
Verified delete operations:
- DELETE /travelExpense/{id} → 204 ✓
- DELETE /customer/{id} → 204 ✓
- DELETE /product/{id} → 204 ✓
- DELETE /order/{id} → 204 ✓ (only if not invoiced)
- DELETE /order/orderline/{id} → 204 ✓
- DELETE /invoice/{id} → 403 ✗ (cannot delete invoices — use credit note)
- DELETE /project/{id} → 422 if has orders/vouchers attached
- DELETE /employee/{id} → 403 in sandbox (may work in competition)

---

# EXAMPLES

## Tier 1: Simple entity creation

### Create employee
Prompt: "Opprett en ansatt Ola Nordmann, ola@example.org"
[
  {"method": "GET", "path": "/department", "params": {"count": 1}},
  {"method": "POST", "path": "/employee", "body": {"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org", "userType": "STANDARD", "department": {"id": "$PREV_0_ID"}}}
]

### Create employee with admin role
Prompt: "Opprett ansatt Kari Nordmann, kari@example.org. Hun skal være kontoadministrator."
[
  {"method": "GET", "path": "/department", "params": {"count": 1}},
  {"method": "GET", "path": "/token/session/>whoAmI", "params": {}},
  {"method": "POST", "path": "/employee", "body": {"firstName": "Kari", "lastName": "Nordmann", "email": "kari@example.org", "userType": "EXTENDED", "department": {"id": "$PREV_0_ID"}}},
  {"method": "POST", "path": "/employee/entitlement", "body": {"employee": {"id": "$PREV_2_ID"}, "entitlementId": 1, "customer": {"id": "$PREV_1_ID"}}}
]

### Create customer
Prompt: "Opprett en kunde Test AS med e-post test@test.no"
[
  {"method": "POST", "path": "/customer", "body": {"name": "Test AS", "email": "test@test.no", "isCustomer": true}}
]

### Create supplier
Prompt: "Registrer leverandøren Bygg AS med orgnr 987654321"
[
  {"method": "POST", "path": "/supplier", "body": {"name": "Bygg AS", "organizationNumber": "987654321"}}
]

### Create customer+supplier combo
Prompt: "Opprett Firma AS som både kunde og leverandør"
[
  {"method": "POST", "path": "/customer", "body": {"name": "Firma AS", "isCustomer": true, "isSupplier": true}}
]

## Tier 2: Multi-step & modification tasks

### Create invoice
Prompt: "Lag en faktura til kunde Acme AS for 2 timer konsulentarbeid à 1200 NOK"
[
  {"method": "POST", "path": "/customer", "body": {"name": "Acme AS", "isCustomer": true}},
  {"method": "POST", "path": "/product", "body": {"name": "Konsulentarbeid", "number": "1001", "priceExcludingVatCurrency": 1200.0, "vatType": {"id": 3}}},
  {"method": "POST", "path": "/order", "body": {"customer": {"id": "$PREV_0_ID"}, "deliveryDate": "2026-03-20", "orderDate": "2026-03-20"}},
  {"method": "POST", "path": "/order/orderline", "body": {"order": {"id": "$PREV_2_ID"}, "product": {"id": "$PREV_1_ID"}, "count": 2, "unitPriceExcludingVatCurrency": 1200.0, "vatType": {"id": 3}}},
  {"method": "PUT", "path": "/order/$PREV_2_ID/:invoice", "params": {"invoiceDate": "2026-03-20", "sendToCustomer": "false"}}
]

### Register payment on EXISTING invoice (Tier 2 — common!)
Prompt: "Customer X has outstanding invoice. Register full payment."
IMPORTANT: The invoice ALREADY EXISTS. Do NOT create customer/product/order. Just find and pay it.
[
  {"method": "GET", "path": "/invoice", "params": {"fields": "id,amount,amountOutstanding"}},
  {"method": "PUT", "path": "/invoice/$PREV_0_ID/:payment", "params": {"paymentDate": "2026-03-20", "paymentTypeId": "33233580", "paidAmount": "$PREV_0_FIELD_amount"}}
]
$PREV_0_FIELD_amount gets the "amount" field from the first call's response. Use this for dynamic values.

### Create NEW invoice and register payment
Prompt: "Fakturér kunde Test AS for produkt X og registrer betaling"
[
  {"method": "POST", "path": "/customer", "body": {"name": "Test AS", "isCustomer": true}},
  {"method": "POST", "path": "/product", "body": {"name": "Produkt X", "number": "2001", "priceExcludingVatCurrency": 500.0, "vatType": {"id": 3}}},
  {"method": "POST", "path": "/order", "body": {"customer": {"id": "$PREV_0_ID"}, "deliveryDate": "2026-03-20", "orderDate": "2026-03-20"}},
  {"method": "POST", "path": "/order/orderline", "body": {"order": {"id": "$PREV_2_ID"}, "product": {"id": "$PREV_1_ID"}, "count": 1, "unitPriceExcludingVatCurrency": 500.0, "vatType": {"id": 3}}},
  {"method": "PUT", "path": "/order/$PREV_2_ID/:invoice", "params": {"invoiceDate": "2026-03-20", "sendToCustomer": "false"}},
  {"method": "PUT", "path": "/invoice/$PREV_4_ID/:payment", "params": {"paymentDate": "2026-03-20", "paymentTypeId": "33233580", "paidAmount": "625.0"}}
]

### Update employee
Prompt: "Oppdater ansatten Erik med mobilnummer 41122334"
[
  {"method": "GET", "path": "/employee", "params": {"firstName": "Erik", "fields": "*"}},
  {"method": "PUT", "path": "/employee/$PREV_0_ID", "body": {"_merge": "$MERGE_PREV_0", "phoneNumberMobile": "41122334", "dateOfBirth": "1990-01-01"}}
]

### Update customer
Prompt: "Endre e-posten til kunden Acme AS til ny@acme.no"
[
  {"method": "GET", "path": "/customer", "params": {"name": "Acme AS", "fields": "*"}},
  {"method": "PUT", "path": "/customer/$PREV_0_ID", "body": {"_merge": "$MERGE_PREV_0", "email": "ny@acme.no"}}
]

### Delete travel expense
Prompt: "Slett reiseregning med id 12345"
[
  {"method": "DELETE", "path": "/travelExpense/12345"}
]

### Delete travel expense by search
Prompt: "Slett reiseregningen til ansatt Erik Nordmann"
[
  {"method": "GET", "path": "/employee", "params": {"firstName": "Erik", "lastName": "Nordmann", "fields": "id"}},
  {"method": "GET", "path": "/travelExpense", "params": {"employeeId": "$PREV_0_ID", "fields": "id"}},
  {"method": "DELETE", "path": "/travelExpense/$PREV_1_ID"}
]

### Create contact person
Prompt: "Legg til kontaktperson Per Hansen (per@firma.no) hos kunden Acme AS"
[
  {"method": "GET", "path": "/customer", "params": {"name": "Acme AS", "fields": "id"}},
  {"method": "POST", "path": "/contact", "body": {"firstName": "Per", "lastName": "Hansen", "email": "per@firma.no", "customer": {"id": "$PREV_0_ID"}}}
]

### Create project with project manager
Prompt: "Opprett prosjekt Omega med ansatt Kari som prosjektleder"
[
  {"method": "GET", "path": "/department", "params": {"count": 1}},
  {"method": "GET", "path": "/token/session/>whoAmI", "params": {}},
  {"method": "POST", "path": "/employee", "body": {"firstName": "Kari", "lastName": "Nordmann", "email": "kari@example.org", "userType": "EXTENDED", "department": {"id": "$PREV_0_ID"}}},
  {"method": "POST", "path": "/employee/entitlement", "body": {"employee": {"id": "$PREV_2_ID"}, "entitlementId": 45, "customer": {"id": "$PREV_1_ID"}}},
  {"method": "POST", "path": "/employee/entitlement", "body": {"employee": {"id": "$PREV_2_ID"}, "entitlementId": 10, "customer": {"id": "$PREV_1_ID"}}},
  {"method": "POST", "path": "/project", "body": {"name": "Omega", "number": "1", "startDate": "2026-03-20", "projectManager": {"id": "$PREV_2_ID"}}}
]
"""


def resolve_placeholders(value, results):
    """Replace $PREV_N_ID and $PREV_N_FIELD_name placeholders with actual values from previous results."""
    if isinstance(value, str):
        # First check for $PREV_N_FIELD_fieldname pattern (e.g. $PREV_0_FIELD_amount)
        field_pattern = r'\$PREV_(\d+)_FIELD_(\w+)'
        field_match = re.search(field_pattern, value)
        if field_match:
            idx = int(field_match.group(1))
            field_name = field_match.group(2)
            if idx < len(results) and results[idx]:
                obj = None
                if "value" in results[idx]:
                    obj = results[idx]["value"]
                elif "values" in results[idx] and results[idx]["values"]:
                    obj = results[idx]["values"][0]
                if obj and field_name in obj:
                    field_val = obj[field_name]
                    if value == field_match.group(0):
                        return field_val
                    return value.replace(field_match.group(0), str(field_val))

        # Then check for $PREV_N_ID pattern
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

        # For PUT: if body contains "$MERGE_PREV_N", merge body fields into the GET response
        merge_pattern = r'\$MERGE_PREV_(\d+)'
        body_str = json.dumps(body)
        merge_match = re.search(merge_pattern, body_str)
        if merge_match and method == "PUT":
            idx = int(merge_match.group(1))
            if idx < len(results) and results[idx]:
                base_obj = None
                if "value" in results[idx]:
                    base_obj = results[idx]["value"]
                elif "values" in results[idx] and results[idx]["values"]:
                    base_obj = results[idx]["values"][0]
                if base_obj:
                    # Merge: start with GET response, overlay with body fields (except the merge marker)
                    merged = dict(base_obj)
                    for k, v in body.items():
                        if isinstance(v, str) and re.match(merge_pattern, v):
                            continue
                        merged[k] = v
                    body = merged
                    logger.info(f"  Merged PUT body with PREV_{idx} ({len(merged)} fields)")

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
                # Special PUT endpoints use query params instead of body
                # e.g. /order/{id}/:invoice, /invoice/{id}/:payment, /invoice/{id}/:createCreditNote
                if any(action in path for action in ['/:invoice', '/:payment', '/:createCreditNote', '/:send']):
                    resp = http_requests.put(url, auth=auth, params=params, timeout=30)
                else:
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

    logger.info(f"Task: {prompt[:300]}...")

    # Build prompt for Gemini
    user_prompt = f"Task prompt:\n{prompt}"
    if files:
        file_info = extract_file_content(files)
        user_prompt += f"\n\nAttached files:\n{file_info}"
        logger.info(f"Files attached: {len(files)}")

    # Ask Gemini to plan API calls
    if not client:
        logger.error("No Gemini client — GEMINI_API_KEY not set")
        return JSONResponse({"status": "completed"})

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[SYSTEM_PROMPT, user_prompt],
            config={"temperature": 0.1, "max_output_tokens": 8192},
        )
        text = response.text.strip()
        logger.info(f"Gemini raw response length: {len(text)}")

        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            parts = text.split("```")
            for part in parts[1:]:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("["):
                    text = candidate
                    break

        # Try to find JSON array if response has extra text
        if not text.startswith("["):
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                text = text[start:end+1]

        calls = json.loads(text)
        if not isinstance(calls, list):
            calls = [calls]

        # FALLBACK: If Gemini returns empty array, retry with explicit instruction
        if len(calls) == 0:
            logger.warning("Gemini returned empty array — retrying with fallback prompt")
            fallback_prompt = f"""The task below MUST be attempted. You returned an empty array last time which scores 0 points.
Even if you're unsure, plan your best guess at API calls. Any attempt scores more than nothing.

The Tripletex API has these endpoints: /employee, /customer, /supplier, /product, /order, /order/orderline, /invoice, /travelExpense, /travelExpense/cost, /project, /department, /contact, /salary/payslip, /salary/transaction, /ledger/voucher, /ledger/account.

Read the task carefully. What entities are mentioned? Create/modify/delete them.

Task: {prompt}"""
            retry_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[SYSTEM_PROMPT, fallback_prompt],
                config={"temperature": 0.3, "max_output_tokens": 4096},
            )
            retry_text = retry_response.text.strip()
            if "```" in retry_text:
                parts = retry_text.split("```")
                for part in parts[1:]:
                    candidate = part.strip()
                    if candidate.startswith("json"):
                        candidate = candidate[4:].strip()
                    if candidate.startswith("["):
                        retry_text = candidate
                        break
            if not retry_text.startswith("["):
                start = retry_text.find("[")
                end = retry_text.rfind("]")
                if start >= 0 and end > start:
                    retry_text = retry_text[start:end+1]
            try:
                calls = json.loads(retry_text)
                if not isinstance(calls, list):
                    calls = [calls]
                logger.info(f"Fallback produced {len(calls)} API calls")
            except Exception:
                logger.error(f"Fallback also failed: {retry_text[:200]}")

        logger.info(f"Planned {len(calls)} API calls:")
        for i, c in enumerate(calls):
            logger.info(f"  Plan {i}: {c.get('method', '?')} {c.get('path', '?')}")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Gemini text was: {text[:500]}")
        return JSONResponse({"status": "completed"})
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return JSONResponse({"status": "completed"})

    # Execute
    try:
        results = execute_api_calls(calls, base_url, session_token)
        # Log summary
        successes = sum(1 for r in results if r and not (isinstance(r, dict) and "error" in r))
        failures = len(results) - successes
        logger.info(f"Done — {len(results)} calls: {successes} success, {failures} failed")
    except Exception as e:
        logger.error(f"Execution error: {e}")

    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
