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
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

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
- Fewer WRITE calls (POST, PUT, DELETE) — minimize these!
- Zero 4xx errors on write calls — every failed write REDUCES your score!

GET calls are FREE — they do NOT count against efficiency. Read as much as you need.
Therefore:
1. ALWAYS search with GET before creating with POST — if an entity already exists, use it!
2. When the prompt gives a product number, org number, or name: GET first, create only if not found.
3. Plan write calls precisely — include all required fields to avoid 422 errors.
4. Use known IDs from ENVIRONMENT to avoid unnecessary writes.
5. A 422 error on a write call + retry = TWO write calls + ONE error. Much worse than one GET + one correct write.

## TASK PATTERNS — Follow these flows!
| Pattern | Example | API Flow |
|---------|---------|----------|
| Create single entity | "Create employee Ola" | POST /employee |
| Create with linking | "Create invoice for customer" | GET /customer → POST /order → POST /invoice |
| Modify existing | "Add phone to contact" | GET /entity?fields=* → PUT /entity/{id} |
| Delete/reverse | "Delete travel expense" | GET /entity → DELETE /entity/{id} |
| Multi-step setup | "Register payment" | GET /customer → GET /product → POST /order → POST /invoice |

CRITICAL RULE: When the prompt references EXISTING entities (customer name, org number, product number, invoice, employee email), ALWAYS use GET to find them. The competition sandbox has pre-populated data specific to each task. These entities ALREADY EXIST — do NOT create them!
CRITICAL: If POST /employee returns "email already exists" (422), the employee is ALREADY in the system. Use GET /employee?email=X to find them instead of creating.
CRITICAL: For cost analysis tasks ("total costs increased"), create projects with POST /project. Do NOT try to put "project" or "projectId" fields on ledger postings — those fields don't exist on postings!

## Batch endpoints — use for multiple entities
When creating MULTIPLE entities of the same type, use list endpoints for fewer write calls:
- POST /employee/list — create multiple employees in one call
- POST /product/list — create multiple products in one call
- POST /department/list — create multiple departments in one call
- POST /customer/list — create/update multiple customers in one call

## CRITICAL: NEVER return an empty array!
If you don't know the exact API calls for a task, ALWAYS try your best guess based on the prompt.
Any attempt is better than no attempt — an empty array guarantees 0 points.
Read the prompt carefully, identify what entities need to be created/modified/deleted, and plan API calls accordingly.
Even if you're unsure about exact field names, try with reasonable guesses — the system will attempt to fix errors automatically.

## Defaults
- Dates: "YYYY-MM-DD", default today's date if not specified
- Due dates for invoices: default 30 days after invoice date

## CALCULATION FORMULAS — USE THESE EXACTLY!
- MVA 25%: If amount is "inkl MVA": netto = brutto / 1.25, MVA = brutto - netto. Example: 12500 inkl MVA → netto=10000, MVA=2500
- MVA 15% (food): netto = brutto / 1.15, MVA = brutto - netto
- Linear depreciation (annual): amount = acquisitionCost / lifetimeYears. Example: 240000 / 5 = 48000 per year
- Linear depreciation (monthly): amount = acquisitionCost / lifetimeYears / 12. Example: 240000 / 5 / 12 = 4000 per month
- Tax provision 22%: taxAmount = taxableProfit * 0.22
- Employer tax (AGA) 14.1%: agaAmount = grossSalary * 0.141
- Tax withholding: taxAmount = grossSalary * taxRate (e.g. 38% = 0.38)
- Net salary: netSalary = grossSalary - taxWithholding
- Exchange rate difference: diff = amount * (newRate - oldRate). Positive = agio (gain), negative = disagio (loss)

## TOP RULES — VIOLATING THESE = 0 POINTS
1. Voucher postings: debit and credit MUST use DIFFERENT account IDs. NEVER same account on both rows.
2. Fields syntax: use PARENTHESES for nested fields: customer(id,name) — NEVER customer.id
3. GET /invoice: ALWAYS include invoiceDateFrom=2020-01-01 and invoiceDateTo=2030-12-31
4. NEVER return an empty JSON array — any attempt is better than nothing
5. Use ENVIRONMENT IDs directly — NEVER hardcode IDs, they change per sandbox
6. For voucher postings: set amountCurrency=amount, amountGross=amount, amountGrossCurrency=amount (always same value)
7. When PDF/files are attached: extract ACTUAL values — never send placeholder text like "FROM PDF"
8. ALL id fields MUST be integers, NEVER strings. Write 123 not "123". ALL amounts MUST be numbers, not strings.
9. When prompt says "send" invoice/faktura: use sendToCustomer=true. Otherwise sendToCustomer=false.
10. ALWAYS set invoiceDueDate on orders — use the date from the prompt, or invoiceDate + 30 days as default.
11. For multiple entities of same type: ALWAYS use batch /list endpoints (POST /department/list, POST /product/list) — fewer writes = higher score!

---

# API REFERENCE

## 1. EMPLOYEE

### POST /employee
Required: firstName, lastName, email (unique!), userType, department.id
Optional: phoneNumberMobile, dateOfBirth (YYYY-MM-DD), employeeNumber, nationalIdentityNumber (personnummer/fødselsnummer), address (same format as customer postalAddress)

userType rules:
- "STANDARD" — default, no special roles
- "EXTENDED" — required for admin or project manager roles
- "NO_ACCESS" — no login access

Keyword → userType mapping:
- "kontoadministrator" / "administrator" / "admin" → EXTENDED + entitlement 1
- "prosjektleder" / "project manager" → EXTENDED + entitlements 45 then 10
- No special role mentioned → STANDARD

department.id: Use the department_id from ENVIRONMENT section (pre-fetched). Do NOT call GET /department.

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

### Employment (POST /employee/employment) — for contract/salary tasks
When creating an employee from a contract (PDF), you need BOTH the employee AND an employment record.
POST /employee/employment creates the employment relationship with salary, occupation code, etc.
Required: employee.id, startDate (YYYY-MM-DD)
Employment details can be included INLINE in the employment POST:

Example — complete employee from contract:
POST /employee with: firstName, lastName, email, dateOfBirth, nationalIdentityNumber, userType, department.id
Then POST /employee/employment with:
{
  "employee": {"id": "$PREV_0_ID"},
  "startDate": "2025-01-01",
  "employmentDetails": [{
    "date": "2025-01-01",
    "employmentType": "ORDINARY",
    "employmentForm": "PERMANENT",
    "remunerationType": "MONTHLY_WAGE",
    "workingHoursScheme": "NOT_SHIFT",
    "percentageOfFullTimeEquivalent": 100.0,
    "annualSalary": 500000.0,
    "occupationCode": {"id": 1}
  }]
}
CRITICAL for occupationCode: The id MUST be an integer! Use {"id": 1} as default if not known.
Do NOT use $PREV reference for occupationCode.id — it often fails. Use id=1 (default) unless you found a specific code via GET.

Key fields in employmentDetails:
- employmentType: "ORDINARY" (standard), "MARITIME"
- employmentForm: "PERMANENT" (fast), "TEMPORARY" (midlertidig)
- remunerationType: "MONTHLY_WAGE", "HOURLY_WAGE", "FEE"
- workingHoursScheme: "NOT_SHIFT" (standard), "ROUND_THE_CLOCK", "CONTINUOUS"
- percentageOfFullTimeEquivalent: 100.0 = full time, 50.0 = half time
- annualSalary: yearly salary in NOK
- hourlyWage: hourly wage (alternative to annualSalary)
- occupationCode: {"id": N} — use GET /employee/employment/occupationCode to find correct code

For PDF contract tasks: Read ALL details from the PDF — name, personnummer, birthday, department, occupation, salary, start date, employment percentage. Include EVERYTHING.

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

### GET /product — SEARCH FIRST!
CRITICAL: If the prompt gives a product NUMBER (e.g. "produkt 7390", "product number 6965"), the product likely ALREADY EXISTS in the sandbox.
ALWAYS search first: GET /product?number=X&fields=id,name,priceExcludingVatCurrency
Only create with POST if GET returns empty results.
Same applies to customers and suppliers with organization numbers — search by organizationNumber first!

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
5. PUT /order/{orderId}/:invoice with query params: invoiceDate=YYYY-MM-DD, sendToCustomer=true/false, invoiceDueDate=YYYY-MM-DD

CRITICAL: Invoicing uses PUT /order/{id}/:invoice — NOT POST /invoice!
CRITICAL: ALWAYS set invoiceDueDate = invoiceDate + 30 days. If prompt says "send"/"sende"/"enviar"/"envoyer", set sendToCustomer=true!
CRITICAL: GET /invoice fields — use invoiceNumber (NOT number!), amount, amountOutstanding, customer(id,name). "number" does NOT exist on InvoiceDTO!
CRITICAL: GET /invoice REQUIRES invoiceDateFrom and invoiceDateTo params! Always include: invoiceDateFrom=2020-01-01, invoiceDateTo=2030-12-31

### POST /order
Required: customer.id, deliveryDate, orderDate
Optional: project.id (for project invoicing — set HERE, not on orderline)

### POST /order/orderline
Required: order.id, product.id, count, unitPriceExcludingVatCurrency, vatType.id, description (ALWAYS include — use product/service name from prompt!)
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
CRITICAL: The date MUST be ON or AFTER the original invoice date! Use today's date to be safe. Never use a date before the invoice was created.
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
For "innlogget bruker" / "current user" / "logged in user": use employee_id from ENVIRONMENT — do NOT create a new employee!

### POST /travelExpense/cost
Required: travelExpense.id, costCategory.id, paymentType.id, date, amountCurrencyIncVat, currency.id
NOTE: Do NOT include "description" or "comment" fields — they do NOT exist on travelExpense/cost!
paymentType: use travel_payment_type_id from ENVIRONMENT
costCategory: use cost_cat_*_id from ENVIRONMENT (e.g. cost_cat_fly_id for flights, cost_cat_taxi_id for taxi)
CRITICAL: costCategory IDs are DIFFERENT per sandbox — NEVER hardcode them! Use the IDs from ENVIRONMENT section.
currency: {"id": 1}=NOK, {"id": 5}=EUR, {"id": 4}=USD

For "dagpenger" / "per diem" / "Tagegeld": use cost_cat_mat_id or create a manual voucher instead.

Example — travel expense with flight + taxi:
[
  {"method": "GET", "path": "/employee", "params": {"email": "ola@example.org", "fields": "id"}},
  {"method": "POST", "path": "/travelExpense", "body": {"employee": {"id": "$PREV_0_ID"}, "title": "Kundebesøk Bergen", "date": "2026-03-20"}},
  {"method": "POST", "path": "/travelExpense/cost", "body": {"travelExpense": {"id": "$PREV_1_ID"}, "costCategory": {"id": "COST_CAT_FLY_ID from ENVIRONMENT"}, "paymentType": {"id": "TRAVEL_PAYMENT_TYPE_ID from ENVIRONMENT"}, "date": "2026-03-20", "amountCurrencyIncVat": 5500.0, "currency": {"id": 1}}},
  {"method": "POST", "path": "/travelExpense/cost", "body": {"travelExpense": {"id": "$PREV_1_ID"}, "costCategory": {"id": "COST_CAT_TAXI_ID from ENVIRONMENT"}, "paymentType": {"id": "TRAVEL_PAYMENT_TYPE_ID from ENVIRONMENT"}, "date": "2026-03-20", "amountCurrencyIncVat": 450.0, "currency": {"id": 1}}}
]
NOTE: Use cost_cat_*_id and travel_payment_type_id from ENVIRONMENT. Do NOT include "description" on cost — use "comment" if needed.

### DELETE /travelExpense/{id} — returns 204 on success

## 7. CONTACT

### POST /contact
Required: firstName, lastName, customer.id
Optional: email, phoneNumberMobile (NOT phoneNumber — that field does not exist on contact!)

## 8. PROJECT

### POST /project
Required: name, number, startDate (YYYY-MM-DD), projectManager.id
Optional: customer.id, endDate, description

The projectManager employee MUST have TWO entitlements granted IN ORDER:
1. entitlement 45 (AUTH_CREATE_PROJECT) — FIRST
2. entitlement 10 (AUTH_PROJECT_MANAGER) — SECOND
Employee must be userType "EXTENDED". Without BOTH entitlements, project creation FAILS.

### POST /activity
For creating activities (e.g. for cost analysis tasks that say "create an activity for each project"):
Required: name, activityType (MUST be "PROJECT_ACTIVITY" for project-related activities, or "GENERAL_ACTIVITY")
CRITICAL: Do NOT put "project" field on the activity — it does NOT exist! Activities are global, not project-specific.
If the task says "create an activity for each project", create the activities separately, then use them in timesheet entries.

## 9. DEPARTMENT

### POST /department
Required: name
Optional: departmentNumber (string, e.g. "200")

## 10. SALARY (PAYROLL) — CRITICAL: READ CAREFULLY!

CRITICAL: NEVER use POST /salary/payslip or POST /salary/transaction — they ALWAYS return 500/403! Use POST /ledger/voucher with voucherType "Lønnsbilag" instead!

### ALTERNATIVE: Register salary as a manual voucher (Lønnsbilag)
This is the CORRECT approach when the salary module is not available:
POST /ledger/voucher with voucherType "Lønnsbilag" and postings for salary accounts.

SALARY RULES — choose the right pattern:
A) SIMPLE salary (no tax/AGA mentioned): 2 postings — debit 5000, credit 2930
B) SALARY WITH TAX (skattetrekk mentioned): 3 postings — debit 5000 (gross), credit 2600 (tax=gross×rate), credit 2930 (net=gross-tax)
C) SALARY WITH TAX + AGA: Pattern B + SEPARATE voucher for AGA — debit 5400 (AGA=gross×0.141), credit 2770

CRITICAL: If prompt mentions skattetrekk/tax rate AND AGA: create TWO vouchers!
Voucher 1: salary with tax withholding (5000/2600/2930)
Voucher 2: employer tax AGA (5400/2770)

Example for SIMPLE salary 49550 kr (pattern A):
POST /ledger/voucher with:
- voucherType: VOUCHER_TYPE_SALARY_ID from ENVIRONMENT
- postings: [
    {account: ACCOUNT_5000_ID, amount: 49550, row: 1, description: "Fastlønn"},
    {account: ACCOUNT_2930_ID, amount: -49550, row: 2, description: "Skyldig lønn"}
  ]

For salary + bonus (49550 + 8300):
- posting 1: account 5000, amount: 57850 (total), row: 1
- posting 2: account 2930, amount: -57850, row: 2

### Salary types (for reference only — salary module may not be available):
- "Fastlønn" (number 2000), "Timelønn" (number 2001)

## 11. VOUCHER / BILAG (ledger/voucher)

### POST /ledger/voucher
For creating manual journal entries (bilag).
Required: date, description, voucherType.id, postings (array)

CRITICAL: Each posting MUST have the "row" field (integer, starting at 1)! Without "row", ALL voucherTypes fail with "systemgenererte" error.

Each posting requires: date, account.id, amount, amountCurrency, amountGross, amountGrossCurrency, currency.id, row, description
Postings MUST balance (sum of amounts = 0). Positive = debit, negative = credit.

CRITICAL: The debit posting and credit posting MUST use DIFFERENT accounts! NEVER use the same account.id for both rows.
When you look up two accounts with separate GET calls, use $PREV_0_ID for debit and $PREV_1_ID for credit — they are DIFFERENT IDs!
Example: If GET call 0 returns account 6010 and GET call 1 returns account 1200, use $PREV_0_ID (6010) for debit and $PREV_1_ID (1200) for credit.

## ACCOUNTING DIMENSIONS (fri regnskapsdimensjon)
The API HAS dedicated dimension endpoints:

### POST /ledger/accountingDimensionName
Create a dimension (e.g. "Region", "Avdeling", "Prosjekttype"):
CRITICAL: The field is called "dimensionName" (NOT "name"!)
Required: dimensionName
Example: {"dimensionName": "Region"}

### POST /ledger/accountingDimensionValue
Create dimension values (e.g. "Nord", "Sør", "Vest"):
CRITICAL: The field is called "displayName" (NOT "name" or "dimensionValueName"!)
Required: displayName
Do NOT include "accountingDimensionName" or "dimensionName" — those fields don't exist on this endpoint!
Example: {"displayName": "Nord"}

### GET /ledger/accountingDimensionName/search — find existing dimensions
### GET /ledger/accountingDimensionValue/search — find existing dimension values

CRITICAL: There is NO API field to link dimensions to voucher postings! Do NOT use "accountingDimensionValues", "customDimensionValues", or similar fields on postings — they do NOT exist and will cause 422 errors.
The correct approach: 1) Create dimension name 2) Create dimension values 3) Create the voucher separately (mention the dimension in the posting description text only).
ALWAYS also create the voucher/bilag part of the task — partial credit is better than 0.

Common accounts: 1920=Bankinnskudd, 2400=Leverandørgjeld, 2700=Utg MVA høy, 2710=Inng MVA høy, 3000=Salgsinntekt, 6800=Kontorrekvisita, 6900=Telefon, 7100=Kontortjenester

### GET /ledger/account
Search by number: ?numberFrom=7100&numberTo=7100&fields=id,number,name

## 12. SUPPLIER INVOICE (leverandørfaktura) — USE VOUCHER!

CRITICAL: POST /supplierInvoice returns 500 — DO NOT USE IT!
Instead, register supplier invoices as a voucher with voucherType "Leverandørfaktura".

### How to register a supplier invoice (as voucher):
For an invoice of 18000 NOK inkl MVA (14400 netto + 3600 MVA 25%):
1. GET /supplier to find the supplier ID
2. POST /ledger/voucher with:
- voucherType: voucher_type_supplier_id from ENVIRONMENT
- 3 postings that MUST balance to 0:
  1. Debit expense account (e.g. 6800 Kontorrekvisita): +14400 (netto beløp)
  2. Debit MVA account (2710 Inngående MVA): +3600 (25% av netto)
  3. Credit supplier account (2400 Leverandørgjeld): -18000 (totalbeløp inkl MVA)

CRITICAL: supplier.id MUST be included on EVERY posting! Without it: "Leverandør mangler" error.
Use account IDs from ENVIRONMENT. Each posting MUST have "row" field.

## 13. TIMESHEET

### POST /timesheet/entry
For registering hours worked.
Required: employee.id, activity.id, date, hours
Optional: project.id (for project-specific hours), comment

For "innlogget bruker" / "logged in user" / "current user": use employee_id from ENVIRONMENT directly — do NOT call GET /token/session or /employee/me!
activity.id: use the first activity_id from ENVIRONMENT (e.g. "Fakturerbart arbeid" or "Administrasjon")

## 14. VOUCHER OPERATIONS

### PUT /ledger/voucher/{id}/:reverse
Reverse an existing voucher — creates a counter-voucher that cancels out the original.
Use this for ERROR CORRECTION instead of creating manual correction vouchers!
Much simpler than manual corrections: just find the wrong voucher and reverse it.

### GET, POST, DELETE /ledger/voucher/openingBalance
For opening balance (åpningsbalanse) tasks. Dedicated endpoint!
POST creates opening balance entries. DELETE removes them.

## 16. TRAVEL EXPENSE — ADDITIONAL FEATURES

### POST /travelExpense/mileageAllowance
For "kjøregodtgjørelse" / "mileage allowance" / "Kilometergodtgjørelse":
Required: travelExpense.id, rateCategory.id, date, departureLocation, destination, km
NOTE: Do NOT include rateType — it causes errors. Just use rateCategory from GET /travelExpense/rateCategory.

### POST /travelExpense/perDiemCompensation
For "diett" / "dagpenger" / "per diem" / "Tagegeld":
Required: travelExpense.id, rateCategory.id, countryCode, overnightAccommodation, location, date
NOTE: Do NOT include rateType — it causes errors. Just use rateCategory.

### PUT /travelExpense/:createVouchers
Create accounting vouchers from an approved travel expense.

## 17. ASSET (anleggsmidler)

### POST /asset
For registering fixed assets (anleggsmidler):
Required: name, acquisitionCost, acquisitionDate
Optional: depreciationAccountId, lifetimeInMonths, incomingBalance, accumulatedDepreciation
After creating asset, register depreciation via POST /ledger/voucher (6010/1200).

## 18. YEAR-END / ANNUAL ACCOUNTS

### GET /ledger/annualAccount — annual accounts data
### GET /ledger/closeGroup — close groups for period closing
### PUT /ledger/posting/:closePostings — close postings for period
### GET /saft/exportSAFT — export SAF-T file
### POST /saft/importSAFT — import SAF-T file

## 19. PURCHASE ORDER (innkjøpsordre)

### POST /purchaseOrder
Required: deliveryDate, supplier.id, ourContact.id (use employee_id from ENVIRONMENT)
Optional: orderDate, receiverEmail
After creating: add order lines with POST /purchaseOrder/orderline

### POST /purchaseOrder/orderline
Required: purchaseOrder.id, product.id (or description), count, unitPrice
For receiving goods: POST /purchaseOrder/goodsReceipt, then POST /purchaseOrder/goodsReceiptLine

## 20. EMPLOYEE ADDITIONAL

### POST /employee/nextOfKin
For "pårørende" / "next of kin" / "emergency contact":
Required: name, phoneNumber
Optional: employee.id

### POST /employee/hourlyCostAndRate
For setting hourly cost and rate for an employee.

## 21. RECEIPT/EXPENSE WITH DEPARTMENT (kvittering/utlegg)

When a task says "register expense from this receipt" or "post expense to department X":
1. Read the attached PDF/image to extract: product name, amount, VAT
2. POST /travelExpense with employee_id from ENVIRONMENT and the department
3. POST /travelExpense/cost with the correct costCategory and amount
OR use POST /ledger/voucher with the expense account and department

For department-specific postings, include department on the voucher or travel expense.
Look up the department first: GET /department?query=X&fields=id,name

## 22. MULTI-CURRENCY + EXCHANGE RATE DIFFERENCES (disagio/agio)

When a task involves foreign currency:
- GET /currency to find currency ID (NOK=1, SEK=2, DKK=3, USD=4, EUR=5, GBP=6)
- Use currency.id in product, invoice, or voucher postings
- GET /currency/{fromCurrencyID}/exchangeRate for exchange rates

For EXCHANGE RATE DIFFERENCES (disagio/agio):
When a customer pays in a different currency and the rate has changed:
1. Calculate the NOK difference: (invoice amount × original rate) - (invoice amount × payment rate)
2. If payment rate < invoice rate: loss = "disagio" → debit account 8160 (Valutakurstap/Disagio)
3. If payment rate > invoice rate: gain = "agio" → credit account 8060 (Valutakursgevinst/Agio)
4. Register payment on the invoice: PUT /invoice/{id}/:payment
5. Register the exchange rate difference as a voucher:
   - Disagio (loss): debit 8160, credit 1500 (kundefordringer)
   - Agio (gain): debit 1500 (kundefordringer), credit 8060
Use account_8160_id and account_8060_id from ENVIRONMENT.

## 22. BANK RECONCILIATION (bankavstemming)

For bank reconciliation tasks with CSV:
1. Read the CSV file to identify ALL transactions (incoming and outgoing payments)
2. For each INCOMING payment: find the matching customer invoice with GET /invoice, then register payment with PUT /invoice/{id}/:payment
3. For each OUTGOING payment (supplier): register as voucher (debit 2400 leverandørgjeld, credit 1920 bank)
4. For unmatched transactions: register as voucher with appropriate accounts

CRITICAL for matching payments to invoices:
- GET /invoice with invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31 to find ALL invoices
- Match by customer name or amount from the CSV
- Use PUT /invoice/{id}/:payment with the correct paidAmount
- For partial payments: paidAmount can be less than invoice amount
- For supplier payments: use POST /ledger/voucher (debit 2400, credit 1920)

## 23. PDF SUPPLIER INVOICE (leverandørfaktura fra PDF)

When a task says "register the supplier invoice from the attached PDF":
1. Read the PDF to extract: supplier name, org number, invoice number, amount, VAT, expense account
2. GET /supplier by organizationNumber or name — create if not found
3. POST /ledger/voucher with voucherType leverandørfaktura + supplier.id on EVERY posting
Same flow as regular supplier invoice, but data comes from the PDF attachment.

## 24. MONTH-END CLOSING (månedsavslutning)

For month-end tasks (periodisering + avskrivning + lønnsavsetning):
1. Periodisering: POST /ledger/voucher — debit kostnadskonto, credit 1700 (forskuddsbetalt)
2. Avskrivning: POST /ledger/voucher — debit 6010/6030, credit 1200/1209 (DIFFERENT accounts!)
3. Lønnsavsetning: POST /ledger/voucher — debit 5000, credit 2900 (påløpt lønn)
Use account_2900_id and account_6030_id from ENVIRONMENT.

## 25. YEAR-END / PERIOD CLOSING (årsoppgjør)

For year-end tasks (avskrivning, periodisering, skatteavsetning):
1. Use account IDs from ENVIRONMENT (account_XXXX_id format)
2. Create vouchers with the CORRECT account IDs — NEVER use the same account for both debit and credit!
3. CRITICAL: Each voucher must debit ONE account and credit a DIFFERENT account.

Example: Depreciation of IT equipment (avskrivning):
- Use account_6010_id from ENVIRONMENT for depreciation expense
- Use account_1209_id from ENVIRONMENT for accumulated depreciation (DIFFERENT account!)
- POST /ledger/voucher with postings:
  - Row 1: account $PREV_0_ID (6010 debit), amount: +16575
  - Row 2: account $PREV_1_ID (1209 credit), amount: -16575
NOTE: $PREV_0_ID and $PREV_1_ID are DIFFERENT accounts! Row 1 uses the expense account, Row 2 uses the asset/liability account.

For annual accounts / period closing:
- GET /ledger/annualAccount — shows configured accounting years
- Use account 8800 (årsresultat) and 2050 (egenkapital) for closing entries

## 24. OTHER ENDPOINTS
- ALL ledger accounts are in ENVIRONMENT — use account_XXXX_id directly
- GET /ledger/posting?dateFrom=X&dateTo=Y — query postings
- GET /ledger/posting/openPost — open postings (utestående poster)
- GET /inventory — warehouse/stock info
- GET /purchaseOrder — purchase orders
- GET /balanceSheet — balance sheet (saldobalanse)
- GET /resultbudget — result budget
- GET /currency/{id}/rate — exchange rates
- GET /token/session/>whoAmI — current user info (NOTE: path uses > not /)

---

# PATTERNS

## Search for existing entities
Use the fields parameter to minimize response size.
CRITICAL: For nested fields, use PARENTHESES not dots! Example: fields=id,amount,customer(id,name) — NOT customer.id!
CRITICAL: When an organization number (orgnr/org.nr) is given in the prompt, ALWAYS search by organizationNumber! This is the most reliable identifier.
- GET /employee?firstName=X&lastName=Y&fields=id,firstName,lastName
- GET /customer?name=X&fields=id,name
- GET /customer?organizationNumber=X&fields=id,name (when orgnr is given — PREFERRED!)
- GET /supplier?organizationNumber=X&fields=id,name (when orgnr is given — PREFERRED!)
- GET /product?name=X&fields=id,name
- GET /product?number=X&fields=id,name,priceExcludingVatCurrency (when product NUMBER is given — PREFERRED!)
- GET /project?name=X&fields=id,name
- GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&fields=id,amount,customer(id,name)

If GET returns empty results (no matches), try broader search or create the entity.
If GET returns results, use the existing entity's ID — do NOT create a duplicate!

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

### Create employee from contract (PDF attachment — Tier 3!)
Prompt: "Du har mottatt en arbeidskontrakt (se vedlagt PDF). Opprett ansatt med alle kontraktdata."
IMPORTANT: Read the attached PDF carefully and extract ALL values. Do NOT use placeholder text — use the ACTUAL values from the PDF.
[
  {"method": "GET", "path": "/employee/employment/occupationCode", "params": {"fields": "id,nameNO,code"}},
  {"method": "POST", "path": "/employee", "body": {"firstName": "Maria", "lastName": "Gonzalez", "email": "maria.gonzalez@firma.no", "dateOfBirth": "1990-05-15", "nationalIdentityNumber": "15059012345", "userType": "STANDARD", "department": {"id": DEPARTMENT_ID}}},
  {"method": "POST", "path": "/employee/employment", "body": {"employee": {"id": "$PREV_1_ID"}, "startDate": "2026-04-01", "employmentDetails": [{"date": "2026-04-01", "employmentType": "ORDINARY", "employmentForm": "PERMANENT", "remunerationType": "MONTHLY_WAGE", "workingHoursScheme": "NOT_SHIFT", "percentageOfFullTimeEquivalent": 80.0, "annualSalary": 520000.0, "occupationCode": {"id": "$PREV_0_ID"}}]}}
]
NOTE: The values above are EXAMPLES. Replace them with the ACTUAL values from the attached PDF/contract. Every field must contain real data, not placeholder text.

### Create employee (uses ENVIRONMENT values — no GET calls needed!)
Prompt: "Opprett en ansatt Ola Nordmann, ola@example.org"
[
  {"method": "POST", "path": "/employee", "body": {"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org", "dateOfBirth": "1985-06-15", "phoneNumberMobile": "99887766", "userType": "STANDARD", "department": {"id": DEPARTMENT_ID}}}
]
NOTE: ALWAYS include dateOfBirth if given. Include phoneNumberMobile, employeeNumber, nationalIdentityNumber, address if mentioned in prompt.

### Create employee with admin role (uses ENVIRONMENT values)
Prompt: "Opprett ansatt Kari Nordmann, kari@example.org. Hun skal være kontoadministrator."
[
  {"method": "POST", "path": "/employee", "body": {"firstName": "Kari", "lastName": "Nordmann", "email": "kari@example.org", "userType": "EXTENDED", "department": {"id": DEPARTMENT_ID}}},
  {"method": "POST", "path": "/employee/entitlement", "body": {"employee": {"id": "$PREV_0_ID"}, "entitlementId": 1, "customer": {"id": COMPANY_ID}}}
]
NOTE: Replace DEPARTMENT_ID and COMPANY_ID with values from ENVIRONMENT. Just 2 calls instead of 4!

### Create customer (INCLUDE ALL FIELDS from prompt!)
[
  {"method": "POST", "path": "/customer", "body": {"name": "Firma AS", "email": "post@firma.no", "organizationNumber": "912345678", "isCustomer": true, "phoneNumber": "22334455", "invoiceEmail": "faktura@firma.no", "postalAddress": {"addressLine1": "Storgata 10", "postalCode": "0182", "city": "Oslo"}}}
]
NOTE: Include ALL fields from prompt: postalAddress, phoneNumber, invoiceEmail, language ("NO"/"EN"). Every missing field = lost points!

### Create supplier: POST /supplier with name, email, organizationNumber
### Both customer+supplier: POST /customer with isCustomer:true, isSupplier:true

## Tier 2: Multi-step & modification tasks

### Create invoice with EXISTING customer and products (COMMON in competition!)
Prompt: "Lag faktura til kunde Montanha Lda (org.nr 860954737) med produkter Utvikling (7390) à 39600 NOK og Vedlikehold (6965) à 14550 NOK. Fakturer og registrer betaling."
IMPORTANT: When product NUMBERS are given, products ALREADY EXIST. Search by number, don't create!
Same for customers with organization numbers — search first!
[
  {"method": "GET", "path": "/customer", "params": {"organizationNumber": "860954737", "fields": "id,name"}},
  {"method": "GET", "path": "/product", "params": {"number": "7390", "fields": "id,name,priceExcludingVatCurrency"}},
  {"method": "GET", "path": "/product", "params": {"number": "6965", "fields": "id,name,priceExcludingVatCurrency"}},
  {"method": "POST", "path": "/order", "body": {"customer": {"id": "$PREV_0_ID"}, "deliveryDate": "2026-03-20", "orderDate": "2026-03-20"}},
  {"method": "POST", "path": "/order/orderline", "body": {"order": {"id": "$PREV_3_ID"}, "product": {"id": "$PREV_1_ID"}, "count": 1, "unitPriceExcludingVatCurrency": 39600.0, "vatType": {"id": 3}, "description": "Utvikling"}},
  {"method": "POST", "path": "/order/orderline", "body": {"order": {"id": "$PREV_3_ID"}, "product": {"id": "$PREV_2_ID"}, "count": 1, "unitPriceExcludingVatCurrency": 14550.0, "vatType": {"id": 3}, "description": "Vedlikehold"}},
  {"method": "PUT", "path": "/order/$PREV_3_ID/:invoice", "params": {"invoiceDate": "2026-03-20", "invoiceDueDate": "2026-04-19", "sendToCustomer": "false"}},
  {"method": "PUT", "path": "/invoice/$PREV_6_ID/:payment", "params": {"paymentDate": "2026-03-20", "paymentTypeId": "PAYMENT_TYPE_BANK_ID", "paidAmount": "$PREV_6_FIELD_amount"}}
]
NOTE: ALWAYS include invoiceDueDate (invoiceDate + 30 days). Products searched by number. Customer by organizationNumber.

### Register payment on EXISTING invoice (common!)
CRITICAL: GET /invoice REQUIRES invoiceDateFrom and invoiceDateTo params!
[
  {"method": "GET", "path": "/invoice", "params": {"invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "fields": "id,amount,amountOutstanding,customer(id,name)"}},
  {"method": "PUT", "path": "/invoice/$PREV_0_ID/:payment", "params": {"paymentDate": "2026-03-22", "paymentTypeId": "PAYMENT_TYPE_BANK_ID from ENVIRONMENT", "paidAmount": "$PREV_0_FIELD_amount"}}
]

### Update entity (GET fields=* → PUT with $MERGE_PREV)
Any entity: GET /entity?search_params&fields=* → PUT /entity/$PREV_0_ID with {"_merge": "$MERGE_PREV_0", ...changes}
For employee PUT: ALWAYS include dateOfBirth (use "1990-01-01" if not specified)

### Delete entity
GET entity first, then DELETE /entity/$PREV_ID. Works for /travelExpense, /customer, /product, /order (not invoiced).
NOT for /invoice (use credit note) or /employee (use PUT to deactivate).

### Create project with project manager (uses ENVIRONMENT values)
Prompt: "Opprett prosjekt Omega med ansatt Kari som prosjektleder"
[
  {"method": "POST", "path": "/employee", "body": {"firstName": "Kari", "lastName": "Nordmann", "email": "kari@example.org", "userType": "EXTENDED", "department": {"id": DEPARTMENT_ID}}},
  {"method": "POST", "path": "/employee/entitlement", "body": {"employee": {"id": "$PREV_0_ID"}, "entitlementId": 45, "customer": {"id": COMPANY_ID}}},
  {"method": "POST", "path": "/employee/entitlement", "body": {"employee": {"id": "$PREV_0_ID"}, "entitlementId": 10, "customer": {"id": COMPANY_ID}}},
  {"method": "POST", "path": "/project", "body": {"name": "Omega", "number": "1", "startDate": "2026-03-20", "projectManager": {"id": "$PREV_0_ID"}}}
]
NOTE: 4 calls instead of 6! DEPARTMENT_ID and COMPANY_ID from ENVIRONMENT.

### Full project cycle (Tier 3 — high value!)
For complex project tasks that require: create project → register hours → register supplier cost → create project invoice:
1. GET /customer by organizationNumber to find customer
2. POST /employee for each team member (or GET if they exist)
3. POST /employee/entitlement (45 then 10) for project manager
4. POST /project with customer.id and projectManager.id
5. POST /timesheet/entry for each employee's hours (use activity from ENVIRONMENT)
6. For supplier costs: POST /ledger/voucher with project.id on postings
7. POST /order with project.id → POST /order/orderline → PUT /order/:invoice
NOTE: Use GET to find existing entities FIRST. Search by organizationNumber or name.

## Tier 2: Supplier invoice (uses ENVIRONMENT account IDs)
### Register supplier invoice (as voucher — /supplierInvoice returns 500!)
Prompt: "Registrer leverandørfaktura INV-001 fra Staples på 12500 NOK inkl MVA for kontorrekvisita"
Step 1: Find the supplier (GET /supplier?name=Staples)
Step 2: Create voucher with supplier.id on EVERY posting
[
  {"method": "GET", "path": "/supplier", "params": {"name": "Staples", "fields": "id"}},
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Leverandørfaktura INV-001 fra Staples", "voucherType": {"id": VOUCHER_TYPE_SUPPLIER_ID}, "postings": [
    {"date": "2026-03-20", "account": {"id": ACCOUNT_6800_ID}, "amount": 10000.0, "amountCurrency": 10000.0, "amountGross": 10000.0, "amountGrossCurrency": 10000.0, "currency": {"id": 1}, "row": 1, "supplier": {"id": "$PREV_0_ID"}, "description": "Kontorrekvisita (netto)"},
    {"date": "2026-03-20", "account": {"id": ACCOUNT_2710_ID}, "amount": 2500.0, "amountCurrency": 2500.0, "amountGross": 2500.0, "amountGrossCurrency": 2500.0, "currency": {"id": 1}, "row": 2, "supplier": {"id": "$PREV_0_ID"}, "description": "Inngående MVA 25%"},
    {"date": "2026-03-20", "account": {"id": ACCOUNT_2400_ID}, "amount": -12500.0, "amountCurrency": -12500.0, "amountGross": -12500.0, "amountGrossCurrency": -12500.0, "currency": {"id": 1}, "row": 3, "supplier": {"id": "$PREV_0_ID"}, "description": "Leverandørgjeld"}
  ]}}
]
CRITICAL: supplier.id MUST be on EVERY posting! Without it you get "Leverandør mangler" error. Use POST /ledger/voucher, NOT /supplierInvoice (500).

## Tier 2: Timesheet
### Register hours on timesheet
Prompt: "Registrer 7.5 timer for ansatt Ole Hansen på prosjekt Alpha den 20. mars"
[
  {"method": "GET", "path": "/employee", "params": {"firstName": "Ole", "lastName": "Hansen", "fields": "id"}},
  {"method": "GET", "path": "/project", "params": {"name": "Alpha", "fields": "id"}},
  {"method": "POST", "path": "/timesheet/entry", "body": {"employee": {"id": "$PREV_0_ID"}, "project": {"id": "$PREV_1_ID"}, "activity": {"id": "FIRST_ACTIVITY_ID from ENVIRONMENT"}, "date": "2026-03-20", "hours": 7.5}}
]
NOTE: Use first activity_id from ENVIRONMENT. Only 3 calls.

### Register hours for logged-in user (uses ENVIRONMENT — 0 GET calls!)
Prompt: "Registrer 7.5 timer for den innloggede brukeren i dag"
[
  {"method": "POST", "path": "/timesheet/entry", "body": {"employee": {"id": "EMPLOYEE_ID from ENVIRONMENT"}, "activity": {"id": "FIRST_ACTIVITY_ID from ENVIRONMENT"}, "date": "2026-03-20", "hours": 7.5}}
]
NOTE: Use employee_id from ENVIRONMENT for "innlogget bruker"/"current user". Just 1 call!

## Tier 2: Payroll (use voucher, NOT salary module!)
### Run payroll — uses ENVIRONMENT directly, 0 GET calls!
Prompt: "Kjør lønn for Randi Haugen. Grunnlønn 49550 kr + engangsbonus 8300 kr"
IMPORTANT: POST /salary/payslip returns 403. Use POST /ledger/voucher with VOUCHER_TYPE_SALARY_ID from ENVIRONMENT!
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-22", "description": "Lønn mars 2026 - Randi Haugen (49550 + 8300 bonus)", "voucherType": {"id": "VOUCHER_TYPE_SALARY_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-22", "account": {"id": "ACCOUNT_5000_ID from ENVIRONMENT"}, "amount": 57850.0, "amountCurrency": 57850.0, "amountGross": 57850.0, "amountGrossCurrency": 57850.0, "currency": {"id": 1}, "row": 1, "description": "Fastlønn 49550 + bonus 8300"}, {"date": "2026-03-22", "account": {"id": "ACCOUNT_2930_ID from ENVIRONMENT"}, "amount": -57850.0, "amountCurrency": -57850.0, "amountGross": -57850.0, "amountGrossCurrency": -57850.0, "currency": {"id": 1}, "row": 2, "description": "Skyldig lønn"}]}}
]
NOTE: Just 1 write call! All IDs from ENVIRONMENT. SALARY_ID for lønnsbilag, ACCOUNT_5000_ID and ACCOUNT_2930_ID for accounts.

## Tier 2: Credit note
### Create credit note for existing invoice
Prompt: "Lag kreditnota for faktura til Acme AS for 'Consulting' (15000 NOK ekskl MVA)"
[
  {"method": "GET", "path": "/invoice", "params": {"customerName": "Acme AS", "invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "fields": "id,invoiceNumber,amount,customer(id,name)"}},
  {"method": "PUT", "path": "/invoice/$PREV_0_ID/:createCreditNote", "params": {"date": "2026-03-22", "comment": "Kreditnota", "sendToCustomer": "false"}}
]
NOTE: MUST include date param (today or after invoice date). Use invoiceDateFrom/To with wide range. creditNoteEmail is NOT a valid param.

## Tier 2: Purring / Reminder charge (purregebyr)
Prompt: "Kunden har en forfalt faktura. Registrer purregebyr 55 NOK. Debet kundefordring (1500), kredit purregebyr-inntekt (3400). Lag også purrefaktura og registrer betaling."
[
  {"method": "GET", "path": "/invoice", "params": {"invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "fields": "id,amount,amountOutstanding,customer(id,name)"}},
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Purregebyr", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_1500_ID from ENVIRONMENT"}, "amount": 55.0, "amountCurrency": 55.0, "amountGross": 55.0, "amountGrossCurrency": 55.0, "currency": {"id": 1}, "row": 1, "description": "Kundefordring purregebyr", "customer": "$PREV_0_FIELD_customer"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_3400_ID from ENVIRONMENT"}, "amount": -55.0, "amountCurrency": -55.0, "amountGross": -55.0, "amountGrossCurrency": -55.0, "currency": {"id": 1}, "row": 2, "description": "Purregebyr-inntekt", "customer": "$PREV_0_FIELD_customer"}]}},
  {"method": "PUT", "path": "/invoice/$PREV_0_ID/:createReminder", "params": {"date": "2026-03-20", "type": "remindersReminder", "sendToCustomer": "true"}}
]
NOTE: CRITICAL: customer.id MUST be on EVERY posting for Purring voucherType! /:createReminder needs "type" param (e.g. "remindersReminder"). Use account 3400 from ENVIRONMENT.

## Tier 2: Partial payment (delbetaling)
Prompt: "Kunden har betalt 10000 kr av en faktura på 25000 kr. Registrer delbetalingen."
[
  {"method": "GET", "path": "/invoice", "params": {"invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "fields": "id,amount,amountOutstanding"}},
  {"method": "PUT", "path": "/invoice/$PREV_0_ID/:payment", "params": {"paymentDate": "2026-03-20", "paymentTypeId": "PAYMENT_TYPE_BANK_ID", "paidAmount": "10000.0"}}
]
NOTE: paidAmount can be LESS than full amount for partial payments. Multiple partial payments are allowed.

## Tier 2: Payment reversal (betalingsreversering)
Prompt: "Betalingen ble returnert av banken. Reverser betalingen."
Use PUT /invoice/{id}/:payment with NEGATIVE paidAmount to reverse a payment:
[
  {"method": "GET", "path": "/invoice", "params": {"invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "fields": "id,amount,amountOutstanding"}},
  {"method": "PUT", "path": "/invoice/$PREV_0_ID/:payment", "params": {"paymentDate": "2026-03-20", "paymentTypeId": "PAYMENT_TYPE_BANK_ID", "paidAmount": "-$PREV_0_FIELD_amount"}}
]
NOTE: Use NEGATIVE amount to reverse. This restores the outstanding amount on the invoice.

## Tier 2: Supplier payment (leverandørbetaling)
Prompt: "Betal leverandørfaktura til Staples på 12500 kr fra bankkonto"
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Betaling leverandørfaktura Staples", "voucherType": {"id": "VOUCHER_TYPE_PAYMENT_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_2400_ID from ENVIRONMENT"}, "amount": 12500.0, "amountCurrency": 12500.0, "amountGross": 12500.0, "amountGrossCurrency": 12500.0, "currency": {"id": 1}, "row": 1, "description": "Betaling leverandørgjeld"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_1920_ID from ENVIRONMENT"}, "amount": -12500.0, "amountCurrency": -12500.0, "amountGross": -12500.0, "amountGrossCurrency": -12500.0, "currency": {"id": 1}, "row": 2, "description": "Fra bankkonto"}]}}
]

## Tier 2: Employee expense (ansattutlegg)
Prompt: "Ansatt Erik har lagt ut 3500 kr for kontorrekvisita. Registrer utlegget."
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Ansattutlegg kontorrekvisita", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_6800_ID from ENVIRONMENT"}, "amount": 3500.0, "amountCurrency": 3500.0, "amountGross": 3500.0, "amountGrossCurrency": 3500.0, "currency": {"id": 1}, "row": 1, "description": "Kontorrekvisita"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_2910_ID from ENVIRONMENT"}, "amount": -3500.0, "amountCurrency": -3500.0, "amountGross": -3500.0, "amountGrossCurrency": -3500.0, "currency": {"id": 1}, "row": 2, "description": "Gjeld til ansatt"}]}}
]

## Tier 2: Update product
Prompt: "Oppdater produktet Konsulenttjeneste med ny pris 1850 NOK ekskl. MVA"
[
  {"method": "GET", "path": "/product", "params": {"name": "Konsulenttjeneste", "fields": "*"}},
  {"method": "PUT", "path": "/product/$PREV_0_ID", "body": {"_merge": "$MERGE_PREV_0", "priceExcludingVatCurrency": 1850.0}}
]

## Tier 2: Create departments (batch)
Prompt: "Opprett avdelingene Økonomi (nr 100), Salg (nr 200) og IT (nr 300)"
[
  {"method": "POST", "path": "/department/list", "body": [{"name": "Økonomi", "departmentNumber": "100"}, {"name": "Salg", "departmentNumber": "200"}, {"name": "IT", "departmentNumber": "300"}]}
]
NOTE: Use /department/list for batch creation — 1 write call instead of 3!

## Tier 2: Private individual customer
Prompt: "Opprett privatkunden Ole Brumm med e-post ole@privat.no"
[
  {"method": "POST", "path": "/customer", "body": {"name": "Ole Brumm", "email": "ole@privat.no", "isCustomer": true, "isPrivateIndividual": true}}
]

## Tier 3: VAT return (MVA-oppgjør)
Prompt: "Registrer MVA-oppgjør for Q1 2026. Utgående MVA 85000 kr, inngående MVA 32000 kr."
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-31", "description": "MVA-oppgjør Q1 2026", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-31", "account": {"id": "ACCOUNT_2700_ID from ENVIRONMENT"}, "amount": 85000.0, "amountCurrency": 85000.0, "amountGross": 85000.0, "amountGrossCurrency": 85000.0, "currency": {"id": 1}, "row": 1, "description": "Utgående MVA"}, {"date": "2026-03-31", "account": {"id": "ACCOUNT_2710_ID from ENVIRONMENT"}, "amount": -32000.0, "amountCurrency": -32000.0, "amountGross": -32000.0, "amountGrossCurrency": -32000.0, "currency": {"id": 1}, "row": 2, "description": "Inngående MVA"}, {"date": "2026-03-31", "account": {"id": "ACCOUNT_1920_ID from ENVIRONMENT"}, "amount": -53000.0, "amountCurrency": -53000.0, "amountGross": -53000.0, "amountGrossCurrency": -53000.0, "currency": {"id": 1}, "row": 3, "description": "MVA-betaling til skatteetaten"}]}}
]
NOTE: Utgående MVA (debet 2700) - Inngående MVA (kredit 2710) = nettobetaling fra bank (kredit 1920). Beløpene MÅ balansere.

## Tier 3: Register fixed asset (anleggsmiddel)
Prompt: "Registrer en maskin til 250000 kr anskaffet 01.01.2026 med levetid 60 måneder"
[
  {"method": "POST", "path": "/asset", "body": {"name": "Maskin", "acquisitionCost": 250000.0, "acquisitionDate": "2026-01-01", "lifetimeInMonths": 60}}
]
NOTE: After creating the asset, depreciation is registered via POST /ledger/voucher (debit 6010, credit 1200).

## Misc: Delete entities
### Delete customer/product/supplier
Prompt: "Slett kunden TestFirma AS"
[
  {"method": "GET", "path": "/customer", "params": {"name": "TestFirma AS", "fields": "id"}},
  {"method": "DELETE", "path": "/customer/$PREV_0_ID"}
]
NOTE: Works for /customer, /product, /order (not invoiced), /travelExpense. NOT for /invoice (use credit note) or /employee (403).

## VOUCHER FALLBACK — For oppgaver uten spesifikt endepunkt
If you cannot find a specific API endpoint for the task, use POST /ledger/voucher with the correct voucherType and account IDs from ENVIRONMENT. ALL accounting operations are ultimately debit/credit postings.

Common voucher patterns (use account IDs from ENVIRONMENT):
| Operation | Debit account | Credit account | VoucherType |
|-----------|--------------|----------------|-------------|
| Lønn (salary) | 5000 Lønn | 2930 Skyldig lønn | Lønnsbilag |
| Skattetrekk | 2930 Skyldig lønn | 2600 Forskuddstrekk | Lønnsbilag |
| Arbeidsgiveravgift | 5400 Arb.avg. | 2770 Skyldig arb.avg. | Lønnsbilag |
| Avskrivning | 6010 Avskrivning | 1200 Maskiner | Manuelt bilag |
| Periodisering (forskudd) | 1700 Forskuddsbetalt | 1920 Bank | Manuelt bilag |
| Periodisering (kostnad) | 6800 Kontorrekvisita | 1700 Forskuddsbetalt | Manuelt bilag |
| Bankgebyr | 7770 Bankgebyr | 1920 Bank | Manuelt bilag |
| Ansattutlegg | 7140 Reisekostnad | 2910 Gjeld ansatte | Manuelt bilag |

CRITICAL RULES for voucher postings:
- Every posting MUST have "row" field (integer, starting at 1)
- Postings MUST balance (sum of amounts = 0). Positive = debit, negative = credit.
- Use account IDs from ENVIRONMENT, NOT account numbers directly.

## Tier 3: Avskrivning (depreciation)
Prompt: "Avskriv kontorutstyr til verdi 50000 kr med 20% lineær avskrivning"
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Avskrivning kontorutstyr 20%", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_6010_ID from ENVIRONMENT"}, "amount": 10000.0, "amountCurrency": 10000.0, "amountGross": 10000.0, "amountGrossCurrency": 10000.0, "currency": {"id": 1}, "row": 1, "description": "Avskrivning 20% av 50000"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_1200_ID from ENVIRONMENT"}, "amount": -10000.0, "amountCurrency": -10000.0, "amountGross": -10000.0, "amountGrossCurrency": -10000.0, "currency": {"id": 1}, "row": 2, "description": "Reduksjon anleggsmidler"}]}}
]

## Tier 3: Periodisering (prepaid expense)
Prompt: "Forskuddsbetal 12000 kr for årlig forsikring, periodiser månedlig"
Step 1 — Betaling: Voucher med 1700 (forskuddsbetalt) debit / 1920 (bank) kredit
Step 2 — Månedlig kostnad: Voucher med 6800 (kostnad) debit / 1700 (forskuddsbetalt) kredit (1000 kr per mnd)
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Forskuddsbetalt forsikring 12 mnd", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_1700_ID from ENVIRONMENT"}, "amount": 12000.0, "amountCurrency": 12000.0, "amountGross": 12000.0, "amountGrossCurrency": 12000.0, "currency": {"id": 1}, "row": 1, "description": "Forskuddsbetalt forsikring"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_1920_ID from ENVIRONMENT"}, "amount": -12000.0, "amountCurrency": -12000.0, "amountGross": -12000.0, "amountGrossCurrency": -12000.0, "currency": {"id": 1}, "row": 2, "description": "Utbetaling fra bank"}]}},
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Periodisering forsikring mars", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_6800_ID from ENVIRONMENT"}, "amount": 1000.0, "amountCurrency": 1000.0, "amountGross": 1000.0, "amountGrossCurrency": 1000.0, "currency": {"id": 1}, "row": 1, "description": "Forsikringskostnad mars"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_1700_ID from ENVIRONMENT"}, "amount": -1000.0, "amountCurrency": -1000.0, "amountGross": -1000.0, "amountGrossCurrency": -1000.0, "currency": {"id": 1}, "row": 2, "description": "Reduksjon forskuddsbetalt"}]}}
]

## Tier 3: Bankgebyr
Prompt: "Registrer bankgebyr på 250 kr"
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Bankgebyr mars 2026", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_7770_ID from ENVIRONMENT"}, "amount": 250.0, "amountCurrency": 250.0, "amountGross": 250.0, "amountGrossCurrency": 250.0, "currency": {"id": 1}, "row": 1, "description": "Bankgebyr"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_1920_ID from ENVIRONMENT"}, "amount": -250.0, "amountCurrency": -250.0, "amountGross": -250.0, "amountGrossCurrency": -250.0, "currency": {"id": 1}, "row": 2, "description": "Fra bankkonto"}]}}
]

## Tier 3: Utvidet lønn (grunnlønn + skattetrekk + arbeidsgiveravgift)
Prompt: "Kjør lønn for ansatt med grunnlønn 45000, skattetrekk 35%, og arbeidsgiveravgift 14.1%"
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Lønnsbilag mars 2026", "voucherType": {"id": "VOUCHER_TYPE_SALARY_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_5000_ID from ENVIRONMENT"}, "amount": 45000.0, "amountCurrency": 45000.0, "amountGross": 45000.0, "amountGrossCurrency": 45000.0, "currency": {"id": 1}, "row": 1, "description": "Grunnlønn"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_2930_ID from ENVIRONMENT"}, "amount": -29250.0, "amountCurrency": -29250.0, "amountGross": -29250.0, "amountGrossCurrency": -29250.0, "currency": {"id": 1}, "row": 2, "description": "Netto lønn (etter skatt)"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_2600_ID from ENVIRONMENT"}, "amount": -15750.0, "amountCurrency": -15750.0, "amountGross": -15750.0, "amountGrossCurrency": -15750.0, "currency": {"id": 1}, "row": 3, "description": "Skattetrekk 35%"}]}},
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Arbeidsgiveravgift mars 2026", "voucherType": {"id": "VOUCHER_TYPE_SALARY_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_5400_ID from ENVIRONMENT"}, "amount": 6345.0, "amountCurrency": 6345.0, "amountGross": 6345.0, "amountGrossCurrency": 6345.0, "currency": {"id": 1}, "row": 1, "description": "Arbeidsgiveravgift 14.1%"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_2770_ID from ENVIRONMENT"}, "amount": -6345.0, "amountCurrency": -6345.0, "amountGross": -6345.0, "amountGrossCurrency": -6345.0, "currency": {"id": 1}, "row": 2, "description": "Skyldig arbeidsgiveravgift"}]}}
]
NOTE: Two vouchers — one for salary (lønn-skatt-netto), one for employer tax (AGA). Amounts MUST balance per voucher.

## Tier 3: Feilretting i hovedbok (error correction in general ledger)
Prompt: "Vi har oppdaget feil i hovedboka. Konto 6860 ble brukt i stedet for 6590 (beløp 5100 kr). Rett feilen."
For EACH error, create a correction voucher that reverses the wrong posting and adds the correct one.
ALL account IDs are in ENVIRONMENT — use account_XXXX_id directly! NO need for GET /ledger/account!
IMPORTANT: Use voucher_type_manual_id from ENVIRONMENT for ALL correction vouchers!
[
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-22", "description": "Korreksjon: Flyttet fra konto 6860 til 6590", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-22", "account": {"id": "ACCOUNT_6590_ID from ENVIRONMENT"}, "amount": 5100.0, "amountCurrency": 5100.0, "amountGross": 5100.0, "amountGrossCurrency": 5100.0, "currency": {"id": 1}, "row": 1, "description": "Korreksjon debit riktig konto 6590"}, {"date": "2026-03-22", "account": {"id": "ACCOUNT_6860_ID from ENVIRONMENT"}, "amount": -5100.0, "amountCurrency": -5100.0, "amountGross": -5100.0, "amountGrossCurrency": -5100.0, "currency": {"id": 1}, "row": 2, "description": "Korreksjon kredit feil konto 6860"}]}}
]
NOTE: Use account IDs from ENVIRONMENT directly! Just 1 write call, no GET needed. Use account_XXXX_id format.

## Update and delete — additional notes
- For department search: use "query" parameter (not "name") — it's more robust.
- DELETE /project may return 422 if project has orders/vouchers attached.

## Tier 3: Accounting dimensions (fri regnskapsdimensjon)
Prompt: "Opprett en fri regnskapsdimensjon kalt 'Region' med verdiene 'Nord', 'Sør' og 'Vest'. Bokfør 15000 kr på konto 6800 fordelt på region Nord."
[
  {"method": "POST", "path": "/ledger/accountingDimensionName", "body": {"dimensionName": "Region"}},
  {"method": "POST", "path": "/ledger/accountingDimensionValue", "body": {"displayName": "Nord"}},
  {"method": "POST", "path": "/ledger/accountingDimensionValue", "body": {"displayName": "Sør"}},
  {"method": "POST", "path": "/ledger/accountingDimensionValue", "body": {"displayName": "Vest"}},
  {"method": "POST", "path": "/ledger/voucher", "body": {"date": "2026-03-20", "description": "Bokføring kontorrekvisita for Region Nord", "voucherType": {"id": "VOUCHER_TYPE_MANUAL_ID from ENVIRONMENT"}, "postings": [{"date": "2026-03-20", "account": {"id": "ACCOUNT_6800_ID from ENVIRONMENT"}, "amount": 15000.0, "amountCurrency": 15000.0, "amountGross": 15000.0, "amountGrossCurrency": 15000.0, "currency": {"id": 1}, "row": 1, "description": "Kontorrekvisita Region Nord"}, {"date": "2026-03-20", "account": {"id": "ACCOUNT_1920_ID from ENVIRONMENT"}, "amount": -15000.0, "amountCurrency": -15000.0, "amountGross": -15000.0, "amountGrossCurrency": -15000.0, "currency": {"id": 1}, "row": 2, "description": "Betalt fra bank"}]}}
]
NOTE: Use /ledger/accountingDimensionName + /ledger/accountingDimensionValue for dimensions. These are REAL API endpoints!

## Tier 3: Reverse existing voucher (feilretting alternativ)
Prompt: "Reverser bilag nr 5 — det ble ført feil"
[
  {"method": "PUT", "path": "/ledger/voucher/5/:reverse", "params": {"date": "2026-03-20"}}
]
NOTE: PUT /:reverse creates a counter-voucher automatically. Much simpler than manual correction! Only 1 call.

## Tier 3: Opening balance (åpningsbalanse)
Prompt: "Sett åpningsbalanse med 500000 kr på bankkonto og 500000 kr i egenkapital"
[
  {"method": "POST", "path": "/ledger/voucher/openingBalance", "body": {"date": "2026-01-01", "description": "Åpningsbalanse", "postings": [{"account": {"id": "ACCOUNT_1920_ID from ENVIRONMENT"}, "amount": 500000.0, "amountCurrency": 500000.0, "amountGross": 500000.0, "amountGrossCurrency": 500000.0, "currency": {"id": 1}, "row": 1, "description": "Bankinnskudd"}, {"account": {"id": ACCOUNT_2050_ID}, "amount": -500000.0, "amountCurrency": -500000.0, "amountGross": -500000.0, "amountGrossCurrency": -500000.0, "currency": {"id": 1}, "row": 2, "description": "Egenkapital"}]}}
]
NOTE: Use the dedicated /ledger/voucher/openingBalance endpoint. POST creates, DELETE removes existing.

## Tier 2: Mileage allowance (kjøregodtgjørelse)
Prompt: "Registrer kjøregodtgjørelse for 150 km fra Oslo til Drammen"
[
  {"method": "POST", "path": "/travelExpense", "body": {"employee": {"id": "EMPLOYEE_ID from ENVIRONMENT"}, "title": "Kjøring Oslo-Drammen", "date": "2026-03-20"}},
  {"method": "GET", "path": "/travelExpense/rateCategory", "params": {"fields": "id,name"}},
  {"method": "POST", "path": "/travelExpense/mileageAllowance", "body": {"travelExpense": {"id": "$PREV_0_ID"}, "rateCategory": {"id": "$PREV_1_ID"}, "date": "2026-03-20", "departureLocation": "Oslo", "destination": "Drammen", "km": 150}}
]
"""


def clean_json_text(text):
    """Clean JSON text from Gemini: remove comments, trailing commas, control chars, extract JSON."""
    # Extract from markdown code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                text = candidate
                break

    # Find JSON array or object if response has extra text
    if not text.startswith("[") and not text.startswith("{"):
        start_bracket = text.find("[")
        start_brace = text.find("{")
        if start_bracket >= 0 and (start_brace < 0 or start_bracket < start_brace):
            end = text.rfind("]")
            if end > start_bracket:
                text = text[start_bracket:end+1]
        elif start_brace >= 0:
            end = text.rfind("}")
            if end > start_brace:
                text = text[start_brace:end+1]

    # Remove control characters (except newline, tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Remove single-line comments (// ...) that are NOT inside strings
    # Strategy: remove lines that are only comments, and trailing comments after JSON values
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('//'):
            continue  # Skip full-line comments
        # Remove trailing comments (after JSON value) — simple heuristic
        # Only remove if // appears after a JSON value character and not inside a string
        in_string = False
        escape_next = False
        comment_pos = None
        for ci, ch in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
            if not in_string and ci < len(line) - 1 and line[ci] == '/' and line[ci+1] == '/':
                comment_pos = ci
                break
        if comment_pos is not None:
            line = line[:comment_pos].rstrip()
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    return text.strip()


def validate_calls(calls):
    """Validate Gemini-generated API calls before execution. Returns (valid_calls, warnings)."""
    valid_methods = {"GET", "POST", "PUT", "DELETE"}
    validated = []
    warnings = []

    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            warnings.append(f"Kall {i}: Ikke et gyldig objekt, hoppet over")
            continue

        method = call.get("method", "GET").upper()
        path = call.get("path", "")

        # Sjekk at metoden er gyldig
        if method not in valid_methods:
            warnings.append(f"Kall {i}: Ugyldig metode '{method}', endret til GET")
            call["method"] = "GET"

        # Sjekk at path starter med /
        if path and not path.startswith("/") and not path.startswith("$"):
            warnings.append(f"Kall {i}: Path '{path}' mangler ledende /, lagt til")
            call["path"] = "/" + path

        # Valider $PREV_N_ID referanser — sjekk at N er rimelig
        path_str = json.dumps(call)
        prev_refs = re.findall(r'\$PREV_(\d+)_', path_str)
        merge_refs = re.findall(r'\$MERGE_PREV_(\d+)', path_str)

        for ref_idx_str in prev_refs + merge_refs:
            ref_idx = int(ref_idx_str)
            if ref_idx >= i:
                warnings.append(
                    f"Kall {i}: Refererer til $PREV_{ref_idx} men er bare kall nr {i} "
                    f"(kan bare referere til 0-{i-1})"
                )
            if ref_idx >= len(calls):
                warnings.append(
                    f"Kall {i}: Refererer til $PREV_{ref_idx} men det finnes bare {len(calls)} kall totalt"
                )

        validated.append(call)

    if warnings:
        logger.warning(f"Validering fant {len(warnings)} advarsler:")
        for w in warnings:
            logger.warning(f"  {w}")

    return validated, warnings


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
                    # If field value is a dict with "id", extract just the id
                    # This handles cases like customer: {"id": 123, "url": "..."} → 123
                    if isinstance(field_val, dict) and "id" in field_val:
                        field_val = field_val["id"]
                    if value == field_match.group(0):
                        return field_val
                    # Handle Gemini's "$PREV_0_FIELD_customer.id" pattern
                    # After replacing $PREV_0_FIELD_customer with the value,
                    # ".id" remains as string suffix — strip it
                    result = value.replace(field_match.group(0), str(field_val))
                    if result == f"{field_val}.id":
                        return field_val
                    return result

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


def execute_api_calls(calls, base_url, session_token, original_prompt="", env_info=None):
    """Execute a sequence of Tripletex API calls."""
    auth = ("0", session_token)
    results = []
    skip_remaining = 0  # Number of original calls to skip (handled by retry)

    for i, call in enumerate(calls):
        # Skip calls that were already handled by a retry replan
        if skip_remaining > 0:
            skip_remaining -= 1
            logger.info(f"Call {i}: SKIPPED (handled by retry replan)")
            continue

        method = call.get("method", "GET").upper()
        path = call.get("path", "")
        params = call.get("params", {})
        body = call.get("body", {})

        path = resolve_placeholders(path, results)
        # Fix ".id" suffix in paths — Gemini writes $PREV_N_FIELD_voucher.id → "123.id"
        path = re.sub(r'(\d+)\.id\b', r'\1', path)
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

        # VALIDATION: Fix voucher postings where all rows use the same account
        # Use dynamic account map from env_info instead of previous GET results
        if method == "POST" and "/ledger/voucher" in path and body.get("postings"):
            postings = body["postings"]
            account_ids = [p.get("account", {}).get("id") for p in postings if isinstance(p, dict)]
            if len(set(account_ids)) == 1 and len(account_ids) >= 2 and env_info:
                # All postings use the same account — this is wrong!
                # Use the dynamic account map (ONLY actual ledger accounts, not customers/products)
                account_map = env_info.get("all_account_map", {})
                if account_map:
                    for p_idx, posting in enumerate(postings):
                        desc = posting.get("description", "").lower()
                        best_match_id = None
                        # Look for account number in description
                        for acc_num, acc_info in account_map.items():
                            if acc_num in desc or acc_info.get("name", "").lower() in desc:
                                best_match_id = acc_info["id"]
                                break
                        if best_match_id:
                            old_id = posting.get("account", {}).get("id")
                            if old_id != best_match_id:
                                posting["account"]["id"] = best_match_id
                                logger.info(f"  Fixed posting {p_idx}: account {old_id} → {best_match_id}")

        # VALIDATION: Add supplier.id to voucher postings if "Leverandør mangler"
        # Tripletex requires supplier.id on postings when voucherType is supplier-related
        if method == "POST" and "/ledger/voucher" in path and body.get("postings"):
            vt_id = body.get("voucherType", {}).get("id")
            supplier_vt = env_info.get("voucher_type_supplier_id") if env_info else None
            # Check if any posting already has supplier — if not, try to find from previous GET results
            postings = body["postings"]
            has_supplier = any(p.get("supplier") for p in postings if isinstance(p, dict))
            if not has_supplier:
                # Look for supplier ID in previous results
                found_supplier_id = None
                for r in results:
                    if r and isinstance(r, dict):
                        obj = r.get("value") or (r.get("values", [None])[0] if r.get("values") else None)
                        if obj and isinstance(obj, dict):
                            if "organizationNumber" in obj and obj.get("id"):
                                # Check if this looks like a supplier (has org number, came from supplier search)
                                found_supplier_id = obj["id"]
                if found_supplier_id and vt_id == supplier_vt:
                    for p in postings:
                        if isinstance(p, dict) and not p.get("supplier"):
                            p["supplier"] = {"id": found_supplier_id}
                    logger.info(f"  Added supplier.id={found_supplier_id} to {len(postings)} postings")

        # VALIDATION: Auto-add dateOfBirth to employee POST/PUT if missing
        if method in ("POST", "PUT") and "/employee" in path and "employment" not in path and "entitlement" not in path:
            if isinstance(body, dict) and "firstName" in body and "dateOfBirth" not in body:
                body["dateOfBirth"] = "1990-01-01"
                logger.info("  Auto-added dateOfBirth=1990-01-01")

        # VALIDATION: Ensure nested ID fields are integers, not strings
        # Fixes "Verdien er ikke av korrekt type" errors on supplier.id, product.id, etc.
        if method in ("POST", "PUT") and body:
            def fix_id_types(obj, path=""):
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        if isinstance(val, dict):
                            fix_id_types(val, current_path)
                        elif isinstance(val, list):
                            for idx, item in enumerate(val):
                                if isinstance(item, dict):
                                    fix_id_types(item, f"{current_path}[{idx}]")
                        elif key == "id" and isinstance(val, str):
                            try:
                                obj[key] = int(val)
                                logger.info(f"  fix_id_types: {current_path} '{val}' → {obj[key]}")
                            except (ValueError, TypeError):
                                logger.warning(f"  fix_id_types: {current_path} has unresolvable string '{val}'")
                        elif key == "id" and not isinstance(val, (int, float)):
                            logger.warning(f"  fix_id_types: {current_path} unexpected type {type(val).__name__}: {val}")
                return obj
            fix_id_types(body)

        # Skip calls with unresolved $PREV references in path — they will always 404
        if "$PREV" in path or "$RETRY" in path:
            logger.warning(f"Call {i}: SKIPPED — unresolved reference in path: {path}")
            results.append(None)
            continue

        url = f"{base_url}{path}"
        logger.info(f"Call {i}: {method} {url}")
        if params:
            logger.info(f"  Params: {json.dumps(params)[:500]}")
        if body:
            logger.info(f"  Body: {json.dumps(body)[:1000]}")

        try:
            if method == "GET":
                resp = http_requests.get(url, auth=auth, params=params, timeout=30)
            elif method == "POST":
                resp = http_requests.post(url, auth=auth, json=body, timeout=30)
            elif method == "PUT":
                # Special PUT endpoints use query params instead of body
                # e.g. /order/{id}/:invoice, /invoice/{id}/:payment, /invoice/{id}/:createCreditNote
                if any(action in path for action in ['/:invoice', '/:payment', '/:createCreditNote', '/:createReminder', '/:send', '/:reverse', '/:createVouchers', '/:closePostings']):
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
                error_text = resp.text[:1000]
                logger.warning(f"  → Error: {error_text[:500]}")
                results.append({"error": resp.status_code, "detail": error_text})

                # Try to fix with Gemini on validation errors
                if resp.status_code in (400, 422):
                    fix = try_fix_call(
                        call, error_text, base_url, auth, results,
                        all_calls=calls, call_index=i,
                        original_prompt=original_prompt, env_info=env_info
                    )
                    if fix and isinstance(fix, dict) and "first_result" in fix:
                        # Replace the failed result with the fixed one
                        first = fix["first_result"]
                        if first and not (isinstance(first, dict) and "error" in first):
                            results[-1] = first
                        elif first:
                            results[-1] = first

                        # Append remaining retry results and skip corresponding original calls
                        remaining = fix.get("remaining_results", [])
                        if remaining:
                            results.extend(remaining)
                            # Skip the remaining original calls that were replanned
                            skip_remaining = len(calls) - (i + 1)
                            logger.info(
                                f"  → Retry replanned {len(remaining)} additional calls, "
                                f"skipping {skip_remaining} original remaining calls"
                            )
            else:
                try:
                    results.append(resp.json())
                except Exception:
                    results.append({"status_code": resp.status_code})

        except Exception as e:
            logger.error(f"  → Exception: {e}")
            results.append(None)

    return results


def try_fix_call(original_call, error_text, base_url, auth, results,
                  all_calls=None, call_index=0, original_prompt="", env_info=None):
    """Ask Gemini to replan the remaining API calls given full context.

    Sends: original task, results so far, the failed call + error, and remaining planned calls.
    Gemini returns a JSON array of corrected/remaining calls to execute.
    """
    # Build context of what has succeeded so far
    results_summary = []
    for ri, r in enumerate(results):
        if r is None:
            results_summary.append(f"Call {ri}: no result")
        elif isinstance(r, dict) and "error" in r:
            results_summary.append(f"Call {ri}: ERROR {r['error']} — {r.get('detail', '')[:200]}")
        elif isinstance(r, dict):
            # Summarize successful result
            if "value" in r:
                val = r["value"]
                summary_fields = {k: v for k, v in val.items()
                                  if k in ("id", "name", "firstName", "lastName", "email",
                                           "amount", "status", "number", "version")}
                results_summary.append(f"Call {ri}: OK — {json.dumps(summary_fields)}")
            elif "values" in r:
                count = len(r.get("values", []))
                first_id = r["values"][0].get("id") if r["values"] else None
                results_summary.append(f"Call {ri}: OK — {count} results, first id={first_id}")
            else:
                results_summary.append(f"Call {ri}: OK — {json.dumps(r)[:150]}")
        else:
            results_summary.append(f"Call {ri}: OK — {str(r)[:150]}")

    # Build remaining calls context
    remaining_calls = []
    if all_calls and call_index + 1 < len(all_calls):
        remaining_calls = all_calls[call_index + 1:]

    # Build environment context for retry
    env_context = ""
    if env_info:
        key_ids = {k: v for k, v in env_info.items() if isinstance(v, (int, float)) and not k.startswith("all_")}
        env_context = f"\n\nAVAILABLE ENVIRONMENT IDs (use directly, no GET needed):\n{json.dumps(key_ids, indent=1)[:2000]}"

    fix_prompt = f"""You are fixing a failed Tripletex API call sequence.
CRITICAL: All id fields MUST be integers (not strings). All amounts MUST be numbers.

ORIGINAL TASK: {original_prompt[:2000]}{env_context}

RESULTS SO FAR (calls 0 to {len(results)-2} succeeded, call {call_index} failed):
{chr(10).join(results_summary[:-1]) if len(results_summary) > 1 else "No previous calls."}

FAILED CALL (index {call_index}):
{json.dumps(original_call, indent=2)}

ERROR:
{error_text[:1000]}

REMAINING PLANNED CALLS AFTER THE FAILED ONE:
{json.dumps(remaining_calls, indent=2) if remaining_calls else "None — this was the last call."}

Return a JSON ARRAY of corrected API calls to execute NOW (the fixed version of the failed call + any remaining calls needed to complete the task).
Use $PREV_N_ID where N refers to the ORIGINAL call indices (0-based from the start of the full sequence).
Also use $RETRY_N_ID to reference the Nth call in YOUR returned array (0-based).
IMPORTANT: If a POST fails because an entity already exists (422 "already exists" / "i bruk"), use GET to find the existing entity instead of creating a new one. This avoids wasting write calls.

IMPORTANT: Return ONLY a valid JSON array. No markdown, no explanation, no comments."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=fix_prompt,
            config={"temperature": 0.1, "max_output_tokens": 8192},
        )
        text = clean_json_text(response.text.strip())
        logger.info(f"  → Fix response length: {len(text)}")

        fixed_calls = json.loads(text)
        if isinstance(fixed_calls, dict):
            fixed_calls = [fixed_calls]
        if not isinstance(fixed_calls, list) or len(fixed_calls) == 0:
            logger.warning("  → Fix returned empty or invalid response")
            return None

        logger.info(f"  → Fix planned {len(fixed_calls)} replacement calls")

        # Execute the fixed calls
        retry_results = []
        all_available_results = list(results)  # Copy of results including the error

        for fi, fixed_call in enumerate(fixed_calls):
            method = fixed_call.get("method", "GET").upper()
            path = fixed_call.get("path", "")
            params = fixed_call.get("params", {})
            body = fixed_call.get("body", {})

            # Resolve $PREV_N references against original results
            path = resolve_placeholders(path, all_available_results)
            params = resolve_placeholders(params, all_available_results)
            body = resolve_placeholders(body, all_available_results)

            # Resolve $RETRY_N_ID references against retry_results
            def resolve_retry_refs(value):
                if isinstance(value, str):
                    retry_match = re.search(r'\$RETRY_(\d+)_ID', value)
                    if retry_match:
                        ridx = int(retry_match.group(1))
                        if ridx < len(retry_results) and retry_results[ridx]:
                            rr = retry_results[ridx]
                            rid = None
                            if isinstance(rr, dict):
                                if "value" in rr:
                                    rid = rr["value"].get("id")
                                elif "values" in rr and rr["values"]:
                                    rid = rr["values"][0].get("id")
                                elif "id" in rr:
                                    rid = rr["id"]
                            if rid is not None:
                                if value == retry_match.group(0):
                                    return rid
                                return value.replace(retry_match.group(0), str(rid))
                    return value
                elif isinstance(value, dict):
                    return {k: resolve_retry_refs(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [resolve_retry_refs(v) for v in value]
                return value

            path = resolve_retry_refs(path)
            params = resolve_retry_refs(params)
            body = resolve_retry_refs(body)

            # Handle $MERGE_PREV_N for PUT
            merge_pattern = r'\$MERGE_PREV_(\d+)'
            body_str = json.dumps(body)
            merge_match = re.search(merge_pattern, body_str)
            if merge_match and method == "PUT":
                idx = int(merge_match.group(1))
                source = all_available_results if idx < len(all_available_results) else retry_results
                source_idx = idx if idx < len(all_available_results) else idx - len(all_available_results)
                if source_idx < len(source) and source[source_idx]:
                    base_obj = None
                    if "value" in source[source_idx]:
                        base_obj = source[source_idx]["value"]
                    elif "values" in source[source_idx] and source[source_idx]["values"]:
                        base_obj = source[source_idx]["values"][0]
                    if base_obj:
                        merged = dict(base_obj)
                        for k, v in body.items():
                            if isinstance(v, str) and re.match(merge_pattern, v):
                                continue
                            merged[k] = v
                        body = merged

            # Fix ID types in retry calls too (string → int)
            if method in ("POST", "PUT") and body:
                def fix_id_types_retry(obj):
                    if isinstance(obj, dict):
                        for key, val in obj.items():
                            if isinstance(val, dict):
                                fix_id_types_retry(val)
                            elif isinstance(val, list):
                                for item in val:
                                    if isinstance(item, dict):
                                        fix_id_types_retry(item)
                            elif key == "id" and isinstance(val, str):
                                try:
                                    obj[key] = int(val)
                                except (ValueError, TypeError):
                                    pass
                    return obj
                fix_id_types_retry(body)

            url = f"{base_url}{path}"
            logger.info(f"  → Retry {fi}: {method} {url}")
            if body:
                logger.info(f"    Body: {json.dumps(body)[:1000]}")

            try:
                if method == "GET":
                    resp = http_requests.get(url, auth=auth, params=params, timeout=30)
                elif method == "POST":
                    resp = http_requests.post(url, auth=auth, json=body, timeout=30)
                elif method == "PUT":
                    if any(action in path for action in
                           ['/:invoice', '/:payment', '/:createCreditNote', '/:createReminder', '/:send', '/:reverse', '/:createVouchers', '/:closePostings']):
                        resp = http_requests.put(url, auth=auth, params=params, timeout=30)
                    else:
                        resp = http_requests.put(url, auth=auth, json=body, timeout=30)
                elif method == "DELETE":
                    resp = http_requests.delete(url, auth=auth, timeout=30)
                else:
                    retry_results.append(None)
                    continue

                logger.info(f"  → Retry {fi}: {resp.status_code}")

                if resp.status_code >= 400:
                    error = resp.text[:500]
                    logger.warning(f"  → Retry {fi} error: {error}")
                    retry_results.append({"error": resp.status_code, "detail": error})
                else:
                    try:
                        retry_results.append(resp.json())
                    except Exception:
                        retry_results.append({"status_code": resp.status_code})

            except Exception as e:
                logger.error(f"  → Retry {fi} exception: {e}")
                retry_results.append(None)

        # Return first successful result (to replace the failed call in results)
        # Also return remaining results to be appended
        first_result = retry_results[0] if retry_results else None
        remaining = retry_results[1:] if len(retry_results) > 1 else []

        return {"first_result": first_result, "remaining_results": remaining}

    except json.JSONDecodeError as e:
        logger.error(f"  → Fix JSON parse error: {e}")
        logger.error(f"  → Fix text: {text[:500] if 'text' in dir() else 'N/A'}")
    except Exception as e:
        logger.error(f"  → Fix failed: {e}")
    return None


def extract_file_content(files):
    """Extract text content and multimodal parts from attached files.
    Returns (text_descriptions, multimodal_parts) where multimodal_parts
    are genai Part objects for PDFs/images that Gemini can read directly."""
    from google.genai import types

    descriptions = []
    parts = []
    for f in files:
        filename = f.get("filename", "unknown")
        mime = f.get("mime_type", "")
        try:
            data = base64.b64decode(f["content_base64"])
            if "pdf" in mime or "image" in mime:
                # Send directly to Gemini as multimodal content
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                descriptions.append(f"[Attached {mime}: {filename}, {len(data)} bytes — content sent to you as multimodal input]")
                logger.info(f"  File {filename}: sending as multimodal ({mime}, {len(data)} bytes)")
            elif "csv" in mime or "text" in mime or "xml" in mime:
                try:
                    text = data.decode("utf-8")[:5000]
                    descriptions.append(f"[File: {filename}]\n{text}")
                except Exception:
                    descriptions.append(f"[Binary: {filename}, {len(data)} bytes]")
            else:
                try:
                    text = data.decode("utf-8")[:2000]
                    descriptions.append(f"[File: {filename}]\n{text}")
                except Exception:
                    descriptions.append(f"[Binary: {filename}, {len(data)} bytes]")
        except Exception:
            descriptions.append(f"[Could not decode: {filename}]")
    return "\n".join(descriptions), parts


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

    logger.info(f"Task: {prompt[:1000]}")

    # === PRE-FLIGHT: Auto-setup and environment discovery ===
    auth = ("0", session_token)
    env_info = {}

    try:
        # 1. Get company ID and employee ID
        whoami_resp = http_requests.get(f"{base_url}/token/session/>whoAmI", auth=auth, timeout=10)
        if whoami_resp.status_code == 200:
            whoami = whoami_resp.json()["value"]
            env_info["company_id"] = whoami.get("companyId")
            env_info["employee_id"] = whoami.get("employeeId")

        # 2. Get first ACTIVE department ID (inactive departments cause 422!)
        dept_resp = http_requests.get(f"{base_url}/department", auth=auth, params={"count": 10, "fields": "id,name,isInactive"}, timeout=10)
        if dept_resp.status_code == 200 and dept_resp.json().get("values"):
            # Find first active department
            active_dept = None
            for dept in dept_resp.json()["values"]:
                if not dept.get("isInactive", False):
                    active_dept = dept
                    break
            if active_dept:
                env_info["department_id"] = active_dept["id"]
                env_info["department_name"] = active_dept.get("name", "")
            else:
                # All departments are inactive — create a new one
                new_dept = http_requests.post(f"{base_url}/department", auth=auth, json={"name": "Avdeling"}, timeout=10)
                if new_dept.status_code == 201:
                    env_info["department_id"] = new_dept.json().get("value", {}).get("id")
                    env_info["department_name"] = "Avdeling"
                    logger.info(f"Created new department (all were inactive): {env_info.get('department_id')}")
                else:
                    # Fallback to first department even if inactive
                    env_info["department_id"] = dept_resp.json()["values"][0]["id"]
                    env_info["department_name"] = dept_resp.json()["values"][0].get("name", "")

        # 3. Ensure bank account is configured (required for invoicing)
        # 3a. Set bank account on LEDGER account 1920
        acc_resp = http_requests.get(
            f"{base_url}/ledger/account",
            auth=auth,
            params={"numberFrom": "1920", "numberTo": "1920", "fields": "id,number,bankAccountNumber,version,isInvoiceAccount,isBankAccount"},
            timeout=10
        )
        if acc_resp.status_code == 200 and acc_resp.json().get("values"):
            acc = acc_resp.json()["values"][0]
            acc["bankAccountNumber"] = "15030100007"
            acc["isBankAccount"] = True
            acc["isInvoiceAccount"] = True
            bank_resp = http_requests.put(f"{base_url}/ledger/account/{acc['id']}", auth=auth, json=acc, timeout=10)
            logger.info(f"Bank account on ledger: {bank_resp.status_code}")
            env_info["bank_configured"] = True

        # 3b. Set bank account on COMPANY level (required for invoice creation!)
        # Try multiple approaches since different sandbox versions accept different methods
        if env_info.get("company_id"):
            try:
                comp_resp = http_requests.get(
                    f"{base_url}/company/{env_info['company_id']}",
                    auth=auth,
                    params={"fields": "id,name,organizationNumber,bankAccountNumber,version"},
                    timeout=10
                )
                if comp_resp.status_code == 200:
                    comp_data = comp_resp.json().get("value", {})
                    if not comp_data.get("bankAccountNumber"):
                        # Approach 1: PUT /company with minimal body
                        put_body = {
                            "id": comp_data.get("id"),
                            "name": comp_data.get("name"),
                            "bankAccountNumber": "15030100007",
                            "version": comp_data.get("version"),
                        }
                        comp_put = http_requests.put(
                            f"{base_url}/company/{env_info['company_id']}",
                            auth=auth,
                            json=put_body,
                            timeout=10
                        )
                        logger.info(f"Company bank (approach 1): {comp_put.status_code}")
                        if comp_put.status_code >= 400:
                            # Approach 2: GET with fields=* and PUT full object
                            comp_full = http_requests.get(
                                f"{base_url}/company/{env_info['company_id']}",
                                auth=auth,
                                params={"fields": "*"},
                                timeout=10
                            )
                            if comp_full.status_code == 200:
                                full_data = comp_full.json().get("value", {})
                                full_data["bankAccountNumber"] = "15030100007"
                                # Remove potentially read-only fields
                                for ro_field in ["changes", "url", "displayName"]:
                                    full_data.pop(ro_field, None)
                                comp_put2 = http_requests.put(
                                    f"{base_url}/company/{env_info['company_id']}",
                                    auth=auth,
                                    json=full_data,
                                    timeout=10
                                )
                                logger.info(f"Company bank (approach 2): {comp_put2.status_code}")
                                if comp_put2.status_code >= 400:
                                    # Approach 3: Try /company/altinn endpoint
                                    try:
                                        altinn_resp = http_requests.put(
                                            f"{base_url}/company/settings/altinn",
                                            auth=auth,
                                            json={"bankAccountNumber": "15030100007"},
                                            timeout=10
                                        )
                                        logger.info(f"Company bank (approach 3 altinn): {altinn_resp.status_code}")
                                    except Exception:
                                        pass
                                    logger.warning(f"Company bank error: {comp_put2.text[:300]}")
                    else:
                        logger.info(f"Company already has bank: {comp_data.get('bankAccountNumber')}")
            except Exception as e:
                logger.warning(f"Company bank setup failed: {e}")

        # 4. Get invoice payment types
        pt_resp = http_requests.get(f"{base_url}/invoice/paymentType", auth=auth, params={"fields": "id,description"}, timeout=10)
        if pt_resp.status_code == 200 and pt_resp.json().get("values"):
            for pt in pt_resp.json()["values"]:
                if "bank" in pt.get("description", "").lower():
                    env_info["payment_type_bank_id"] = pt["id"]
                elif "kontant" in pt.get("description", "").lower():
                    env_info["payment_type_cash_id"] = pt["id"]

        # 5. Get travel expense payment type
        te_pt_resp = http_requests.get(f"{base_url}/travelExpense/paymentType", auth=auth, params={"fields": "id,description", "count": 1}, timeout=10)
        if te_pt_resp.status_code == 200 and te_pt_resp.json().get("values"):
            env_info["travel_payment_type_id"] = te_pt_resp.json()["values"][0]["id"]

        # 6. Get voucherType IDs (for supplier invoices)
        try:
            vt_resp = http_requests.get(f"{base_url}/ledger/voucherType", auth=auth, params={"fields": "id,name"}, timeout=10)
            if vt_resp.status_code == 200 and vt_resp.json().get("values"):
                for vt in vt_resp.json()["values"]:
                    name_lower = vt.get("name", "").lower()
                    if "leverand" in name_lower:
                        env_info["voucher_type_supplier_id"] = vt["id"]
                    elif "kunde" in name_lower or "salg" in name_lower or "utgående" in name_lower:
                        env_info["voucher_type_customer_id"] = vt["id"]
                    elif "lønn" in name_lower or "lonn" in name_lower:
                        env_info["voucher_type_salary_id"] = vt["id"]
                    elif "betaling" == name_lower or name_lower == "betaling":
                        env_info["voucher_type_payment_id"] = vt["id"]
                    elif "manuelt" in name_lower or "manuell" in name_lower or "manuel" in name_lower or "memorial" in name_lower:
                        env_info["voucher_type_manual_id"] = vt["id"]
                    elif "purring" in name_lower:
                        env_info["voucher_type_purring_id"] = vt["id"]
                # Log all voucherTypes for debugging
                env_info["all_voucher_types"] = [
                    {"id": vt["id"], "name": vt.get("name", "")}
                    for vt in vt_resp.json()["values"]
                ]
                # Fallback: if no manual type found, use a general-purpose type
                # NEVER use Purring — it requires customer.id on every posting!
                if "voucher_type_manual_id" not in env_info:
                    # Prefer these types as manual fallback (most general-purpose)
                    preferred_fallback = ["terminoppgave", "bankavstemming", "ansattutlegg"]
                    excluded_ids = {
                        env_info.get("voucher_type_supplier_id"),
                        env_info.get("voucher_type_customer_id"),
                        env_info.get("voucher_type_salary_id"),
                        env_info.get("voucher_type_payment_id"),
                        env_info.get("voucher_type_purring_id"),  # NEVER use Purring!
                    }
                    # First try preferred types
                    for vt in vt_resp.json()["values"]:
                        if vt["id"] not in excluded_ids and vt.get("name", "").lower() in preferred_fallback:
                            env_info["voucher_type_manual_id"] = vt["id"]
                            logger.info(f"Manual voucher fallback: {vt.get('name')} (id={vt['id']})")
                            break
                    # If still not found, use any non-excluded type
                    if "voucher_type_manual_id" not in env_info:
                        for vt in vt_resp.json()["values"]:
                            if vt["id"] not in excluded_ids:
                                env_info["voucher_type_manual_id"] = vt["id"]
                                logger.info(f"Manual voucher last-resort: {vt.get('name')} (id={vt['id']})")
                                break
        except Exception:
            pass

        # 6b. Get travelExpense costCategory IDs (CRITICAL — IDs differ per sandbox!)
        try:
            cc_resp = http_requests.get(f"{base_url}/travelExpense/costCategory", auth=auth, params={"fields": "id,description", "count": 50}, timeout=10)
            if cc_resp.status_code == 200 and cc_resp.json().get("values"):
                cost_cats = {}
                for cc in cc_resp.json()["values"]:
                    desc = cc.get("description", "").lower()
                    cost_cats[cc["id"]] = cc.get("description", "")
                    if desc == "fly":
                        env_info["cost_cat_fly_id"] = cc["id"]
                    elif desc == "flytog":
                        env_info["cost_cat_flytog_id"] = cc["id"]
                    elif desc == "flybuss":
                        env_info["cost_cat_flybuss_id"] = cc["id"]
                    elif "hotell" in desc:
                        env_info["cost_cat_hotell_id"] = cc["id"]
                    elif desc == "taxi":
                        env_info["cost_cat_taxi_id"] = cc["id"]
                    elif desc == "tog":
                        env_info["cost_cat_tog_id"] = cc["id"]
                    elif "mat" == desc:
                        env_info["cost_cat_mat_id"] = cc["id"]
                    elif "parkering" in desc:
                        env_info["cost_cat_parkering_id"] = cc["id"]
                    elif "bom" in desc:
                        env_info["cost_cat_bom_id"] = cc["id"]
                    elif "drivstoff" == desc:
                        env_info["cost_cat_drivstoff_id"] = cc["id"]
                    elif "kontorrekvisita" in desc:
                        env_info["cost_cat_kontor_id"] = cc["id"]
                    elif "telefon" in desc:
                        env_info["cost_cat_telefon_id"] = cc["id"]
                    elif "buss" == desc:
                        env_info["cost_cat_buss_id"] = cc["id"]
                env_info["all_cost_categories"] = [{"id": k, "desc": v} for k, v in list(cost_cats.items())[:20]]
        except Exception:
            pass

        # 7. Get activity IDs (for timesheet)
        try:
            act_resp = http_requests.get(f"{base_url}/activity", auth=auth, params={"fields": "id,name", "count": 10}, timeout=10)
            if act_resp.status_code == 200 and act_resp.json().get("values"):
                env_info["activity_ids"] = [
                    {"id": a["id"], "name": a.get("name", "")}
                    for a in act_resp.json()["values"][:5]
                ]
        except Exception:
            pass

        # 8. Get salary type IDs (for payroll)
        try:
            st_resp = http_requests.get(f"{base_url}/salary/type", auth=auth, params={"fields": "id,number,name", "count": 10}, timeout=10)
            if st_resp.status_code == 200 and st_resp.json().get("values"):
                env_info["salary_type_ids"] = [
                    {"id": s["id"], "number": s.get("number"), "name": s.get("name", "")}
                    for s in st_resp.json()["values"][:5]
                ]
        except Exception:
            pass

        # 9. Get common ledger account IDs (for vouchers/supplier invoices)
        try:
            la_resp = http_requests.get(
                f"{base_url}/ledger/account", auth=auth,
                params={"fields": "id,number,name", "count": 1000},
                timeout=15
            )
            if la_resp.status_code == 200 and la_resp.json().get("values"):
                # Dynamic account mapping — ALL accounts, not just hardcoded ones
                account_map = {}
                for acc in la_resp.json()["values"]:
                    acc_num = str(acc.get("number", ""))
                    acc_id = acc.get("id")
                    acc_name = acc.get("name", "")
                    if acc_num and acc_id:
                        account_map[acc_num] = {"id": acc_id, "name": acc_name}
                        env_info[f"account_{acc_num}_id"] = acc_id
                env_info["all_account_map"] = account_map
                logger.info(f"  Loaded {len(account_map)} accounts dynamically")
                # All accounts are now mapped dynamically above
        except Exception:
            pass

        logger.info(f"Pre-flight done: {json.dumps(env_info)}")

    except Exception as e:
        logger.warning(f"Pre-flight failed (non-critical): {e}")

    # === BUILD PROMPT WITH ENVIRONMENT INFO ===
    env_block = ""
    if env_info:
        env_block = f"""

## ENVIRONMENT (pre-fetched — use these directly, do NOT call GET for them)
- today_date: {__import__('datetime').date.today().isoformat()}
- company_id: {env_info.get('company_id', 'unknown')}
- employee_id (logged-in user): {env_info.get('employee_id', 'unknown')}
- department_id: {env_info.get('department_id', 'unknown')} (name: "{env_info.get('department_name', '')}")
- bank_account: {'configured' if env_info.get('bank_configured') else 'unknown'}
- invoice_payment_type_bank_id: {env_info.get('payment_type_bank_id', 'unknown')}
- invoice_payment_type_cash_id: {env_info.get('payment_type_cash_id', 'unknown')}
- travel_payment_type_id: {env_info.get('travel_payment_type_id', 'unknown')}
- voucher_type_supplier_id: {env_info.get('voucher_type_supplier_id', 'unknown')}
- voucher_type_customer_id: {env_info.get('voucher_type_customer_id', 'unknown')}
- voucher_type_salary_id (Lønnsbilag): {env_info.get('voucher_type_salary_id', 'unknown')}
- voucher_type_payment_id (Betaling): {env_info.get('voucher_type_payment_id', 'unknown')}
- voucher_type_manual_id (Manuelt bilag — for corrections, depreciation, accruals, bank charges): {env_info.get('voucher_type_manual_id', 'unknown')}
- activity_ids: {json.dumps(env_info.get('activity_ids', []))}
- salary_type_ids: {json.dumps(env_info.get('salary_type_ids', []))}

Travel expense cost categories (USE THESE IDs, not hardcoded ones!):
- cost_cat_fly_id: {env_info.get('cost_cat_fly_id', 'unknown')}
- cost_cat_hotell_id: {env_info.get('cost_cat_hotell_id', 'unknown')}
- cost_cat_taxi_id: {env_info.get('cost_cat_taxi_id', 'unknown')}
- cost_cat_tog_id: {env_info.get('cost_cat_tog_id', 'unknown')}
- cost_cat_mat_id: {env_info.get('cost_cat_mat_id', 'unknown')}
- cost_cat_parkering_id: {env_info.get('cost_cat_parkering_id', 'unknown')}
- cost_cat_bom_id: {env_info.get('cost_cat_bom_id', 'unknown')}
- cost_cat_drivstoff_id: {env_info.get('cost_cat_drivstoff_id', 'unknown')}
- cost_cat_kontor_id: {env_info.get('cost_cat_kontor_id', 'unknown')}
- cost_cat_telefon_id: {env_info.get('cost_cat_telefon_id', 'unknown')}
- cost_cat_buss_id: {env_info.get('cost_cat_buss_id', 'unknown')}
- all_cost_categories: {json.dumps(env_info.get('all_cost_categories', []))}
LEDGER ACCOUNTS (use account_NUMBER_id format — ALL are pre-fetched, NO GET needed!):
{chr(10).join(f'- account_{num}_id ({info["name"]}): {info["id"]}' for num, info in sorted(env_info.get('all_account_map', {}).items(), key=lambda x: int(x[0])))}

CRITICAL: Use the account IDs above directly — NEVER call GET /ledger/account! Every account in the chart is listed above.
Match the account to the task: "kontorrekvisita"→6800, "kontortjenester"→6500/7100, "reisekostnad"→7140, etc.

INCLUDE EVERY FIELD from the prompt! Scoring is FIELD-BY-FIELD:
- Customer: name, email, organizationNumber, postalAddress(addressLine1,postalCode,city), phoneNumber, invoiceEmail, language
- Employee: firstName, lastName, email, dateOfBirth, nationalIdentityNumber, phoneNumberMobile, employeeNumber, address
- Product: name, number, priceExcludingVatCurrency, vatType.id (3=25%, 5=15%, 6=0%)
- Invoice: invoiceDueDate (ALWAYS set = invoiceDate + 30 days), sendToCustomer (true if "send" in prompt)
- Orderline: description (use product/service name from prompt!)
- Voucher: description should include WHAT, WHO, PERIOD, CALCULATION
"""

    # === TWO-STEP PREFETCH: Get customer/product IDs BEFORE Gemini call ===
    invoice_prefetch_context = ""
    prompt_lower = prompt.lower()
    is_prefetch_task = any(kw in prompt_lower for kw in [
        "faktura", "invoice", "rechnung", "factura", "fatura", "facture",
        "ordre", "order", "bestilling", "bestellung", "commande", "pedido",
        "fakturere", "fakturering", "invoicing",
        "projekt", "project", "prosjekt", "proyecto", "projet",
        "timer", "hours", "stunden", "horas", "heures",
        "lønn", "salary", "gehalt", "salario", "salaire", "nómina",
        "leverand", "supplier", "lieferant", "fornecedor", "proveedor", "fournisseur",
        "gutschrift", "kreditnota", "credit note", "nota de crédito", "avoir",
        "betaling", "payment", "zahlung", "pago", "pagamento", "paiement",
    ])

    if is_prefetch_task and env_info.get("company_id"):
        logger.info("Invoice task detected — running two-step prefetch")
        prefetch_results = []

        # 1. Find customer(s) by organization number
        orgnr_matches = re.findall(
            r'(?:org\.?\s*(?:nr\.?|n[uú]m(?:ero)?|nummer)?\.?|organisasjonsnummer|organization\s*number|Org\.-Nr\.?)\s*[:=]?\s*(\d{9})',
            prompt, re.IGNORECASE
        )
        for orgnr in set(orgnr_matches):
            try:
                cust_resp = http_requests.get(
                    f"{base_url}/customer", auth=auth,
                    params={"organizationNumber": orgnr, "fields": "id,name,organizationNumber", "count": 3},
                    timeout=10,
                )
                if cust_resp.status_code == 200:
                    cust_values = cust_resp.json().get("values", [])
                    if cust_values:
                        c = cust_values[0]
                        prefetch_results.append(f"CUSTOMER (orgnr {orgnr}): id={c['id']}, name=\"{c.get('name', '')}\"")
                        logger.info(f"  Prefetch customer orgnr={orgnr} → id={c['id']}")
                    else:
                        prefetch_results.append(f"CUSTOMER NOT FOUND for orgnr {orgnr} — create with POST /customer")
            except Exception:
                pass

        # 2. Find product(s) by product number (parenthesized numbers after product names)
        prod_matches = re.findall(r'[\(\(](\d{3,5})[\)\)]', prompt)
        for prod_num in set(prod_matches):
            try:
                prod_resp = http_requests.get(
                    f"{base_url}/product", auth=auth,
                    params={"number": prod_num, "fields": "id,name,number,priceExcludingVatCurrency", "count": 3},
                    timeout=10,
                )
                if prod_resp.status_code == 200:
                    prod_values = prod_resp.json().get("values", [])
                    if prod_values:
                        p = prod_values[0]
                        prefetch_results.append(f"PRODUCT (number {prod_num}): id={p['id']}, name=\"{p.get('name', '')}\"")
                        logger.info(f"  Prefetch product nr={prod_num} → id={p['id']}")
                    else:
                        prefetch_results.append(f"PRODUCT NOT FOUND for number {prod_num} — create with POST /product")
            except Exception:
                pass

        # 3. Also prefetch supplier if this is a supplier-related task
        if any(kw in prompt_lower for kw in ["leverand", "supplier", "lieferant", "fornecedor", "proveedor", "fournisseur"]):
            for orgnr in set(orgnr_matches):
                try:
                    sup_resp = http_requests.get(
                        f"{base_url}/supplier", auth=auth,
                        params={"organizationNumber": orgnr, "fields": "id,name,organizationNumber", "count": 3},
                        timeout=10,
                    )
                    if sup_resp.status_code == 200:
                        sup_values = sup_resp.json().get("values", [])
                        if sup_values:
                            s = sup_values[0]
                            prefetch_results.append(f"SUPPLIER (orgnr {orgnr}): id={s['id']}, name=\"{s.get('name', '')}\"")
                            logger.info(f"  Prefetch supplier orgnr={orgnr} → id={s['id']}")
                        else:
                            prefetch_results.append(f"SUPPLIER NOT FOUND for orgnr {orgnr} — create with POST /supplier")
                except Exception:
                    pass

        # 4. Prefetch employees by email (prevents "email already exists" errors)
        email_matches = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', prompt)
        for email in set(email_matches):
            try:
                emp_resp = http_requests.get(
                    f"{base_url}/employee", auth=auth,
                    params={"email": email, "fields": "id,firstName,lastName,email", "count": 3},
                    timeout=10,
                )
                if emp_resp.status_code == 200:
                    emp_values = emp_resp.json().get("values", [])
                    if emp_values:
                        e = emp_values[0]
                        prefetch_results.append(
                            f"EMPLOYEE (email {email}): id={e['id']}, name=\"{e.get('firstName', '')} {e.get('lastName', '')}\""
                            f" — ALREADY EXISTS, use this id, do NOT create a new employee!"
                        )
                        logger.info(f"  Prefetch employee email={email} → id={e['id']}")
            except Exception:
                pass

        # 5. Prefetch existing invoices if task mentions payment/credit/reversal
        if any(kw in prompt_lower for kw in ["betaling", "payment", "zahlung", "pago", "pagamento", "paiement",
                                               "kreditnota", "credit note", "avoir", "nota de crédito",
                                               "reverser", "reverse", "stornieren", "revertir"]):
            try:
                inv_resp = http_requests.get(
                    f"{base_url}/invoice", auth=auth,
                    params={"invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31",
                            "fields": "id,invoiceNumber,amount,amountOutstanding,customer(id,name)", "count": 20},
                    timeout=10,
                )
                if inv_resp.status_code == 200:
                    inv_values = inv_resp.json().get("values", [])
                    if inv_values:
                        for inv in inv_values[:5]:
                            cust_name = inv.get("customer", {}).get("name", "unknown") if inv.get("customer") else "unknown"
                            prefetch_results.append(
                                f"INVOICE id={inv['id']}, number={inv.get('invoiceNumber', 'N/A')}, "
                                f"amount={inv.get('amount', 0)}, outstanding={inv.get('amountOutstanding', 0)}, "
                                f"customer=\"{cust_name}\""
                            )
                        logger.info(f"  Prefetch invoices: {len(inv_values)} found")
            except Exception:
                pass

        if prefetch_results:
            invoice_prefetch_context = "\n\n## PRE-FETCHED DATA (use these IDs directly — do NOT call GET for them!)\n"
            invoice_prefetch_context += "\n".join(f"- {r}" for r in prefetch_results)
            invoice_prefetch_context += "\nIMPORTANT: Use these IDs directly. For employees marked ALREADY EXISTS, use their id — do NOT create a new employee with POST!"
            logger.info(f"  Prefetch: {len(prefetch_results)} entities found")
    # === END TWO-STEP PREFETCH ===

    user_prompt = f"Task prompt:\n{prompt}{invoice_prefetch_context}"
    multimodal_parts = []
    if files:
        file_info, multimodal_parts = extract_file_content(files)
        user_prompt += f"\n\nAttached files:\n{file_info}"
        logger.info(f"Files attached: {len(files)} ({len(multimodal_parts)} multimodal)")

    # Ask Gemini to plan API calls
    if not client:
        logger.error("No Gemini client — GEMINI_API_KEY not set")
        return JSONResponse({"status": "completed"})

    try:
        # Build content parts — text first, then any multimodal file parts
        content_parts = [SYSTEM_PROMPT + env_block, user_prompt] + multimodal_parts
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content_parts,
            config={"temperature": 0.2, "max_output_tokens": 16384},
        )
        raw_text = response.text.strip()
        logger.info(f"Gemini raw response length: {len(raw_text)}")

        # Use robust JSON cleaning
        text = clean_json_text(raw_text)

        # Replace placeholder strings with actual env values BEFORE JSON parsing
        # Gemini sometimes outputs "VOUCHER_TYPE_SUPPLIER_ID" instead of the numeric ID
        if env_info:
            replacements = {
                "VOUCHER_TYPE_SUPPLIER_ID": env_info.get("voucher_type_supplier_id"),
                "VOUCHER_TYPE_CUSTOMER_ID": env_info.get("voucher_type_customer_id"),
                "VOUCHER_TYPE_SALARY_ID": env_info.get("voucher_type_salary_id"),
                "VOUCHER_TYPE_PAYMENT_ID": env_info.get("voucher_type_payment_id"),
                "VOUCHER_TYPE_MANUAL_ID": env_info.get("voucher_type_manual_id"),
                "VOUCHER_TYPE_LONNSBILAG_ID": env_info.get("voucher_type_salary_id"),
                "PAYMENT_TYPE_BANK_ID": env_info.get("payment_type_bank_id"),
                "PAYMENT_TYPE_CASH_ID": env_info.get("payment_type_cash_id"),
                "DEPARTMENT_ID": env_info.get("department_id"),
                "COMPANY_ID": env_info.get("company_id"),
                "EMPLOYEE_ID": env_info.get("employee_id"),
            }
            # Add activity ID (first one)
            activity_ids = env_info.get("activity_ids", [])
            if activity_ids:
                replacements["FIRST_ACTIVITY_ID"] = activity_ids[0].get("id")
            # Add travel payment type
            if env_info.get("travel_payment_type_id"):
                replacements["TRAVEL_PAYMENT_TYPE_ID"] = env_info.get("travel_payment_type_id")
            # Add ALL integer values from env_info — catches cost categories, payment types, accounts, etc.
            for key, value in env_info.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    replacements[key.upper()] = value
                    # Also add without _id suffix variants
                    if key.endswith("_id"):
                        replacements[key.upper()] = value
            # Also add "from ENVIRONMENT" variants
            for key, value in list(replacements.items()):
                if value is not None:
                    replacements[f"{key} from ENVIRONMENT"] = value
                    replacements[f"{key} FROM ENVIRONMENT"] = value

            replaced_count = 0
            # Sort by key length DESCENDING to prevent partial matches
            # e.g. "ACCOUNT_5000_ID from ENVIRONMENT" before "ACCOUNT_5000_ID"
            for placeholder, real_value in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
                if real_value is not None:
                    # Replace quoted version: "DEPARTMENT_ID" → 12345
                    if f'"{placeholder}"' in text:
                        text = text.replace(f'"{placeholder}"', str(real_value))
                        replaced_count += 1
                        logger.info(f"  Replaced '{placeholder}' → {real_value}")
                    # Replace bare version: DEPARTMENT_ID → 12345 (fixes invalid JSON)
                    if f': {placeholder}' in text or f':{placeholder}' in text:
                        text = re.sub(rf':\s*{re.escape(placeholder)}\b', f': {real_value}', text)
                        replaced_count += 1
                        logger.info(f"  Replaced bare '{placeholder}' → {real_value}")
            if replaced_count > 0:
                logger.info(f"  Total placeholders replaced: {replaced_count}")

        try:
            calls = json.loads(text)
        except json.JSONDecodeError:
            # Last resort: try to fix common issues
            logger.warning(f"Initial JSON parse failed, attempting additional cleanup")
            # Try removing any non-JSON prefix/suffix
            text_stripped = text.strip()
            if text_stripped:
                calls = json.loads(text_stripped)
            else:
                raise
        if not isinstance(calls, list):
            calls = [calls]

        # SECONDARY placeholder replacement — catches any remaining placeholders after JSON parsing
        # Primary replacement happens on raw text BEFORE json.loads (line ~1797)
        placeholder_map = {}
        for key, value in env_info.items():
            if isinstance(value, (int, float)):
                # Map both uppercase and original formats
                placeholder_map[key.upper()] = value
                placeholder_map[key] = value
        # Also map common patterns from systemprompt examples
        name_mappings = {
            "VOUCHER_TYPE_SUPPLIER_ID": env_info.get("voucher_type_supplier_id"),
            "VOUCHER_TYPE_CUSTOMER_ID": env_info.get("voucher_type_customer_id"),
            "VOUCHER_TYPE_SALARY_ID": env_info.get("voucher_type_salary_id"),
            "VOUCHER_TYPE_PAYMENT_ID": env_info.get("voucher_type_payment_id"),
            "VOUCHER_TYPE_MANUAL_ID": env_info.get("voucher_type_manual_id"),
            "VOUCHER_TYPE_LONNSBILAG_ID": env_info.get("voucher_type_salary_id"),
            "PAYMENT_TYPE_BANK_ID": env_info.get("payment_type_bank_id"),
            "PAYMENT_TYPE_CASH_ID": env_info.get("payment_type_cash_id"),
            "DEPARTMENT_ID": env_info.get("department_id"),
            "COMPANY_ID": env_info.get("company_id"),
            "EMPLOYEE_ID": env_info.get("employee_id"),
        }
        for k, v in name_mappings.items():
            if v is not None:
                placeholder_map[k] = v
        # Add ACCOUNT_XXXX_ID patterns
        for key, value in env_info.items():
            if key.startswith("account_") and isinstance(value, int):
                placeholder_map[key.upper()] = value

        def replace_placeholders_in_value(val):
            if isinstance(val, str):
                # Check if the entire string is a placeholder
                upper_val = val.strip().upper().replace(" ", "_")
                # Remove common suffixes like "FROM ENVIRONMENT"
                for suffix in [" FROM ENVIRONMENT", "_FROM_ENVIRONMENT", " from ENVIRONMENT"]:
                    if val.upper().endswith(suffix.upper()):
                        upper_val = val[:len(val)-len(suffix)].strip().upper().replace(" ", "_")
                        break
                if upper_val in placeholder_map:
                    logger.info(f"  Replaced placeholder '{val}' → {placeholder_map[upper_val]}")
                    return placeholder_map[upper_val]
                # Also check without _ID suffix variations
                for pk, pv in placeholder_map.items():
                    if val.upper().replace(" ", "_").replace("-", "_") == pk:
                        logger.info(f"  Replaced placeholder '{val}' → {pv}")
                        return pv
                return val
            elif isinstance(val, dict):
                return {k: replace_placeholders_in_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [replace_placeholders_in_value(v) for v in val]
            return val

        calls = replace_placeholders_in_value(calls)

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
            retry_text = clean_json_text(retry_response.text.strip())
            try:
                calls = json.loads(retry_text)
                if not isinstance(calls, list):
                    calls = [calls]
                logger.info(f"Fallback produced {len(calls)} API calls")
            except Exception:
                logger.error(f"Fallback also failed: {retry_text[:200]}")

        # Validate calls before execution
        calls, validation_warnings = validate_calls(calls)

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
        results = execute_api_calls(calls, base_url, session_token, original_prompt=prompt, env_info=env_info)
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
