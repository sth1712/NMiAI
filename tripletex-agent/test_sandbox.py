"""
Tripletex Agent — Comprehensive Sandbox Test Suite
Kjør: python3 test_sandbox.py

Tester agenten mot sandbox for alle kjente oppgavetyper.
Verifiserer at resultater faktisk ble opprettet.
"""
import requests
import json
import sys
import time

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

auth = ("0", SANDBOX_TOKEN)

def send_task(prompt, timeout=120):
    """Send a task to the agent and return status code."""
    r = requests.post(
        f"{AGENT_URL}/solve",
        json={
            "prompt": prompt,
            "files": [],
            "tripletex_credentials": {
                "base_url": SANDBOX_URL,
                "session_token": SANDBOX_TOKEN
            }
        },
        timeout=timeout
    )
    return r.status_code, r.text

def count_entities(entity_type, params=None):
    """Count entities in sandbox."""
    p = {"fields": "id", "count": 0}
    if params:
        p.update(params)
    r = requests.get(f"{SANDBOX_URL}/{entity_type}", auth=auth, params=p)
    if r.status_code == 200:
        return r.json().get("fullResultSize", 0)
    return -1

def search_entity(entity_type, **kwargs):
    """Search for entity by fields.
    Fetches a list of recent entities and filters them locally
    to avoid API search implementation quirks.
    """
    # Define fields appropriate for each entity type to prevent 400 errors
    field_map = {
        "employee": "id,firstName,lastName,email,employeeNumber",
        "customer": "id,name,email,organizationNumber",
        "product": "id,name,number,priceExcludingVatCurrency",
        "department": "id,name,departmentNumber",
        "supplier": "id,name,email,organizationNumber",
    }
    fields = field_map.get(entity_type, "id,name,email") # Default to common fields

    # Fetch a list of the most recent entities
    params = {"fields": fields, "count": 250}
    try:
        r = requests.get(f"{SANDBOX_URL}/{entity_type}", auth=auth, params=params, timeout=20)
        if r.status_code != 200:
            print(f"  (Search query failed for {entity_type} with status {r.status_code}, params {params})")
            return []
        values = r.json().get("values", [])
    except requests.exceptions.RequestException as e:
        print(f"  (Search query failed for {entity_type}: {e})")
        return []

    if not values:
        return []

    # Filter the results locally
    filtered_results = []
    for item in values:
        match = True
        for key, value in kwargs.items():
            item_value = item.get(key)
            # Ensure we are comparing strings, handle None gracefully
            if item_value is None or str(value).lower() not in str(item_value).lower():
                match = False
                break
        if match:
            filtered_results.append(item)

    return filtered_results

# ===== TESTS =====

def test_health():
    print("=== Health Check ===")
    r = requests.get(f"{AGENT_URL}/health", timeout=10)
    ok = r.status_code == 200
    print(f"{'✅' if ok else '❌'} Status: {r.status_code}")
    return ok

def test_employee_simple():
    print("\n=== Opprett ansatt (enkel) ===")
    ts = int(time.time())
    email = f"test.emp.{ts}@sandbox.no"
    before = count_entities("employee")

    code, text = send_task(f"Opprett en ansatt med navn Test Ansatt, e-post {email}")
    after = count_entities("employee")
    found = search_entity("employee", email=email)

    ok = len(found) > 0
    print(f"{'✅' if ok else '❌'} Agent: {code}, employees {before}→{after}, found={len(found)}")
    return ok

def test_employee_admin():
    print("\n=== Opprett ansatt (admin) ===")
    ts = int(time.time())
    email = f"test.admin.{ts}@sandbox.no"

    code, text = send_task(f"Opprett en ansatt med navn Admin Test, e-post {email}. Vedkommende skal være kontoadministrator.")
    found = search_entity("employee", email=email)

    ok = len(found) > 0
    # Sjekk om admin-entitlement ble satt
    if found:
        emp_id = found[0]["id"]
        r = requests.get(f"{SANDBOX_URL}/employee/entitlement", auth=auth,
                        params={"employeeId": emp_id, "entitlementId": 1, "fields": "id"})
        has_admin = r.json().get("fullResultSize", 0) > 0
        print(f"{'✅' if has_admin else '⚠️'} Admin-rolle: {'ja' if has_admin else 'nei'}")

    print(f"{'✅' if ok else '❌'} Ansatt opprettet: {ok}")
    return ok

def test_customer():
    print("\n=== Opprett kunde ===")
    ts = int(time.time())
    name = f"TestKunde {ts} AS"

    code, text = send_task(f"Opprett en kunde med navn {name}, e-post kunde{ts}@test.no")
    found = search_entity("customer", name=name)

    ok = len(found) > 0
    print(f"{'✅' if ok else '❌'} Kunde opprettet: {ok}")
    return ok

def test_product():
    print("\n=== Opprett produkt (med pris og MVA) ===")
    ts = int(time.time())

    code, text = send_task(f"Opprett produktet \"Konsulenttime {ts}\" med produktnummer {ts}. Prisen er 1200 NOK eksklusiv MVA, med standard 25% MVA-sats.")
    found = search_entity("product", number=str(ts))

    ok = len(found) > 0
    if found:
        prod = requests.get(f"{SANDBOX_URL}/product/{found[0]['id']}", auth=auth, params={"fields": "*"}).json()["value"]
        print(f"  Pris: {prod.get('priceExcludingVatCurrency')} NOK ekskl, {prod.get('priceIncludingVatCurrency')} inkl")

    print(f"{'✅' if ok else '❌'} Produkt opprettet: {ok}")
    return ok

def test_department():
    print("\n=== Opprett avdeling ===")
    ts = int(time.time())
    name = f"TestAvdeling {ts}"

    code, text = send_task(f"Opprett en avdeling med navn {name} og avdelingsnummer {ts % 1000}")
    found = search_entity("department", name=name)

    ok = len(found) > 0
    print(f"{'✅' if ok else '❌'} Avdeling opprettet: {ok}")
    return ok

def test_supplier():
    print("\n=== Opprett leverandør ===")
    ts = int(time.time())
    name = f"Leverandør {ts} AS"

    code, text = send_task(f"Opprett en leverandør med navn {name}, e-post lev{ts}@test.no")
    found = search_entity("supplier", name=name)

    ok = len(found) > 0
    print(f"{'✅' if ok else '❌'} Leverandør opprettet: {ok}")
    return ok

def test_spanish_prompt():
    print("\n=== Spansk prompt ===")
    ts = int(time.time())

    code, text = send_task(f'Crea el producto "Servicio {ts}" con número de producto {ts}. El precio es 850 NOK sin IVA, con la tasa estándar del 25%.')
    found = search_entity("product", number=str(ts))

    ok = len(found) > 0
    print(f"{'✅' if ok else '❌'} Spansk produkt opprettet: {ok}")
    return ok

# ===== MAIN =====

if __name__ == "__main__":
    print("Tripletex Agent — Comprehensive Sandbox Test Suite")
    print(f"Agent: {AGENT_URL}")
    print(f"Sandbox: {SANDBOX_URL}")
    print()

    if not test_health():
        print("\n❌ Agent er nede!")
        sys.exit(1)

    tests = [
        ("Ansatt (enkel)", test_employee_simple),
        ("Ansatt (admin)", test_employee_admin),
        ("Kunde", test_customer),
        ("Produkt", test_product),
        ("Avdeling", test_department),
        ("Leverandør", test_supplier),
        ("Spansk prompt", test_spanish_prompt),
    ]

    results = []
    for name, test_fn in tests:
        try:
            ok = test_fn()
            results.append((name, ok))
        except Exception as e:
            print(f"❌ {name} FEILET: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("OPPSUMMERING")
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{passed}/{total} tester bestått\n")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")

    if passed < total:
        print(f"\n⚠️ {total - passed} tester feilet — sjekk logger:")
        print(f"  gcloud run services logs read tripletex-agent --region europe-north1 --limit 30")
