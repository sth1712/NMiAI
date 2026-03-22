"""
Tripletex Agent — 15-oppgave simulering
Kjører 15 realistiske oppgaver mot den deployede agenten og verifiserer mot sandbox.
"""
import requests
import json
import sys
import time
import traceback

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

auth = ("0", SANDBOX_TOKEN)
TS = int(time.time())

results = []

def send_task(prompt, timeout=180):
    """Send oppgave til agenten. Returner (status_code, respons_tekst, tid_brukt)."""
    start = time.time()
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
    elapsed = time.time() - start
    return r.status_code, r.text, elapsed

def search_entities(entity_type, count=250, **extra_params):
    """Hent entiteter fra sandbox."""
    field_map = {
        "employee": "id,firstName,lastName,email,phoneNumberMobile,dateOfBirth",
        "customer": "id,name,email,organizationNumber,phoneNumber",
        "product": "id,name,number,priceExcludingVatCurrency",
        "department": "id,name,departmentNumber",
        "supplier": "id,name,organizationNumber",
        "contact": "id,firstName,lastName,email,customer",
        "project": "id,name,number,projectManager",
        "travelExpense": "id,title,employee",
        "order": "id,customer,orderDate",
        "invoice": "id,invoiceNumber,amount,customer",
    }
    params = {"fields": field_map.get(entity_type, "id,name"), "count": count}
    params.update(extra_params)
    try:
        r = requests.get(f"{SANDBOX_URL}/{entity_type}", auth=auth, params=params, timeout=20)
        if r.status_code == 200:
            return r.json().get("values", [])
    except:
        pass
    return []

def find_by_field(entities, field, value):
    """Filtrer entitetliste lokalt."""
    return [e for e in entities if value.lower() in str(e.get(field, "")).lower()]


# ============================
# TIER 1: Enkle oppgaver
# ============================

def task_01():
    """Opprett ansatt med telefon og fødselsdato (norsk)"""
    email = f"kari.nordmann.{TS}@sandbox.no"
    prompt = f"""Opprett en ansatt med følgende informasjon:
- Fornavn: Kari
- Etternavn: Nordmann
- E-post: {email}
- Mobilnummer: 99887766
- Fødselsdato: 1985-06-15"""

    code, text, elapsed = send_task(prompt)

    employees = search_entities("employee")
    found = find_by_field(employees, "email", email)

    ok = len(found) > 0
    details = ""
    if found:
        emp = found[0]
        has_phone = emp.get("phoneNumberMobile") == "99887766"
        has_dob = "1985-06-15" in str(emp.get("dateOfBirth", ""))
        details = f"telefon={'OK' if has_phone else 'MANGLER'}, fdato={'OK' if has_dob else 'MANGLER'}"

    return ok, code, elapsed, details

def task_02():
    """Opprett kunde med orgnr og adresse (engelsk)"""
    name = f"Nordic Solutions {TS} AS"
    prompt = f"""Create a customer with the following details:
- Name: {name}
- Organization number: 912345678
- Email: contact@nordicsolutions.no
- Address: Storgata 10, 0182 Oslo"""

    code, text, elapsed = send_task(prompt)

    customers = search_entities("customer")
    found = find_by_field(customers, "name", f"Nordic Solutions {TS}")

    ok = len(found) > 0
    details = ""
    if found:
        c = found[0]
        has_org = str(c.get("organizationNumber", "")).replace(" ", "") == "912345678"
        details = f"orgnr={'OK' if has_org else 'MANGLER'}"
        # Sjekk adresse
        cust_full = requests.get(f"{SANDBOX_URL}/customer/{c['id']}", auth=auth, params={"fields": "*"}).json().get("value", {})
        addr = cust_full.get("postalAddress", {})
        has_addr = bool(addr.get("addressLine1"))
        details += f", adresse={'OK' if has_addr else 'MANGLER'}"

    return ok, code, elapsed, details

def task_03():
    """Opprett leverandør med orgnr (tysk)"""
    name = f"Lieferant Schmidt {TS} GmbH"
    prompt = f"""Erstellen Sie einen Lieferanten mit folgenden Informationen:
- Name: {name}
- Organisationsnummer: 998877665
- E-Mail: schmidt@lieferant.de"""

    code, text, elapsed = send_task(prompt)

    suppliers = search_entities("supplier")
    found = find_by_field(suppliers, "name", f"Lieferant Schmidt {TS}")

    ok = len(found) > 0
    details = ""
    if found:
        has_org = str(found[0].get("organizationNumber", "")).replace(" ", "") == "998877665"
        details = f"orgnr={'OK' if has_org else 'MANGLER'}"

    return ok, code, elapsed, details

def task_04():
    """Opprett 3 produkter med ulike MVA-satser (spansk)"""
    prompt = f"""Crea tres productos con las siguientes especificaciones:

1. Producto "Consultoría {TS}" con número {TS}1, precio 1500 NOK sin IVA, IVA estándar 25%
2. Producto "Alimentación {TS}" con número {TS}2, precio 200 NOK sin IVA, IVA del 15% (comida)
3. Producto "Transporte {TS}" con número {TS}3, precio 350 NOK sin IVA, IVA del 12% (transporte)"""

    code, text, elapsed = send_task(prompt)

    products = search_entities("product")
    found1 = find_by_field(products, "name", f"Consultoría {TS}")
    found2 = find_by_field(products, "name", f"Alimentación {TS}")
    found3 = find_by_field(products, "name", f"Transporte {TS}")

    count = sum(1 for f in [found1, found2, found3] if len(f) > 0)
    ok = count == 3
    details = f"{count}/3 produkter opprettet"

    return ok, code, elapsed, details

def task_05():
    """Opprett avdeling (portugisisk)"""
    name = f"Departamento Vendas {TS}"
    prompt = f"""Crie um departamento com as seguintes informações:
- Nome: {name}
- Número do departamento: {TS % 900 + 100}"""

    code, text, elapsed = send_task(prompt)

    depts = search_entities("department")
    found = find_by_field(depts, "name", f"Departamento Vendas {TS}")

    ok = len(found) > 0
    details = ""

    return ok, code, elapsed, details


# ============================
# TIER 2: Flersteg-oppgaver
# ============================

def task_06():
    """Opprett kunde, produkt, ordre, ordrelinje og faktura (norsk)"""
    cust_name = f"FakturaKunde {TS} AS"
    prompt = f"""Opprett en komplett faktura med følgende:
- Kunde: {cust_name}
- Produkt: Rådgivningstime, pris 1500 NOK ekskl. MVA, 25% MVA
- Ordrelinje: 3 timer rådgivning
- Fakturer ordren med fakturadato 2026-03-20, ikke send til kunden"""

    code, text, elapsed = send_task(prompt)

    # Verifiser kunde
    customers = search_entities("customer")
    found_cust = find_by_field(customers, "name", f"FakturaKunde {TS}")

    # Verifiser faktura
    invoices = search_entities("invoice", invoiceDateFrom="2026-03-01", invoiceDateTo="2026-03-31")
    # Sjekk om det finnes en ny faktura

    ok = len(found_cust) > 0
    details = f"kunde={'OK' if found_cust else 'MANGLER'}, fakturaer_funnet={len(invoices)}"

    return ok, code, elapsed, details

def task_07():
    """Opprett ansatt med admin-rolle (fransk)"""
    email = f"admin.francais.{TS}@sandbox.no"
    prompt = f"""Créez un employé avec les informations suivantes:
- Prénom: Jean
- Nom: Dupont
- E-mail: {email}
- Il doit être administrateur du compte (kontoadministrator)"""

    code, text, elapsed = send_task(prompt)

    employees = search_entities("employee")
    found = find_by_field(employees, "email", email)

    ok = len(found) > 0
    details = ""
    if found:
        emp_id = found[0]["id"]
        r = requests.get(f"{SANDBOX_URL}/employee/entitlement", auth=auth,
                        params={"employeeId": emp_id, "entitlementId": 1, "fields": "id"})
        has_admin = r.json().get("fullResultSize", 0) > 0
        details = f"admin_rolle={'OK' if has_admin else 'MANGLER'}"
        ok = ok and has_admin

    return ok, code, elapsed, details

def task_08():
    """Opprett prosjekt med prosjektleder og kunde (nynorsk)"""
    email = f"prosjektleiar.{TS}@sandbox.no"
    cust_name = f"ProsjektKunde {TS} AS"
    prompt = f"""Opprett eit prosjekt med følgjande informasjon:
- Prosjektnamn: Omega Prosjekt {TS}
- Prosjektnummer: {TS % 10000}
- Startdato: 2026-04-01
- Prosjektleiar: Olav Haugen, e-post {email}
- Kunde: {cust_name}

Prosjektleiaren må opprettast som tilsett med EXTENDED-tilgang og få dei nødvendige rettane (AUTH_CREATE_PROJECT og AUTH_PROJECT_MANAGER)."""

    code, text, elapsed = send_task(prompt)

    projects = search_entities("project")
    found = find_by_field(projects, "name", f"Omega Prosjekt {TS}")

    ok = len(found) > 0
    details = ""
    if found:
        details = f"prosjekt_id={found[0].get('id')}"

    return ok, code, elapsed, details

def task_09():
    """Opprett reiseregning med 3 kostnader (norsk)"""
    prompt = f"""Opprett en reiseregning for den første ansatte i systemet med følgende:
- Tittel: Kundebesøk Bergen {TS}
- Dato: 2026-03-15

Legg til disse kostnadene:
1. Fly tur/retur Oslo-Bergen: 2500 NOK
2. Hotell 1 natt: 1200 NOK
3. Taxi fra flyplass: 450 NOK"""

    code, text, elapsed = send_task(prompt)

    travel = search_entities("travelExpense")
    found = find_by_field(travel, "title", f"Kundebesøk Bergen {TS}")

    ok = len(found) > 0
    details = ""
    if found:
        te_id = found[0]["id"]
        costs = requests.get(f"{SANDBOX_URL}/travelExpense/cost", auth=auth,
                           params={"travelExpenseId": te_id, "fields": "id,costCategory,amountCurrencyIncVat"}).json()
        cost_count = len(costs.get("values", []))
        details = f"kostnader={cost_count}/3"
        ok = ok and cost_count >= 2  # Minst 2 av 3 er godkjent

    return ok, code, elapsed, details

def task_10():
    """Oppdater en eksisterende kunde med ny telefon (engelsk)"""
    # Først opprett en kunde å oppdatere
    cust_name = f"UpdateMe {TS} Ltd"
    r = requests.post(f"{SANDBOX_URL}/customer", auth=auth, json={
        "name": cust_name, "isCustomer": True, "email": f"update.{TS}@test.no"
    })
    cust_id = r.json().get("value", {}).get("id") if r.status_code == 201 else None

    if not cust_id:
        return False, 0, 0, "Kunne ikke opprette testkunde"

    prompt = f"""Update the customer named "{cust_name}" with a new phone number: 22334455"""

    code, text, elapsed = send_task(prompt)

    # Verifiser at telefonnummeret er oppdatert
    updated = requests.get(f"{SANDBOX_URL}/customer/{cust_id}", auth=auth, params={"fields": "*"}).json().get("value", {})
    has_phone = updated.get("phoneNumber") == "22334455" or updated.get("phoneNumberMobile") == "22334455"

    ok = has_phone
    details = f"telefon={updated.get('phoneNumber', 'null')}, mobil={updated.get('phoneNumberMobile', 'null')}"

    return ok, code, elapsed, details


# ============================
# VANSKELIGE oppgaver
# ============================

def task_11():
    """Registrer leverandørfaktura (nynorsk) — den som feilet i konkurransen"""
    # Opprett leverandør først
    supplier_name = f"LevFaktura {TS} AS"
    r = requests.post(f"{SANDBOX_URL}/supplier", auth=auth, json={
        "name": supplier_name
    })
    supplier_id = r.json().get("value", {}).get("id") if r.status_code == 201 else None

    prompt = f"""Registrer ein leverandørfaktura med følgjande informasjon:
- Leverandør: {supplier_name}
- Fakturanummer: LF-{TS}
- Fakturadato: 2026-03-15
- Forfallsdato: 2026-04-15"""

    code, text, elapsed = send_task(prompt)

    # Verifiser
    inv = requests.get(f"{SANDBOX_URL}/supplierInvoice", auth=auth,
                      params={"invoiceDateFrom": "2026-03-01", "invoiceDateTo": "2026-03-31", "fields": "id,invoiceNumber,supplier"}).json()
    found = [i for i in inv.get("values", []) if f"LF-{TS}" in str(i.get("invoiceNumber", ""))]

    ok = len(found) > 0
    details = f"leverandør_opprettet={'OK' if supplier_id else 'FEIL'}, faktura={'FUNNET' if found else 'IKKE FUNNET'}"

    return ok, code, elapsed, details

def task_12():
    """Slett en reiseregning (norsk)"""
    # Opprett en reiseregning å slette
    # Finn en ansatt først
    employees = search_entities("employee")
    if not employees:
        return False, 0, 0, "Ingen ansatte funnet"

    emp_id = employees[0]["id"]
    r = requests.post(f"{SANDBOX_URL}/travelExpense", auth=auth, json={
        "employee": {"id": emp_id},
        "title": f"SlettMeg {TS}",
        "date": "2026-03-10"
    })
    te_id = r.json().get("value", {}).get("id") if r.status_code == 201 else None

    if not te_id:
        return False, 0, 0, f"Kunne ikke opprette test-reiseregning: {r.text[:200]}"

    prompt = f"""Slett reiseregningen med tittel "SlettMeg {TS}"."""

    code, text, elapsed = send_task(prompt)

    # Verifiser at den er slettet
    check = requests.get(f"{SANDBOX_URL}/travelExpense/{te_id}", auth=auth)
    is_deleted = check.status_code == 404 or check.status_code == 410

    # Alternativt: sjekk at den ikke finnes i listen
    if not is_deleted:
        travel = search_entities("travelExpense")
        still_exists = any(t.get("id") == te_id for t in travel)
        is_deleted = not still_exists

    ok = is_deleted
    details = f"te_id={te_id}, slettet={'JA' if is_deleted else 'NEI'}"

    return ok, code, elapsed, details

def task_13():
    """Opprett kontaktperson hos kunde (engelsk)"""
    # Opprett kunde først
    cust_name = f"ContactCorp {TS} AS"
    r = requests.post(f"{SANDBOX_URL}/customer", auth=auth, json={
        "name": cust_name, "isCustomer": True
    })
    cust_id = r.json().get("value", {}).get("id") if r.status_code == 201 else None

    if not cust_id:
        return False, 0, 0, "Kunne ikke opprette testkunde"

    prompt = f"""Create a contact person at the customer "{cust_name}" with these details:
- First name: Sarah
- Last name: Johnson
- Email: sarah.johnson@contactcorp.no
- Phone: 98765432"""

    code, text, elapsed = send_task(prompt)

    contacts = requests.get(f"{SANDBOX_URL}/contact", auth=auth,
                           params={"fields": "id,firstName,lastName,email,customer", "count": 100}).json()
    found = [c for c in contacts.get("values", [])
             if c.get("firstName", "").lower() == "sarah" and c.get("lastName", "").lower() == "johnson"]

    ok = len(found) > 0
    details = f"kunde_id={cust_id}, kontakt={'FUNNET' if found else 'IKKE FUNNET'}"

    return ok, code, elapsed, details

def task_14():
    """Opprett ansatt med prosjektleder-rolle og lag prosjekt (tysk)"""
    email = f"projektleiter.{TS}@sandbox.de"
    prompt = f"""Erstellen Sie einen Mitarbeiter und ein Projekt:

1. Mitarbeiter:
   - Vorname: Hans
   - Nachname: Müller
   - E-Mail: {email}
   - Rolle: Projektmanager (prosjektleder)

2. Projekt:
   - Name: Projekt Alpha {TS}
   - Projektnummer: {TS % 10000 + 1}
   - Startdatum: 2026-04-01
   - Projektleiter: Hans Müller (der oben erstellte Mitarbeiter)"""

    code, text, elapsed = send_task(prompt)

    # Verifiser ansatt
    employees = search_entities("employee")
    found_emp = find_by_field(employees, "email", email)

    # Verifiser prosjekt
    projects = search_entities("project")
    found_proj = find_by_field(projects, "name", f"Projekt Alpha {TS}")

    ok = len(found_emp) > 0 and len(found_proj) > 0
    details = f"ansatt={'OK' if found_emp else 'MANGLER'}, prosjekt={'OK' if found_proj else 'MANGLER'}"

    return ok, code, elapsed, details

def task_15():
    """Lønnskjøring — opprett payslip med fastlønn (norsk)"""
    # Finn en ansatt
    employees = search_entities("employee")
    if not employees:
        return False, 0, 0, "Ingen ansatte funnet"

    emp = employees[0]
    emp_name = f"{emp.get('firstName', '')} {emp.get('lastName', '')}"

    prompt = f"""Opprett en lønnsslipp for ansatt {emp_name} med følgende:
- Dato: 2026-03-31
- År: 2026
- Måned: 3
- Legg til en fastlønn (salaryType for Fastlønn) på 45000 NOK"""

    code, text, elapsed = send_task(prompt)

    # Verifiser — lønnsmodulen kan være deaktivert
    payslips = requests.get(f"{SANDBOX_URL}/salary/payslip", auth=auth,
                           params={"fields": "id,employee,date", "count": 50}).json()

    ok = False
    details = ""
    if payslips.get("values"):
        # Sjekk om det finnes en nylig payslip
        found = [p for p in payslips["values"] if p.get("employee", {}).get("id") == emp["id"]]
        ok = len(found) > 0
        details = f"payslips_funnet={len(found)}"
    elif payslips.get("status") == 403:
        details = "lønnsmodul_deaktivert"
    else:
        details = f"ingen_payslips, status={payslips.get('status', 'ok')}"

    return ok, code, elapsed, details


# ============================
# KJØR ALLE OPPGAVER
# ============================

TASKS = [
    (1, "TIER 1", "Opprett ansatt med telefon og fødselsdato (norsk)", task_01),
    (2, "TIER 1", "Opprett kunde med orgnr og adresse (engelsk)", task_02),
    (3, "TIER 1", "Opprett leverandør med orgnr (tysk)", task_03),
    (4, "TIER 1", "Opprett 3 produkter med ulike MVA-satser (spansk)", task_04),
    (5, "TIER 1", "Opprett avdeling (portugisisk)", task_05),
    (6, "TIER 2", "Opprett kunde+produkt+ordre+faktura (norsk)", task_06),
    (7, "TIER 2", "Opprett ansatt med admin-rolle (fransk)", task_07),
    (8, "TIER 2", "Opprett prosjekt med prosjektleder+kunde (nynorsk)", task_08),
    (9, "TIER 2", "Opprett reiseregning med 3 kostnader (norsk)", task_09),
    (10, "TIER 2", "Oppdater eksisterende kunde med ny telefon (engelsk)", task_10),
    (11, "HARD", "Registrer leverandørfaktura (nynorsk)", task_11),
    (12, "HARD", "Slett en reiseregning (norsk)", task_12),
    (13, "HARD", "Opprett kontaktperson hos kunde (engelsk)", task_13),
    (14, "HARD", "Ansatt m/prosjektleder-rolle + prosjekt (tysk)", task_14),
    (15, "HARD", "Lønnskjøring — payslip med fastlønn (norsk)", task_15),
]

if __name__ == "__main__":
    print("=" * 70)
    print("TRIPLETEX AGENT — 15-OPPGAVE SIMULERING")
    print(f"Agent: {AGENT_URL}")
    print(f"Sandbox: {SANDBOX_URL}")
    print(f"Tidsstempel: {TS}")
    print("=" * 70)

    # Health check
    try:
        r = requests.get(f"{AGENT_URL}/health", timeout=10)
        print(f"\nHealth check: {r.status_code} — {'OK' if r.status_code == 200 else 'FEIL'}")
    except Exception as e:
        print(f"\nHealth check FEILET: {e}")
        sys.exit(1)

    results = []

    for num, tier, desc, task_fn in TASKS:
        print(f"\n{'='*70}")
        print(f"OPPGAVE {num:2d} [{tier}]: {desc}")
        print("-" * 70)

        try:
            ok, code, elapsed, details = task_fn()
            status = "OK" if ok else "FEIL"
            results.append({
                "num": num,
                "tier": tier,
                "desc": desc,
                "ok": ok,
                "code": code,
                "elapsed": elapsed,
                "details": details
            })
            print(f"  Status:  {status}")
            print(f"  HTTP:    {code}")
            print(f"  Tid:     {elapsed:.1f}s")
            print(f"  Detaljer: {details}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            results.append({
                "num": num,
                "tier": tier,
                "desc": desc,
                "ok": False,
                "code": 0,
                "elapsed": 0,
                "details": f"EXCEPTION: {e}"
            })

    # ============================
    # RAPPORT
    # ============================
    print("\n")
    print("=" * 70)
    print("SLUTTRAPPORT")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r["ok"])

    # Per tier
    for tier_name in ["TIER 1", "TIER 2", "HARD"]:
        tier_results = [r for r in results if r["tier"] == tier_name]
        tier_passed = sum(1 for r in tier_results if r["ok"])
        print(f"\n{tier_name}: {tier_passed}/{len(tier_results)} bestått")
        for r in tier_results:
            symbol = "PASS" if r["ok"] else "FAIL"
            print(f"  [{symbol}] #{r['num']:2d} {r['desc']}")
            if r["details"]:
                print(f"         {r['details']}")
            if r["elapsed"] > 0:
                print(f"         Tid: {r['elapsed']:.1f}s")

    print(f"\n{'='*70}")
    print(f"TOTALT: {passed}/{total} bestått ({100*passed/total:.0f}%)")
    print(f"{'='*70}")

    # Feilanalyse
    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"\nFEILANALYSE ({len(failures)} feil):")
        for r in failures:
            print(f"  #{r['num']:2d} [{r['tier']}] {r['desc']}")
            print(f"      Detaljer: {r['details']}")
