"""
Test Tripletex-agenten mot sandbox.
Kjør: python3 test_sandbox.py
"""
import requests
import json
import sys

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

def test_health():
    print("=== Test 1: Health check ===")
    r = requests.get(f"{AGENT_URL}/health", timeout=10)
    print(f"Status: {r.status_code} — {r.text}")
    return r.status_code == 200

def test_create_employee():
    print("\n=== Test 2: Opprett ansatt via agent ===")
    r = requests.post(
        f"{AGENT_URL}/solve",
        json={
            "prompt": "Opprett en ansatt med navn Kari Hansen, e-post kari@hansen.no",
            "files": [],
            "tripletex_credentials": {
                "base_url": SANDBOX_URL,
                "session_token": SANDBOX_TOKEN
            }
        },
        timeout=120
    )
    print(f"Agent response: {r.status_code} — {r.text}")
    return r.status_code == 200

def test_create_customer():
    print("\n=== Test 3: Opprett kunde via agent ===")
    r = requests.post(
        f"{AGENT_URL}/solve",
        json={
            "prompt": "Opprett en kunde med navn Testbedrift AS, e-post post@testbedrift.no",
            "files": [],
            "tripletex_credentials": {
                "base_url": SANDBOX_URL,
                "session_token": SANDBOX_TOKEN
            }
        },
        timeout=120
    )
    print(f"Agent response: {r.status_code} — {r.text}")
    return r.status_code == 200

def verify_sandbox():
    print("\n=== Verifisering: Sjekk sandbox ===")
    auth = ("0", SANDBOX_TOKEN)

    r = requests.get(f"{SANDBOX_URL}/employee", auth=auth,
                     params={"fields": "id,firstName,lastName,email"}, timeout=10)
    employees = r.json().get("values", [])
    print(f"Ansatte ({len(employees)}):")
    for e in employees:
        print(f"  - {e.get('firstName')} {e.get('lastName')} ({e.get('email')})")

    r = requests.get(f"{SANDBOX_URL}/customer", auth=auth,
                     params={"fields": "id,name,email", "isCustomer": True}, timeout=10)
    customers = r.json().get("values", [])
    print(f"Kunder ({len(customers)}):")
    for c in customers:
        print(f"  - {c.get('name')} ({c.get('email')})")

if __name__ == "__main__":
    print("Tripletex Agent — Sandbox Test\n")

    if not test_health():
        print("FEIL: Agent er nede!")
        sys.exit(1)

    test_create_employee()
    test_create_customer()
    verify_sandbox()

    print("\n=== Ferdig ===")
