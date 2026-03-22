#!/bin/bash
# Analyser logger automatisk — kjør etter submissions
# Bruk: bash analyze_logs.sh

echo "=== LOGG-ANALYSE $(date) ==="
echo ""

echo "--- ALLE OPPGAVER (siste 30) ---"
gcloud run services logs read tripletex-agent --region europe-north1 --limit 2000 --format="value(textPayload)" 2>/dev/null | grep "INFO:main:Task:" | tail -30

echo ""
echo "--- FEIL-OVERSIKT ---"
gcloud run services logs read tripletex-agent --region europe-north1 --limit 2000 --format="value(textPayload)" 2>/dev/null | grep -E "(Error:|WARNING)" | tail -20

echo ""
echo "--- RESULTATER (success/failed) ---"
gcloud run services logs read tripletex-agent --region europe-north1 --limit 2000 --format="value(textPayload)" 2>/dev/null | grep "Done —" | tail -30

echo ""
echo "--- PLACEHOLDER-ERSTATNINGER ---"
gcloud run services logs read tripletex-agent --region europe-north1 --limit 2000 --format="value(textPayload)" 2>/dev/null | grep "Replaced" | tail -10

echo ""
echo "--- SAME-ACCOUNT FIXES ---"
gcloud run services logs read tripletex-agent --region europe-north1 --limit 2000 --format="value(textPayload)" 2>/dev/null | grep "Fixed posting" | tail -10
