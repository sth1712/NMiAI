#!/bin/bash
# Deploy Tripletex-agent til Cloud Run
# Bruk: ./deploy.sh DIN_GEMINI_API_NOEKKEL

if [ -z "$1" ]; then
    echo "Bruk: ./deploy.sh DIN_GEMINI_API_NOEKKEL"
    echo "Hent nøkkel fra aistudio.google.com"
    exit 1
fi

echo "Deployer med Gemini API-nøkkel: ${1:0:8}..."

gcloud run deploy tripletex-agent \
    --source . \
    --region europe-north1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --update-env-vars "GEMINI_API_KEY=$1"

echo ""
echo "Test med: python3 ~/NMiAI/tripletex-agent/test_sandbox.py"
