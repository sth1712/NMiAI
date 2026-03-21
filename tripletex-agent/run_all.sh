#!/bin/bash
# Alt-i-ett: pull → deploy → test → logger
# Bruk: bash run_all.sh DIN_GEMINI_NØKKEL

if [ -z "$1" ]; then
    echo "Bruk: bash run_all.sh DIN_GEMINI_NØKKEL"
    exit 1
fi

echo "=== 1. Git pull ==="
git pull

echo ""
echo "=== 2. Deploy ==="
bash deploy.sh "$1"

echo ""
echo "=== 3. Venter 5 sek på oppstart ==="
sleep 5

echo ""
echo "=== 4. Health check ==="
curl -s https://tripletex-agent-421519138388.europe-north1.run.app/health
echo ""

echo ""
echo "=== 5. Kjører kritiske tester ==="
python3 test_critical.py

echo ""
echo "=== 6. Logger (kompakt) ==="
gcloud run services logs read tripletex-agent --region europe-north1 --limit 100 --format="value(textPayload)" 2>/dev/null | grep -E "(Task:|Params:|Replaced|→ [0-9]|Error:|Done)"
