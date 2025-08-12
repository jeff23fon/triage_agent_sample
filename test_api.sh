#!/bin/bash

response=$(curl -s -X POST http://localhost:8082/v1/agents/triage \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is your name?"}
    ],
    "settings": {},
    "conversation_id": "abc123"
  }')

echo "Response: $response"

if echo "$response" | grep -q '"conversation_id":"abc123"' && echo "$response" | grep -q '"answer":'; then
  echo "Integration test passed."
  exit 0
else
  echo "Integration test failed."
  exit 1
fi