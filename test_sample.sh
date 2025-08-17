#!/bin/bash

response=$(curl -s -X POST http://localhost:8082/v1/agents/sample \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "get menu"}
    ],
    "settings": {},
    "conversation_id": "sample123"
  }')

echo "Response: $response"

if echo "$response" | grep -q '"conversation_id":"sample123"' && echo "$response" | grep -q '"answer":'; then
  echo "Sample agent integration test passed."
  exit 0
else
  echo "Sample agent integration test failed."
  exit 1
fi