export function buildInitialMessagePayload(text, messageId) {
  return {
    jsonrpc: '2.0',
    id: messageId,
    method: 'message/send',
    params: {
      message: {
        messageId,
        timestamp: new Date().toISOString(),
        role: 'user',
        parts: [{ text }],
      },
      metadata: {},
    },
  };
}

export function buildContinuationPayload(text, contextId, messageId) {
  return {
    jsonrpc: '2.0',
    id: messageId,
    method: 'message/send',
    params: {
      message: {
        messageId,
        contextId,
        timestamp: new Date().toISOString(),
        role: 'user',
        parts: [{ text }],
      },
      metadata: {},
    },
  };
}

export async function sendJsonRpcMessage(endpoint, payload) {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => {
    throw new Error('Server returned non-JSON response');
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${JSON.stringify(data)}`);
  }

  if (data.error) {
    throw new Error(`RPC error ${data.error.code}: ${data.error.message}`);
  }

  return data;
}
