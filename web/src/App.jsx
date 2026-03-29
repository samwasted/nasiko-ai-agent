import { useEffect, useMemo, useState } from 'react';
import ChatPanel from './components/ChatPanel';
import DebugPanel from './components/DebugPanel';
import PresetsPanel from './components/PresetsPanel';
import {
  buildContinuationPayload,
  buildInitialMessagePayload,
  sendJsonRpcMessage,
} from './api/a2aClient';
import { buildDebugView, extractContextId, extractLatestArtifactText } from './utils/responseParser';

const STORAGE_KEY = 'black-swan-chat-ui-v1';

function canUseStorage() {
  try {
    return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
  } catch {
    return false;
  }
}

function createMessage(role, text) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    role,
    text,
  };
}

function loadStoredState() {
  if (!canUseStorage()) {
    return {
      endpoint: 'http://localhost:5000/',
      contextId: '',
      messages: [],
    };
  }

  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}');
    return {
      endpoint: parsed.endpoint || 'http://localhost:5000/',
      contextId: parsed.contextId || '',
      messages: Array.isArray(parsed.messages) ? parsed.messages : [],
    };
  } catch {
    return {
      endpoint: 'http://localhost:5000/',
      contextId: '',
      messages: [],
    };
  }
}

export default function App() {
  const initial = useMemo(loadStoredState, []);

  const [endpoint, setEndpoint] = useState(initial.endpoint);
  const [contextId, setContextId] = useState(initial.contextId);
  const [messages, setMessages] = useState(initial.messages);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [rawResponse, setRawResponse] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);

  useEffect(() => {
    if (!canUseStorage()) {
      return;
    }

    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          endpoint,
          contextId,
          messages,
        })
      );
    } catch {
      // Ignore storage write issues to avoid crashing the UI.
    }
  }, [endpoint, contextId, messages]);

  const append = (role, text) => {
    setMessages((prev) => [...prev, createMessage(role, text)]);
  };

  const sendMessage = async (text, forceContinuation = false) => {
    const trimmed = (text || '').trim();
    if (!trimmed) {
      return;
    }

    setError('');
    setIsLoading(true);
    append('user', trimmed);

    const messageId = `msg-${Date.now()}`;
    const useContinuation = forceContinuation || !!contextId;
    const payload = useContinuation
      ? buildContinuationPayload(trimmed, contextId, messageId)
      : buildInitialMessagePayload(trimmed, messageId);

    try {
      const response = await sendJsonRpcMessage(endpoint, payload);
      setRawResponse(response);
      setDebugInfo(buildDebugView(response));

      const nextContext = extractContextId(response);
      if (nextContext) {
        setContextId(nextContext);
      }

      const textReply = extractLatestArtifactText(response);
      append('agent', textReply || JSON.stringify(response, null, 2));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      append('agent', `Error: ${message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = (text) => {
    sendMessage(text, false);
    setInput('');
  };

  const handleSendYes = () => {
    sendMessage('yes', true);
  };

  return (
    <main className="app">
      <header className="topbar">
        <h1>Black Swan Chat Tester</h1>
        <p>Test A2A message/send flow with context continuation and debug visibility.</p>
      </header>

      <section className="layout">
        <div className="left-col">
          <PresetsPanel onSelect={setInput} />
          <ChatPanel
            messages={messages}
            input={input}
            setInput={setInput}
            isLoading={isLoading}
            hasContext={!!contextId}
            contextId={contextId}
            onSend={handleSend}
            onSendYes={handleSendYes}
            error={error}
          />
        </div>

        <div className="right-col">
          <DebugPanel
            endpoint={endpoint}
            setEndpoint={setEndpoint}
            debugInfo={debugInfo}
            rawResponse={rawResponse}
          />
        </div>
      </section>
    </main>
  );
}
