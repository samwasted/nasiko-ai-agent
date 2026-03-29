import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

export default function ChatPanel({
  messages,
  input,
  setInput,
  isLoading,
  hasContext,
  contextId,
  onSend,
  onSendYes,
  error,
}) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const submit = (event) => {
    event.preventDefault();
    onSend(input);
  };

  return (
    <section className="card chat-card">
      <div className="chat-header">
        <h2>Chat Tester</h2>
        <div className="context-pill">
          {hasContext ? `Context: ${contextId}` : 'Context: none'}
        </div>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="chat-list" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="hint">Send a prompt to start a session.</div>
        ) : null}

        {messages.map((message) => (
          <article
            key={message.id}
            className={`bubble ${message.role === 'user' ? 'user' : 'agent'}`}
          >
            <header>{message.role.toUpperCase()}</header>
            <div className="md-content">
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                {message.text}
              </ReactMarkdown>
            </div>
          </article>
        ))}
      </div>

      <form className="chat-form" onSubmit={submit}>
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          rows={4}
          placeholder="Type your prompt here..."
        />
        <div className="actions">
          <button type="submit" className="btn btn-primary" disabled={isLoading || !input.trim()}>
            {isLoading ? 'Sending...' : 'Send'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onSendYes}
            disabled={isLoading || !hasContext}
          >
            Send yes
          </button>
        </div>
      </form>
    </section>
  );
}
