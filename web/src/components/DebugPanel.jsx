export default function DebugPanel({ endpoint, setEndpoint, debugInfo, rawResponse }) {
  return (
    <section className="card debug-card">
      <h2>Debug</h2>

      <label className="label" htmlFor="endpoint">
        Endpoint
      </label>
      <input
        id="endpoint"
        className="endpoint-input"
        value={endpoint}
        onChange={(event) => setEndpoint(event.target.value)}
      />

      <div className="debug-grid">
        <div>
          <strong>Context ID</strong>
          <div>{debugInfo?.contextId || '-'}</div>
        </div>
        <div>
          <strong>Task Status</strong>
          <div>{debugInfo?.taskStatus || '-'}</div>
        </div>
        <div>
          <strong>Artifact Count</strong>
          <div>{debugInfo?.artifactCount ?? 0}</div>
        </div>
      </div>

      <h3>Latest Text</h3>
      <pre className="raw-block">{debugInfo?.latestText || ''}</pre>

      <h3>Raw JSON</h3>
      <pre className="raw-block">{rawResponse ? JSON.stringify(rawResponse, null, 2) : ''}</pre>
    </section>
  );
}
