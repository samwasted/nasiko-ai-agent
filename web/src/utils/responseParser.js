export function extractContextId(response) {
  return response?.result?.contextId || null;
}

export function extractTaskStatus(response) {
  return response?.result?.status?.state || 'unknown';
}

export function extractLatestArtifactText(response) {
  const artifacts = response?.result?.artifacts;
  if (!Array.isArray(artifacts) || artifacts.length === 0) {
    return '';
  }

  const latest = artifacts[artifacts.length - 1];
  const parts = Array.isArray(latest?.parts) ? latest.parts : [];
  const textParts = parts
    .filter((part) => part && part.kind === 'text' && typeof part.text === 'string')
    .map((part) => part.text);

  return textParts.join('\n').trim();
}

export function buildDebugView(response) {
  return {
    contextId: extractContextId(response),
    taskStatus: extractTaskStatus(response),
    artifactCount: Array.isArray(response?.result?.artifacts)
      ? response.result.artifacts.length
      : 0,
    latestText: extractLatestArtifactText(response),
  };
}
