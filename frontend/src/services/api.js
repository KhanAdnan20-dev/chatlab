const BASE_URL = (import.meta.env && import.meta.env.VITE_API_BASE_URL) || 'http://127.0.0.1:8000';

export async function sendMessage(message, sessionId = null, socraticMode = false, requestId = null, depthMode = 'auto', modelChoice = 'gemma') {
  const payload = { message, session_id: sessionId, socratic_mode: socraticMode, depth_mode: depthMode, model_choice: modelChoice };
  if (requestId) payload.request_id = requestId;
  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error (${response.status})`);
  }
  return response.json();
}

export async function streamMessage(message, sessionId = null, onToken, onMetadata, onError, socraticMode = false, requestId = null, depthMode = 'auto', modelChoice = 'gemma', onThink = null) {
  try {
    const payload = { message, session_id: sessionId, socratic_mode: socraticMode, depth_mode: depthMode, model_choice: modelChoice };
    if (requestId) payload.request_id = requestId;
    const response = await fetch(`${BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // preserve remainder

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            if (data.type === 'token') {
              onToken(data.content);
            } else if (data.type === 'think') {
              if (onThink) onThink(data.content);
            } else if (data.type === 'metadata') {
              onMetadata(data);
            }
          } catch (e) {
            // skip unparseable SSE line
          }
        }
      }
    }
  } catch (err) {
    if (onError) onError(err);
    else throw err;
  }
}


export async function startQuiz(topic = 'all', difficulty = 'medium', numQuestions = 5) {
  const response = await fetch(`${BASE_URL}/quiz/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic, difficulty, num_questions: numQuestions }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to start quiz session');
  }
  return response.json();
}

export async function submitQuizAnswer(sessionId, answerIndex) {
  const response = await fetch(`${BASE_URL}/quiz/${sessionId}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer_index: answerIndex }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to submit answer');
  }
  return response.json();
}

export async function nextQuizQuestion(sessionId) {
  const response = await fetch(`${BASE_URL}/quiz/${sessionId}/next`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to advance to next question');
  }
  return response.json();
}

export async function getQuizScore(sessionId) {
  const response = await fetch(`${BASE_URL}/quiz/${sessionId}/score`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to retrieve score');
  }
  return response.json();
}

export async function finishQuiz(sessionId) {
  const response = await fetch(`${BASE_URL}/quiz/${sessionId}/finish`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to finish quiz session');
  }
  return response.json();
}
