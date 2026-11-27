export async function api(path, key, body) {
  const multipart = body instanceof FormData
  const response = await fetch(`/v1/${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {
      ...(key ? { Authorization: `Bearer ${key.trim()}` } : {}),
      ...(multipart || body === undefined ? {} : { 'Content-Type': 'application/json' }),
    },
    body: multipart ? body : JSON.stringify(body),
  })
  if (response.ok && response.headers.get('content-type')?.startsWith('audio/')) {
    return response.blob()
  }
  const result = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = result?.detail
    throw new Error(typeof detail === 'string' ? detail : detail?.[0]?.msg || 'The API is unavailable. Check that the backend is running.')
  }
  if (result === null) throw new Error('The backend did not return an API response.')
  return result
}

export function fileForm(file) {
  const form = new FormData()
  form.append('file', file)
  return form
}
