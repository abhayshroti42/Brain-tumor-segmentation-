// Backend address; override with VITE_API_URL in frontend/.env if needed
export const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export async function getHealth() {
  const res = await fetch(`${API_URL}/health`)
  if (!res.ok) throw new Error(`Backend returned ${res.status}`)
  return res.json()
}
export async function predictNpz(file) {
  const body = new FormData()
  body.append('file', file)
  const res = await fetch(`${API_URL}/predict/npz`, { method: 'POST', body })
  const json = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(json.detail ?? `Server returned ${res.status}`)
  return json
}

async function getBytes(path) {
  const res = await fetch(`${API_URL}${path}`)
  if (!res.ok) throw new Error(`Backend returned ${res.status} for ${path}`)
  return new Uint8Array(await res.arrayBuffer())
}

export const getMask = (id) => getBytes(`/results/${id}/mask`)
export const getImage = (id, channel) => getBytes(`/results/${id}/image/${channel}`)