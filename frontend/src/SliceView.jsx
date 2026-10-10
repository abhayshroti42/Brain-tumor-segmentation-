import { useEffect, useRef } from 'react'

export const N = 128
// Class colors: 1 = NCR/NET (red), 2 = Edema (yellow), 3 = Enhancing Tumor (blue)
export const CLASSES = [
  { id: 1, name: 'NCR/NET', color: [239, 68, 68] },
  { id: 2, name: 'Edema', color: [250, 204, 21] },
  { id: 3, name: 'Enhancing Tumor', color: [59, 130, 246] },
]
const COLOR = Object.fromEntries(CLASSES.map((c) => [c.id, c.color]))

// Volumes are (H, W, D) in C order: index = (h * N + w) * N + d
const at = (h, w, d) => (h * N + w) * N + d

// Maps (row, col) on screen to a volume index for the chosen axis and slice
function planeIndex(axis, s) {
  if (axis === 'D') return (r, c) => at(r, c, s) // rows = H, cols = W
  if (axis === 'W') return (r, c) => at(r, s, c) // rows = H, cols = D
  return (r, c) => at(s, r, c) // axis H: rows = W, cols = D
}

// Draws one slice. z = zoom factor; (r0, c0) = top-left of the visible window.
// With z = 1 and (0, 0) the whole 128 x 128 slice is shown.
function draw(canvas, idx, image, mask, opacity, visible, z, r0, c0) {
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(N, N)
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const i = idx(r0 + Math.floor(r / z), c0 + Math.floor(c / z))
      const v = image[i]
      const col = mask && visible[mask[i]] ? COLOR[mask[i]] : null
      const p = (r * N + c) * 4
      for (let k = 0; k < 3; k++) {
        img.data[p + k] = col ? v * (1 - opacity) + col[k] * opacity : v
      }
      img.data[p + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

// Centroid (row, col) of predicted voxels in this slice; slice center if none
function focusOf(idx, mask) {
  let sr = 0
  let sc = 0
  let n = 0
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      if (mask[idx(r, c)]) {
        sr += r
        sc += c
        n++
      }
    }
  }
  return n ? [sr / n, sc / n] : [N / 2, N / 2]
}

export default function SliceView({
  title, axis, slice, onSlice, image, mask, opacity, visible, zoom = 1,
}) {
  const left = useRef(null)
  const right = useRef(null)
  const wrap = useRef(null)

  useEffect(() => {
    const idx = planeIndex(axis, slice)
    const [fr, fc] = focusOf(idx, mask)
    const size = N / zoom
    const r0 = Math.round(Math.min(Math.max(fr - size / 2, 0), N - size))
    const c0 = Math.round(Math.min(Math.max(fc - size / 2, 0), N - size))
    draw(left.current, idx, image, null, 0, visible, zoom, r0, c0) // MRI only
    draw(right.current, idx, image, mask, opacity, visible, zoom, r0, c0) // MRI + prediction
  }, [axis, slice, image, mask, opacity, visible, zoom])

  // Mouse wheel over the images changes the slice (non-passive so the page doesn't scroll)
  useEffect(() => {
    const el = wrap.current
    const onWheel = (e) => {
      e.preventDefault()
      onSlice(Math.min(N - 1, Math.max(0, slice + (e.deltaY > 0 ? 1 : -1))))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [slice, onSlice])

  const cls = 'w-full bg-black'
  const style = { imageRendering: 'pixelated' }
  return (
    <div className="rounded-2xl border border-slate-700/60 bg-slate-800/70 p-3 transition hover:border-indigo-400/60">
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="font-medium">{title}</span>
        <span className="rounded-full bg-slate-900 px-2 py-0.5 text-xs text-slate-300">
          {slice + 1} / {N}
        </span>
      </div>
      <div ref={wrap} title="Scroll to change slice" className="grid grid-cols-2 gap-1.5">
        <div>
          <canvas ref={left} width={N} height={N} className={cls} style={style} />
          <p className="mt-1 text-center text-xs text-slate-400">MRI</p>
        </div>
        <div>
          <canvas ref={right} width={N} height={N} className={cls} style={style} />
          <p className="mt-1 text-center text-xs text-slate-400">Prediction</p>
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={N - 1}
        value={slice}
        onChange={(e) => onSlice(Number(e.target.value))}
        className="mt-3 w-full accent-indigo-500"
      />
    </div>
  )
}