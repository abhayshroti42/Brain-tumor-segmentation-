import { useEffect, useRef, useState } from 'react'
import { API_URL, getHealth, predictNpz, getMask, getImage } from './api'
import SliceView, { CLASSES } from './SliceView'
import Info from './Info'

const VIEWS = [
  { axis: 'D', title: 'Slice along D' },
  { axis: 'W', title: 'Slice along W' },
  { axis: 'H', title: 'Slice along H' },
]
const MID = { D: 64, W: 64, H: 64 }
const card = 'rounded-2xl border border-slate-700/60 bg-slate-800/70 p-5 backdrop-blur'
const btn = 'rounded-xl text-sm font-medium transition active:scale-95'
const secondary = `${btn} bg-slate-700 px-4 py-2 hover:bg-slate-600`
// Slice (per axis) holding the most predicted tumor voxels; middle if there are none.
// Volume index = (h * 128 + w) * 128 + d
function densestSlices(mask) {
  const N = 128
  const count = { D: new Array(N).fill(0), W: new Array(N).fill(0), H: new Array(N).fill(0) }
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    count.D[i % N]++
    count.W[Math.floor(i / N) % N]++
    count.H[Math.floor(i / (N * N))]++
  }
  const best = (a) => (Math.max(...a) === 0 ? 64 : a.indexOf(Math.max(...a)))
  return { D: best(count.D), W: best(count.W), H: best(count.H) }
}
export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)
  const [file, setFile] = useState(null)
  const [channel, setChannel] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [mask, setMask] = useState(null)
  const [image, setImage] = useState(null)
  const [slices, setSlices] = useState(MID)
  const [opacity, setOpacity] = useState(0.5)
  const [visible, setVisible] = useState({ 1: true, 2: true, 3: true })
  const [dragging, setDragging] = useState(false)
  const [info, setInfo] = useState(false)
  const [large, setLarge] = useState(false)
  const [zoom, setZoom] = useState(1)
  const inputRef = useRef(null)

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setHealthError(e.message))
  }, [])
  useEffect(() => {
    if (!result) return
    getImage(result.id, channel).then(setImage).catch((e) => setError(e.message))
  }, [result, channel])

  function pick(f) {
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.npz')) return setError('Please choose a .npz file')
    setError(null)
    setFile(f)
  }
  function reset() {
    setFile(null); setResult(null); setMask(null); setImage(null)
    setError(null); setSlices(MID); inputRef.current.value = ''
  }
  async function analyze() {
    setLoading(true); setError(null); setResult(null); setImage(null); setMask(null)
    try {
      const r = await predictNpz(file)
      const m = await getMask(r.id)
      setMask(m)
      setSlices(densestSlices(m))
      setResult(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const ready = health?.model_ready
  const notice = healthError ? 'Cannot reach the backend. Start it on port 8000 and refresh.'
    : health && !ready ? `Model not loaded: ${health.error}` : null
  const dot = healthError ? 'bg-red-500' : !health ? 'bg-slate-500' : ready ? 'bg-emerald-400' : 'bg-amber-400'

  return (
    <div className="min-h-screen text-slate-100"
      style={{ background: 'radial-gradient(circle at 20% 0%, #1e1b4b 0%, #0f172a 45%)' }}>
      <header className="flex items-center justify-between px-6 py-4">
        <h1 className="bg-clip-text text-2xl font-bold text-transparent"
          style={{ backgroundImage: 'linear-gradient(90deg,#a5b4fc,#60a5fa)' }}>
          Brain Tumor Segmentation
        </h1>
        <div className="flex items-center gap-3">
          <span title={ready ? 'Backend ready' : 'Backend not ready'} className={`h-2.5 w-2.5 rounded-full ${dot}`} />
          <button onClick={() => setInfo(true)} aria-label="About"
            className={`${btn} h-8 w-8 border border-slate-600 hover:bg-slate-700`}>i</button>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-4 px-6 pb-10">
        {(notice || error) && (
          <p className="rounded-xl border border-red-700 bg-red-950/70 p-3 text-sm text-red-300">{notice ?? error}</p>
        )}

        <div className={`${card} space-y-4`}>
          <div onClick={() => inputRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]) }}
            className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition ${
              dragging ? 'scale-[1.01] border-indigo-400 bg-indigo-500/10' : 'border-slate-600 hover:border-slate-400'}`}>
            <input ref={inputRef} type="file" accept=".npz" hidden onChange={(e) => pick(e.target.files[0])} />
            <p className="text-2xl text-indigo-300">↑</p>
            <p className="text-slate-200">{file ? file.name : 'Drop a .npz file here, or click to browse'}</p>
            {file && <p className="text-xs text-slate-400">{(file.size / 1e6).toFixed(1)} MB</p>}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm text-slate-400">Display channel</span>
            <div className="inline-flex rounded-xl bg-slate-900 p-1">
              {[0, 1, 2, 3].map((i) => (
                <button key={i} onClick={() => setChannel(i)}
                  className={`${btn} px-3 py-1.5 ${channel === i ? 'bg-indigo-500 text-white shadow' : 'text-slate-400 hover:text-white'}`}>
                  {i}
                </button>
              ))}
            </div>
            <div className="ml-auto flex gap-2">
              <button onClick={reset} className={secondary}>Reset</button>
              <button disabled={!file || !ready || loading} onClick={analyze}
                style={{ background: 'linear-gradient(135deg,#6366f1,#3b82f6)' }}
                className={`${btn} px-6 py-2.5 text-white shadow-lg shadow-blue-500/30 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40`}>
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                    Analyzing...
                  </span>
                ) : 'Analyze'}
              </button>
            </div>
          </div>
        </div>

        {result && mask && image && (
          <>
            <div className={`${card} flex flex-wrap items-center gap-4`}>
              <label className="flex items-center gap-2 text-sm">
                Overlay
                <input type="range" min={0} max={1} step={0.05} value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))} className="accent-indigo-500" />
                <span className="w-10 text-slate-400">{Math.round(opacity * 100)}%</span>
              </label>
              {CLASSES.map((c) => (
                <button key={c.id} aria-pressed={visible[c.id]}
                  onClick={() => setVisible({ ...visible, [c.id]: !visible[c.id] })}
                  className={`${btn} flex items-center gap-2 border px-3 py-1.5 ${
                    visible[c.id] ? 'border-slate-500 bg-slate-700' : 'border-slate-700 text-slate-500 line-through'}`}>
                  <span className="h-3 w-3 rounded-full"
                    style={{ background: `rgb(${c.color})`, opacity: visible[c.id] ? 1 : 0.3 }} />
                  {c.name} <span className="text-slate-400">{result.voxel_counts[c.name]}</span>
                </button>
              ))}
              <div className="ml-auto flex gap-2">
                <button onClick={() => setSlices(MID)} className={secondary}>Center slices</button>
                    <button onClick={() => setSlices(densestSlices(mask))} className={secondary}>Jump to prediction</button>
                                    <button onClick={() => setLarge(!large)} className={secondary}>
                  {large ? 'Smaller view' : 'Enlarge view'}
                </button>
                                <div className="inline-flex rounded-xl bg-slate-900 p-1">
                  {[1, 2, 4, 8].map((z) => (
                    <button key={z} onClick={() => setZoom(z)}
                      className={`${btn} px-3 py-1.5 ${zoom === z ? 'bg-indigo-500 text-white shadow' : 'text-slate-400 hover:text-white'}`}>
                      {z}×
                    </button>
                  ))}
                </div>
                <a href={`${API_URL}/results/${result.id}/mask.npz`} className={secondary}>⬇ Download mask</a>
              </div>
            </div>
              <div className={`grid gap-4 ${large ? '' : 'md:grid-cols-3'}`}>
              {VIEWS.map((v) => (
                <SliceView key={v.axis} title={v.title} axis={v.axis} slice={slices[v.axis]}
                  image={image} mask={mask} opacity={opacity} visible={visible} zoom={zoom}
                  onSlice={(s) => setSlices({ ...slices, [v.axis]: s })} />
              ))}
            </div>
          </>
        )}
      </main>
      <Info open={info} onClose={() => setInfo(false)} />
    </div>
  )
}