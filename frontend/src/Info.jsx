import { useEffect } from 'react'

export default function Info({ open, onClose }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onClick={onClose}>
      <div className="max-w-lg rounded-2xl border border-slate-700 bg-slate-800 p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}>
        <h2 className="mb-3 text-lg font-semibold">About this tool</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm text-slate-300">
          <li>Research use only. Not a medical device and not a diagnosis.</li>
          <li>The overlay shows what the model predicted, not whether it is correct. No Dice or IoU
            is computed, because no ground-truth mask is supplied.</li>
          <li>The dataset creator describes channels 0 to 3 as T1, T1ce, T2 and FLAIR, but this
            mapping has not been independently verified.</li>
          <li>Only preprocessed .npz patches are supported. Raw MRI (NIfTI) is disabled because its
            preprocessing is unverified.</li>
        </ul>
        <button onClick={onClose}
          className="mt-5 rounded-lg bg-slate-700 px-4 py-2 text-sm transition hover:bg-slate-600 active:scale-95">
          Close
        </button>
      </div>
    </div>
  )
}