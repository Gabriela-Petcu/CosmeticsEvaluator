import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { getHistory, deleteEvaluation, evaluateById, getProducts } from '../api/evaluate'

const VERDICT_STYLES = {
  'Recomandat': { border: 'border-green-200', bg: 'bg-green-50', badge: 'bg-green-100 text-green-800', score: 'text-green-600' },
  'Nerecomandat': { border: 'border-red-200', bg: 'bg-red-50', badge: 'bg-red-100 text-red-800', score: 'text-red-500' },
  default: { border: 'border-amber-200', bg: 'bg-amber-50', badge: 'bg-amber-100 text-amber-800', score: 'text-amber-600' },
}

function getStyle(verdict) {
  return VERDICT_STYLES[verdict] || VERDICT_STYLES.default
}

export default function HistoryPage() {
  const { isAuthenticated } = useAuth()
  const navigate = useNavigate()

  const [history, setHistory] = useState([])
  const [filtered, setFiltered] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('toate')
  const [sort, setSort] = useState('data')
  const [selected, setSelected] = useState(null)
  const [reEvaluating, setReEvaluating] = useState(false)
  const [deleteModal, setDeleteModal] = useState(null) // id-ul evaluării de șters

  useEffect(() => {
    if (!isAuthenticated) { navigate('/login'); return }
    getHistory()
      .then(res => {
        const data = res.data || []
        setHistory(data)
        setFiltered(data)
        if (data.length > 0) setSelected(data[0])
      })
      .catch(() => setHistory([]))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    let result = [...history]
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter(e =>
        e.name?.toLowerCase().includes(q) ||
        e.brand?.toLowerCase().includes(q) ||
        e.productId?.toLowerCase().includes(q)
      )
    }
    if (filter === 'rec') result = result.filter(e => e.finalVerdict === 'Recomandat')
    else if (filter === 'no') result = result.filter(e => e.finalVerdict === 'Nerecomandat')
    else if (filter === 'other') result = result.filter(e => e.finalVerdict !== 'Recomandat' && e.finalVerdict !== 'Nerecomandat')

    if (sort === 'data') result.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    else if (sort === 'prob_desc') result.sort((a, b) => (b.mlProbability || 0) - (a.mlProbability || 0))
    else if (sort === 'prob_asc') result.sort((a, b) => (a.mlProbability || 0) - (b.mlProbability || 0))

    setFiltered(result)
  }, [search, filter, sort, history])

  const confirmDelete = async () => {
    if (!deleteModal) return
    try {
      await deleteEvaluation(deleteModal)
      const updated = history.filter(e => e.id !== deleteModal)
      setHistory(updated)
      if (selected?.id === deleteModal) setSelected(updated[0] || null)
    } catch {
      alert('Eroare la ștergere.')
    } finally {
      setDeleteModal(null)
    }
  }

  const handleViewResult = async () => {
    if (!selected) return
    setReEvaluating(true)
    try {
      const numericId = parseInt(selected.productId)
      if (!isNaN(numericId) && numericId > 0) {
        const res = await evaluateById(numericId)
        sessionStorage.setItem('skiniq_result', JSON.stringify(res.data))
        navigate('/result/latest')
        return
      }
      const productsRes = await getProducts()
      const products = productsRes.data || []
      const found = products.find(p => {
        const pName = (p.name || '').toLowerCase().trim()
        const sName = (selected.name || selected.productId || '').toLowerCase().trim()
        return pName === sName || pName.includes(sName) || sName.includes(pName)
      })
      if (found) {
        const res = await evaluateById(found.id)
        sessionStorage.setItem('skiniq_result', JSON.stringify(res.data))
        navigate('/result/latest')
        return
      }
      const partial = {
        scorFinal: 0,
        merita: selected.finalVerdict === 'Recomandat' ? 1 : 0,
        meritaML: selected.finalVerdict === 'Recomandat' ? 1 : 0,
        probabilitateML: selected.mlProbability || 0,
        fitScore: 0,
        sePotriveste: 0,
        verdictFinal: selected.finalVerdict || '',
        explicatieFinala: '',
        motivePozitive: [],
        motiveNegative: [],
        topFactoriML: [],
        productName: selected.name || selected.productId,
        brand: selected.brand || '',
        price: selected.price || null,
      }
      sessionStorage.setItem('skiniq_result', JSON.stringify(partial))
      navigate('/result/latest')
    } catch (err) {
      console.error('handleViewResult error:', err)
    } finally {
      setReEvaluating(false)
    }
  }

  const recCount = history.filter(h => h.finalVerdict === 'Recomandat').length
  const noCount = history.filter(h => h.finalVerdict === 'Nerecomandat').length
  const otherCount = history.length - recCount - noCount

  return (
    <div>
      {/* MODAL ȘTERGERE */}
      {deleteModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center px-4"
          onClick={() => setDeleteModal(null)}>
          <div className="bg-white rounded-2xl p-8 max-w-sm w-full shadow-xl text-center flex flex-col gap-5"
            onClick={e => e.stopPropagation()}>
            <div className="text-4xl">🗑️</div>
            <h2 className="font-serif text-2xl font-light text-ink">
              Ștergi această<br />
              <em className="italic text-rose-primary">evaluare?</em>
            </h2>
            <p className="text-sm text-muted leading-relaxed">
              Această acțiune este permanentă și nu poate fi anulată. Evaluarea va fi eliminată din istoricul tău.
            </p>
            <div className="flex flex-col gap-2">
              <button onClick={confirmDelete}
                className="flex items-center justify-center gap-2 py-3 px-6 bg-red-500 text-white rounded-lg text-sm font-medium hover:bg-red-600 transition-colors cursor-pointer">
                🗑️ da, șterge evaluarea
              </button>
              <button onClick={() => setDeleteModal(null)}
                className="btn-outline py-3">
                nu, păstrează-o
              </button>
            </div>
          </div>
        </div>
      )}

      {/* HERO */}
      <div className="bg-cream-warm border-b border-rose-border px-9 py-7 flex items-end justify-between">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <div className="w-5 h-px bg-rose-primary" />
            <span className="text-xs tracking-widest text-rose-primary font-medium">ISTORICUL MEU</span>
          </div>
          <h1 className="font-serif text-3xl font-light text-ink">
            Toate evaluările <em className="italic text-rose-primary">tale.</em>
          </h1>
          <p className="text-sm text-muted mt-2">Produsele pe care le-ai analizat, sortate cronologic.</p>
        </div>
        <div className="flex gap-3">
          {[
            { num: history.length, label: 'evaluate', color: 'text-rose-primary' },
            { num: recCount, label: 'recomandate', color: 'text-green-600' },
            { num: noCount, label: 'nerecomandate', color: 'text-red-500' },
            { num: otherCount, label: 'incerte', color: 'text-amber-600' },
          ].map(s => (
            <div key={s.label} className="bg-rose-light border border-rose-border rounded-xl px-5 py-3 text-center">
              <div className={`font-serif text-2xl font-light ${s.color}`}>{s.num}</div>
              <div className="text-xs text-muted mt-1">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* TOOLBAR */}
      <div className="px-9 py-3.5 border-b border-rose-border bg-cream flex items-center gap-3 flex-wrap">
        <input
          className="input-field max-w-[220px] py-2"
          placeholder="Caută în istoric…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div className="flex gap-2">
          {[
            { id: 'toate', label: 'toate' },
            { id: 'rec', label: 'recomandate' },
            { id: 'no', label: 'nerecomandate' },
            { id: 'other', label: 'incerte' },
          ].map(f => (
            <button key={f.id} onClick={() => setFilter(f.id)}
              className={`text-xs px-4 py-1.5 rounded-full border transition-colors cursor-pointer
                ${filter === f.id
                  ? 'bg-rose-light border-rose-mid text-rose-dark font-medium'
                  : 'border-gray-200 text-muted hover:border-rose-mid bg-white'}`}>
              {f.label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs text-muted">
          <span>sortează</span>
          <select value={sort} onChange={e => setSort(e.target.value)}
            className="border border-gray-200 rounded-lg text-xs px-2 py-1.5 bg-white outline-none cursor-pointer">
            <option value="data">dată — recent</option>
            <option value="prob_desc">probabilitate — descrescător</option>
            <option value="prob_asc">probabilitate — crescător</option>
          </select>
        </div>
      </div>

      {/* BODY */}
      <div className="grid grid-cols-[1fr_300px] min-h-[calc(100vh-260px)]">

        {/* LISTA */}
        <div className="border-r border-rose-border px-9 py-6">
          {loading ? (
            <div className="text-center py-16 text-muted animate-pulse text-sm">se încarcă...</div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-4xl mb-3">📋</div>
              <div className="text-sm text-muted mb-2">
                {history.length === 0 ? 'Nu ai nicio evaluare încă.' : 'Niciun rezultat pentru filtrele selectate.'}
              </div>
              {history.length === 0 && (
                <Link to="/evaluate" className="text-xs text-rose-primary hover:underline">
                  Evaluează primul tău produs →
                </Link>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-soft tracking-widest">EVALUĂRI</span>
                <span className="text-xs text-soft">{filtered.length} produse</span>
              </div>
              {filtered.map(entry => {
                const style = getStyle(entry.finalVerdict)
                const isSelected = selected?.id === entry.id
                const prob = entry.mlProbability ? Math.round(entry.mlProbability * 100) : null
                return (
                  <div key={entry.id}
                    onClick={() => setSelected(entry)}
                    className={`flex items-center gap-4 p-4 rounded-xl border cursor-pointer transition-all
                      ${isSelected
                        ? `${style.border} ${style.bg}`
                        : 'border-rose-border bg-white hover:border-rose-mid'}`}>
                    <div className="w-10 h-10 rounded-lg bg-rose-light flex items-center justify-center flex-shrink-0 text-lg">
                      🧴
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium tracking-widest text-soft">{entry.brand}</div>
                      <div className="text-sm font-medium text-ink mt-0.5 truncate">{entry.name || entry.productId}</div>
                      <div className="text-xs text-muted mt-1">
                        {new Date(entry.createdAt).toLocaleDateString('ro-RO', { day: 'numeric', month: 'long', year: 'numeric' })}
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1.5 flex-shrink-0">
                      {prob !== null && (
                        <div className={`font-serif text-xl font-light ${style.score}`}>{prob}%</div>
                      )}
                      <div className={`text-xs px-2 py-0.5 rounded font-medium ${style.badge}`}>
                        {entry.finalVerdict || 'necunoscut'}
                      </div>
                    </div>
                    <button
                      onClick={e => { e.stopPropagation(); setDeleteModal(entry.id) }}
                      className="text-soft hover:text-red-400 transition-colors text-sm flex-shrink-0 cursor-pointer"
                      title="șterge">
                      🗑️
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* SIDEBAR */}
        <div className="px-6 py-6 bg-cream-warm flex flex-col gap-5">
          <h2 className="section-title border-b border-rose-border pb-3">Detalii evaluare</h2>

          {selected ? (
            <>
              <div>
                <div className="w-14 h-16 rounded-xl bg-rose-light border border-rose-border flex items-center justify-center text-2xl mb-3">🧴</div>
                <div className="text-xs font-medium tracking-widest text-soft">{selected.brand}</div>
                <div className="text-base font-medium text-ink mt-1 leading-tight">{selected.name || selected.productId}</div>
                <div className="text-xs text-muted mt-2">
                  {new Date(selected.createdAt).toLocaleDateString('ro-RO', { day: 'numeric', month: 'long', year: 'numeric' })}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs text-muted">PROBABILITATE ML</div>
                  <div className="text-sm font-medium text-rose-primary">
                    {selected.mlProbability ? `${Math.round(selected.mlProbability * 100)}%` : '—'}
                  </div>
                </div>
                <div className="h-1.5 bg-rose-border rounded-full overflow-hidden">
                  <div className="h-full bg-rose-primary rounded-full"
                    style={{ width: `${selected.mlProbability ? selected.mlProbability * 100 : 0}%` }} />
                </div>
              </div>

              <div className={`p-3 rounded-xl border ${getStyle(selected.finalVerdict).border} ${getStyle(selected.finalVerdict).bg}`}>
                <div className={`text-xs font-medium tracking-widest mb-1 ${getStyle(selected.finalVerdict).score}`}>
                  {(selected.finalVerdict || 'NECUNOSCUT').toUpperCase()}
                </div>
                <div className={`text-xs leading-relaxed ${getStyle(selected.finalVerdict).score}`}>
                  {selected.finalVerdict === 'Recomandat'
                    ? 'Produsul este evaluat pozitiv și compatibil cu profilul tău.'
                    : selected.finalVerdict === 'Nerecomandat'
                      ? 'Produsul nu este recomandat pe baza evaluării complete.'
                      : 'Evaluare cu rezultat incert — verifică detaliile.'}
                </div>
              </div>

              <div className="flex flex-col gap-2 mt-auto">
                <button onClick={handleViewResult} disabled={reEvaluating}
                  className="btn-primary flex items-center justify-center gap-2 py-2.5 text-xs disabled:opacity-60 cursor-pointer">
                  {reEvaluating ? '⚙️ se încarcă...' : '👁️ vezi rezultatul complet'}
                </button>
                <button onClick={() => setDeleteModal(selected.id)}
                  className="flex items-center justify-center gap-2 py-2.5 text-xs border border-red-200 text-red-400 rounded-lg hover:bg-red-50 transition-colors cursor-pointer">
                  🗑️ șterge evaluarea
                </button>
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-muted text-sm">
              <div className="text-3xl mb-2">👆</div>
              Selectează o evaluare din listă
            </div>
          )}
        </div>
      </div>
    </div>
  )
}