import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { getProducts, evaluateById } from '../api/evaluate'

export default function EvaluatePage() {
  const { isAuthenticated, user } = useAuth()
  const navigate = useNavigate()

  const [products, setProducts] = useState([])
  const [filtered, setFiltered] = useState([])
  const [search, setSearch] = useState('')
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingProducts, setLoadingProducts] = useState(true)
  const [error, setError] = useState('')

  const skinLabel = {
    oily: 'ten gras', dry: 'ten uscat', combination: 'ten mixt',
    sensitive: 'ten sensibil', normal: 'ten normal',
  }
  const concernLabel = {
    acne: 'acnee', dehydration: 'deshidratare', anti_aging: 'anti-aging',
    dark_spots: 'pete', redness: 'roșeață', dullness: 'ten tern',
  }

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const urlQuery = params.get('q')
    if (urlQuery) setSearch(urlQuery)

    getProducts()
      .then(res => setProducts(res.data || []))
      .catch(() => setProducts([]))
      .finally(() => setLoadingProducts(false))
  }, [])

  useEffect(() => {
    if (!search.trim()) {
      setFiltered(products)
    } else {
      const q = search.toLowerCase()
      setFiltered(products.filter(p =>
        p.name?.toLowerCase().includes(q) ||
        p.brand?.toLowerCase().includes(q)
      ))
    }
  }, [search, products])

  const handleEvaluate = async () => {
    if (!selectedProduct) return
    if (!isAuthenticated) { navigate('/login'); return }
    setLoading(true)
    setError('')
    try {
      const res = await evaluateById(selectedProduct.id)
      sessionStorage.setItem('skiniq_result', JSON.stringify(res.data))
      navigate('/result/latest')
    } catch (err) {
      setError(err.response?.data?.detail || 'Eroare la evaluare. Încearcă din nou.')
    } finally {
      setLoading(false)
    }
  }

  const cleanName = (name) => {
    if (!name) return ''
    return name
      .replace(/,Ñ¢/g, '®')
      .replace(/Ñ¢/g, '®')
      .replace(/√®/g, 'è')
      .replace(/√©/g, 'é')
      .replace(/√à/g, 'à')
      .replace(/¬Æ/g, '®')
      .replace(/â€™/g, "'")
      .replace(/â€"/g, '—')
      .replace(/Ã©/g, 'é')
      .replace(/Ã¨/g, 'è')
      .replace(/[^\x00-\x7F\u00C0-\u024F]/g, '')
      .trim()
  }

  return (
    <div>
      <div className="bg-cream-warm border-b border-rose-border px-9 py-8">
        <div className="flex items-start justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-5 h-px bg-rose-primary" />
              <span className="text-xs tracking-widest text-rose-primary font-medium">EVALUARE PRODUS</span>
            </div>
            <h1 className="font-serif text-3xl font-light text-ink leading-tight">
              Analizează orice produs<br />
              de <em className="italic text-rose-primary">skincare.</em>
            </h1>
            <p className="text-sm text-muted mt-3 max-w-md leading-relaxed">
              Selectează un produs din catalog. Algoritmul îl evaluează instant în funcție de profilul tău.
            </p>
          </div>

          {isAuthenticated && user && (
            <div className="flex items-center gap-2 bg-rose-light border border-rose-border rounded-lg px-4 py-2.5 flex-shrink-0">
              <span className="text-xs text-rose-dark font-medium">profil activ</span>
              <span className="text-xs bg-white border border-rose-border rounded-full px-3 py-1 text-rose-dark">
                {skinLabel[user.skinType] || user.skinType}
              </span>
              <span className="text-xs bg-white border border-rose-border rounded-full px-3 py-1 text-rose-dark">
                {concernLabel[user.mainConcern] || user.mainConcern}
              </span>
              <span className="text-xs bg-white border border-rose-border rounded-full px-3 py-1 text-rose-dark">
                buget {user.budgetLevel}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 min-h-[calc(100vh-200px)]">

        {/* STÂNGA — Lista produse */}
        <div className="border-r border-rose-border px-9 py-7 flex flex-col gap-5">
          <div className="flex border-b border-rose-border">
            <div className="text-xs tracking-wide px-4 py-2.5 border-b-2 border-rose-primary text-rose-primary font-medium">
              din catalog
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <input
              className="input-field"
              placeholder="Caută brand sau produs…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            {loadingProducts ? (
              <div className="text-sm text-muted animate-pulse text-center py-8">se încarcă produsele...</div>
            ) : (
              <div className="flex flex-col gap-2 max-h-[420px] overflow-y-auto pr-1">
                {filtered.slice(0, 50).map(product => (
                  <div key={product.id}
                    onClick={() => setSelectedProduct(product)}
                    className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all
                      ${selectedProduct?.id === product.id
                        ? 'border-rose-primary bg-rose-light'
                        : 'border-rose-border bg-white hover:border-rose-mid'}`}>
                    <div className="w-9 h-9 rounded-lg bg-rose-light flex items-center justify-center flex-shrink-0 text-base">
                      🧴
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium tracking-widest text-soft">{product.brand}</div>
                      <div className="text-sm font-medium text-ink truncate">{cleanName(product.name)}</div>
                    </div>
                    {selectedProduct?.id === product.id && (
                      <div className="w-5 h-5 rounded-full bg-rose-primary flex items-center justify-center flex-shrink-0">
                        <span className="text-white text-xs">✓</span>
                      </div>
                    )}
                  </div>
                ))}
                {filtered.length === 0 && (
                  <div className="text-center py-8 text-muted text-sm">
                    Niciun produs găsit pentru "{search}"
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="px-9 py-7 flex flex-col gap-6">

          <div className="card flex flex-col gap-4">
            <div className="text-xs font-medium tracking-widest text-soft">PRODUS SELECTAT</div>
            {selectedProduct ? (
              <div className="flex items-center gap-4">
                <div className="w-14 h-16 rounded-xl bg-rose-light flex items-center justify-center text-2xl flex-shrink-0">🧴</div>
                <div>
                  <div className="text-xs font-medium tracking-widest text-soft">{selectedProduct.brand}</div>
                  <div className="text-base font-medium text-ink mt-1">{selectedProduct.name}</div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-soft py-4 text-center">
                Selectează un produs din listă
              </div>
            )}
          </div>

          {isAuthenticated && user && (
            <div className="bg-rose-light border border-rose-border rounded-xl p-4 flex gap-3">
              <span className="text-rose-primary text-lg flex-shrink-0">ℹ️</span>
              <div className="text-xs text-rose-dark leading-relaxed">
                <strong className="font-medium block mb-1">Evaluarea e personalizată pentru tine</strong>
                Algoritmul va ține cont de profilul tău — {skinLabel[user.skinType] || user.skinType}, {concernLabel[user.mainConcern] || user.mainConcern}, buget {user.budgetLevel}.
              </div>
            </div>
          )}

          {!isAuthenticated && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex gap-3">
              <span className="text-lg flex-shrink-0">⚠️</span>
              <div className="text-xs text-amber-800 leading-relaxed">
                <strong className="font-medium block mb-1">Autentifică-te pentru evaluare personalizată</strong>
                Fără cont, evaluarea nu poate ține cont de profilul tău de ten.
                <a href="/login" className="text-rose-primary hover:underline ml-1">Intră în cont →</a>
              </div>
            </div>
          )}

          {error && (
            <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
              {error}
            </div>
          )}

          <button
            onClick={handleEvaluate}
            disabled={loading || !selectedProduct}
            className="btn-primary flex items-center justify-center gap-2 py-4 disabled:opacity-50 disabled:cursor-not-allowed">
            {loading ? (
              <><span className="animate-spin">⚙️</span> se analizează...</>
            ) : (
              <>🧠 analizează produsul</>
            )}
          </button>
          <div className="text-xs text-soft text-center">evaluarea durează câteva secunde</div>
        </div>
      </div>
    </div>
  )
}