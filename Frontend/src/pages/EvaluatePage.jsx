import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { getProducts, evaluateById, evaluateManual } from '../api/evaluate'

const CATEGORIES = [
  { key: 'category_Anti-Aging', label: 'anti-aging' },
  { key: 'category_Blemish_&_Acne_Treatments', label: 'acnee' },
  { key: 'category_Exfoliators', label: 'exfolianți' },
  { key: 'category_Eye_Creams_&_Treatments', label: 'ochi' },
  { key: 'category_Face_Masks', label: 'măști' },
  { key: 'category_Face_Oils', label: 'uleiuri faciale' },
  { key: 'category_Face_Serums', label: 'serumuri' },
  { key: 'category_Face_Sunscreen', label: 'SPF' },
  { key: 'category_Face_Wash_&_Cleansers', label: 'curățare' },
  { key: 'category_Facial_Peels', label: 'peeling' },
  { key: 'category_Mists_&_Essences', label: 'mist/esențe' },
  { key: 'category_Moisturizer_&_Treatments', label: 'tratamente' },
  { key: 'category_Moisturizers', label: 'hidratare' },
  { key: 'category_Night_Creams', label: 'cremă noapte' },
  { key: 'category_Toners', label: 'toner' },
  { key: 'category_Blotting_Papers', label: 'blotting' },
]

export default function EvaluatePage() {
  const { isAuthenticated, user } = useAuth()
  const navigate = useNavigate()

  const [tab, setTab] = useState('catalog')
  const [products, setProducts] = useState([])
  const [filtered, setFiltered] = useState([])
  const [search, setSearch] = useState('')
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingProducts, setLoadingProducts] = useState(true)
  const [error, setError] = useState('')

  // Manual form
  const [form, setForm] = useState({
    brand: '', name: '', review_score: '', n_of_reviews: '',
    n_of_loves: '', price_per_ounce: '', price: '',
  })
  const [categories, setCategories] = useState({})

  useEffect(() => {
  const params = new URLSearchParams(window.location.search)
  const urlQuery = params.get('q')
  if (urlQuery) {
    setSearch(urlQuery)
    setTab('catalog')
  }

  getProducts()
    .then(res => {
      setProducts(res.data || [])
    })
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

  const handleEvaluateCatalog = async () => {
    if (!selectedProduct) return
    if (!isAuthenticated) { navigate('/login'); return }
    setLoading(true)
    setError('')
    try {
      const res = await evaluateById(selectedProduct.id)
      // Salvăm rezultatul în sessionStorage și navigăm
      sessionStorage.setItem('skiniq_result', JSON.stringify(res.data))
      navigate('/result/latest')
    } catch (err) {
      setError(err.response?.data?.detail || 'Eroare la evaluare. Încearcă din nou.')
    } finally {
      setLoading(false)
    }
  }

  const handleEvaluateManual = async () => {
    if (!isAuthenticated) { navigate('/login'); return }
    setLoading(true)
    setError('')
    try {
      const catObj = {}
      CATEGORIES.forEach(c => { catObj[c.key] = categories[c.key] ? 1 : 0 })
      const payload = {
        ...catObj,
        review_score: parseFloat(form.review_score),
        n_of_reviews: parseInt(form.n_of_reviews),
        n_of_loves: parseInt(form.n_of_loves),
        price_per_ounce: parseFloat(form.price_per_ounce),
        price: parseFloat(form.price) || 0,
        brand: form.brand || 'Unknown',
        name: form.name || 'Unknown',
      }
      const res = await evaluateManual(payload)
      sessionStorage.setItem('skiniq_result', JSON.stringify(res.data))
      navigate('/result/latest')
    } catch (err) {
      setError(err.response?.data?.detail || 'Eroare la evaluare. Verifică datele introduse.')
    } finally {
      setLoading(false)
    }
  }

  const skinLabel = {
    oily: 'ten gras', dry: 'ten uscat', combination: 'ten mixt',
    sensitive: 'ten sensibil', normal: 'ten normal',
  }
  const concernLabel = {
    acne: 'acnee', dehydration: 'deshidratare', anti_aging: 'anti-aging',
    dark_spots: 'pete', redness: 'roșeață', dullness: 'ten tern',
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
      {/* HERO */}
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
              Selectează din catalog sau introdu manual datele unui produs. Algoritmul îl evaluează instant în funcție de profilul tău.
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

      {/* BODY */}
      <div className="grid grid-cols-2 min-h-[calc(100vh-200px)]">

        {/* STÂNGA */}
        <div className="border-r border-rose-border px-9 py-7 flex flex-col gap-5">
          {/* Tabs */}
          <div className="flex border-b border-rose-border">
            <button onClick={() => setTab('catalog')}
              className={`text-xs tracking-wide px-4 py-2.5 border-b-2 transition-colors cursor-pointer
                ${tab === 'catalog' ? 'border-rose-primary text-rose-primary font-medium' : 'border-transparent text-muted hover:text-ink'}`}>
              din catalog
            </button>
          </div>

          {/* CATALOG */}
          {tab === 'catalog' && (
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
          )}

          {/* MANUAL */}
          {tab === 'manual' && (
            <div className="flex flex-col gap-4 overflow-y-auto max-h-[480px] pr-1">
              <div className="grid grid-cols-2 gap-3">
                <Field label="BRAND" required>
                  <input className="input-field" placeholder="ex. The Ordinary"
                    value={form.brand} onChange={e => setForm(f => ({ ...f, brand: e.target.value }))} />
                </Field>
                <Field label="NUME PRODUS" required>
                  <input className="input-field" placeholder="ex. Hyaluronic Acid"
                    value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
                </Field>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Field label="RATING" required hint="0–5">
                  <input className="input-field" type="number" placeholder="ex. 4.5" min="0" max="5" step="0.1"
                    value={form.review_score} onChange={e => setForm(f => ({ ...f, review_score: e.target.value }))} />
                </Field>
                <Field label="NR. RECENZII" required>
                  <input className="input-field" type="number" placeholder="ex. 3241"
                    value={form.n_of_reviews} onChange={e => setForm(f => ({ ...f, n_of_reviews: e.target.value }))} />
                </Field>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Field label="NR. LOVES" required>
                  <input className="input-field" type="number" placeholder="ex. 12500"
                    value={form.n_of_loves} onChange={e => setForm(f => ({ ...f, n_of_loves: e.target.value }))} />
                </Field>
                <Field label="PREȚ/OZ" required hint="preț ÷ cantitate în oz">
                  <input className="input-field" type="number" placeholder="ex. 85.48" step="0.01"
                    value={form.price_per_ounce} onChange={e => setForm(f => ({ ...f, price_per_ounce: e.target.value }))} />
                </Field>
              </div>
              <Field label="PREȚ TOTAL" hint="opțional">
                <input className="input-field" type="number" placeholder="ex. 72" step="0.01"
                  value={form.price} onChange={e => setForm(f => ({ ...f, price: e.target.value }))} />
              </Field>
              <div>
                <div className="text-xs font-medium tracking-widest text-gray-500 mb-2">CATEGORIA PRODUSULUI</div>
                <div className="grid grid-cols-3 gap-1.5">
                  {CATEGORIES.map(c => (
                    <label key={c.key}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-xs cursor-pointer transition-all
                        ${categories[c.key] ? 'border-rose-primary bg-rose-light text-rose-dark' : 'border-gray-200 bg-white text-muted hover:border-rose-mid'}`}>
                      <input type="checkbox" className="accent-rose-primary"
                        checked={!!categories[c.key]}
                        onChange={e => setCategories(cat => ({ ...cat, [c.key]: e.target.checked }))} />
                      {c.label}
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* DREAPTA — Preview + Submit */}
        <div className="px-9 py-7 flex flex-col gap-6">

          {/* Preview produs selectat */}
          <div className="card flex flex-col gap-4">
            <div className="text-xs font-medium tracking-widest text-soft">PRODUS SELECTAT</div>
            {tab === 'catalog' && selectedProduct ? (
              <>
                <div className="flex items-center gap-4">
                  <div className="w-14 h-16 rounded-xl bg-rose-light flex items-center justify-center text-2xl flex-shrink-0">🧴</div>
                  <div>
                    <div className="text-xs font-medium tracking-widest text-soft">{selectedProduct.brand}</div>
                    <div className="text-base font-medium text-ink mt-1">{selectedProduct.name}</div>
                  </div>
                </div>
              </>
            ) : tab === 'manual' && form.name ? (
              <div className="flex items-center gap-4">
                <div className="w-14 h-16 rounded-xl bg-rose-light flex items-center justify-center text-2xl flex-shrink-0">🧴</div>
                <div>
                  <div className="text-xs font-medium tracking-widest text-soft">{form.brand || 'Brand necunoscut'}</div>
                  <div className="text-base font-medium text-ink mt-1">{form.name}</div>
                  {form.review_score && <div className="text-xs text-muted mt-1">⭐ {form.review_score} · {form.n_of_reviews} recenzii</div>}
                </div>
              </div>
            ) : (
              <div className="text-sm text-soft py-4 text-center">
                {tab === 'catalog' ? 'Selectează un produs din listă' : 'Completează formularul din stânga'}
              </div>
            )}
          </div>

          {/* Info profil */}
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
            onClick={tab === 'catalog' ? handleEvaluateCatalog : handleEvaluateManual}
            disabled={loading || (tab === 'catalog' && !selectedProduct) || (tab === 'manual' && !form.review_score)}
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

function Field({ label, children, required, hint }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-medium tracking-widest text-gray-500">
        {label} {required && <span className="text-rose-primary">*</span>}
      </label>
      {children}
      {hint && <div className="text-xs text-soft">{hint}</div>}
    </div>
  )
}