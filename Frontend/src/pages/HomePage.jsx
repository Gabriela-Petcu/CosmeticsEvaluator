import { Link, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { getHistory } from '../api/evaluate'

const SKIN_LABEL = {
  oily: 'ten gras', dry: 'ten uscat', combination: 'ten mixt',
  sensitive: 'ten sensibil', normal: 'ten normal',
}
const CONCERN_LABEL = {
  acne: 'acnee', dehydration: 'deshidratare', anti_aging: 'anti-aging',
  dark_spots: 'pete', redness: 'roșeață', dullness: 'ten tern',
}

const CATEGORIES = [
  { icon: '💧', name: 'serumuri', filter: 'serum' },
  { icon: '🌸', name: 'hidratare', filter: 'moisturizer' },
  { icon: '☀️', name: 'SPF', filter: 'sunscreen' },
  { icon: '🌿', name: 'curățare', filter: 'cleanser' },
  { icon: '👁️', name: 'ochi', filter: 'eye' },
]

const VERDICT_BADGE = {
  'Recomandat': 'bg-green-100 text-green-800',
  'Nerecomandat': 'bg-red-100 text-red-800',
  default: 'bg-amber-100 text-amber-800',
}

export default function HomePage() {
  const { isAuthenticated, user } = useAuth()
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [recentHistory, setRecentHistory] = useState([])
  const [showAuthModal, setShowAuthModal] = useState(false)

  useEffect(() => {
    if (isAuthenticated) {
      getHistory()
        .then(res => setRecentHistory((res.data || []).slice(0, 3)))
        .catch(() => setRecentHistory([]))
    }
  }, [isAuthenticated])

  const handleSearch = (e) => {
    e.preventDefault()
    if (!isAuthenticated) { setShowAuthModal(true); return }
    if (search.trim()) navigate(`/evaluate?q=${encodeURIComponent(search)}`)
    else navigate('/evaluate')
  }

  const handleCategory = (filter) => {
    if (!isAuthenticated) { setShowAuthModal(true); return }
    navigate(`/evaluate?q=${filter}`)
  }

  return (
    <div>
      {/* MODAL AUTH */}
      {showAuthModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center px-4"
          onClick={() => setShowAuthModal(false)}>
          <div className="bg-white rounded-2xl p-8 max-w-sm w-full shadow-xl text-center flex flex-col gap-5"
            onClick={e => e.stopPropagation()}>
            <div className="text-4xl">✨</div>
            <h2 className="font-serif text-2xl font-light text-ink">
              Bucură-te de SkinIQ<br />
              <em className="italic text-rose-primary">gratuit.</em>
            </h2>
            <p className="text-sm text-muted leading-relaxed">
              Creează-ți un cont gratuit pentru a evalua produse, a primi recomandări personalizate și a-ți salva istoricul.
            </p>
            <div className="flex flex-col gap-2">
              <Link to="/register" className="btn-primary flex items-center justify-center gap-2 py-3">
                creează cont gratuit
              </Link>
              <Link to="/login" className="btn-outline flex items-center justify-center gap-2 py-3">
                am deja cont — autentifică-mă
              </Link>
            </div>
            <button onClick={() => setShowAuthModal(false)}
              className="text-xs text-soft hover:text-muted transition-colors cursor-pointer">
              poate mai târziu
            </button>
          </div>
        </div>
      )}

      {/* HERO */}
      <div className="bg-cream-warm border-b border-rose-border px-16 py-20 flex flex-col items-center text-center gap-8">
        <div className="flex items-center gap-2">
          <div className="w-8 h-px bg-rose-primary" />
          <span className="text-xs tracking-widest text-rose-primary font-medium">EVALUARE INTELIGENTĂ</span>
          <div className="w-8 h-px bg-rose-primary" />
        </div>

        <h1 className="font-serif text-6xl font-light leading-tight text-ink max-w-2xl">
          Descoperă ce merită<br />
          <em className="italic text-rose-primary">pielea ta.</em>
        </h1>

        <p className="text-sm text-muted leading-relaxed max-w-lg">
          Introduci un produs de skincare. Algoritmul analizează recenzii, popularitate
          și compatibilitatea cu profilul tău de ten — fără sponsorizări, fără compromisuri.
        </p>

        <form onSubmit={handleSearch} className="flex border border-rose-mid rounded-lg overflow-hidden w-full max-w-xl shadow-sm">
          <input
            className="flex-1 px-5 py-4 text-sm outline-none bg-white placeholder-soft"
            placeholder="Caută un produs sau brand…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          <button type="submit"
            className="bg-rose-primary text-white px-8 text-xs tracking-widest hover:bg-rose-deeper transition-colors font-medium">
            analizează
          </button>
        </form>

        <div className="flex items-center gap-8">
          <TrustItem icon="🗄️" text="9.000+ produse" />
          <div className="w-px h-4 bg-rose-border" />
          <TrustItem icon="🧠" text="model ML antrenat" />
          <div className="w-px h-4 bg-rose-border" />
          <TrustItem icon="🛡️" text="0 sponsorizări" />
          <div className="w-px h-4 bg-rose-border" />
          <TrustItem icon="✨" text="personalizat pentru tine" />
        </div>
      </div>

      {/* PROFILE STRIP */}
      {isAuthenticated && user && (
        <div className="bg-rose-light border-b border-rose-border px-9 py-3 flex items-center gap-4">
          <span className="text-xs font-medium text-rose-dark tracking-wide">profilul tău activ</span>
          <div className="flex gap-2">
            <ProfileTag text={SKIN_LABEL[user.skinType] || user.skinType} icon="💧" />
            <ProfileTag text={CONCERN_LABEL[user.mainConcern] || user.mainConcern} icon="✨" />
            <ProfileTag text={`buget ${user.budgetLevel}`} icon="💰" />
          </div>
          <Link to="/profile" className="ml-auto text-xs text-rose-primary hover:underline tracking-wide">
            editează profilul →
          </Link>
        </div>
      )}

      {/* CATEGORII */}
      <div className="px-9 py-10 border-b border-rose-border bg-cream">
        <h2 className="font-serif text-2xl font-light text-center text-ink mb-7">
          Explorează după categorie
        </h2>
        <div className="grid grid-cols-5 gap-4 max-w-3xl mx-auto">
          {CATEGORIES.map(cat => (
            <div key={cat.name}
              onClick={() => handleCategory(cat.filter)}
              className="card flex flex-col items-center gap-3 py-5 hover:border-rose-mid transition-colors group cursor-pointer">
              <div className="w-12 h-12 rounded-full bg-rose-light flex items-center justify-center text-2xl group-hover:bg-rose-border transition-colors">
                {cat.icon}
              </div>
              <div className="text-xs text-gray-600 font-medium">{cat.name}</div>
            </div>
          ))}
        </div>
      </div>

      {/* EVALUĂRI RECENTE */}
      {isAuthenticated && recentHistory.length > 0 && (
        <div className="px-9 py-10 bg-cream-warm border-b border-rose-border">
          <div className="flex items-baseline justify-between mb-6">
            <h2 className="font-serif text-2xl font-light text-ink">Evaluările tale recente</h2>
            <Link to="/history" className="text-xs tracking-widest text-rose-primary hover:underline">
              vezi toate →
            </Link>
          </div>
          <div className="grid grid-cols-3 gap-4">
            {recentHistory.map(entry => {
              const prob = entry.mlProbability ? Math.round(entry.mlProbability * 100) : null
              const badgeClass = VERDICT_BADGE[entry.finalVerdict] || VERDICT_BADGE.default
              return (
                <div key={entry.id} className="card flex flex-col gap-3 hover:border-rose-mid transition-colors">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-rose-light flex items-center justify-center text-xl flex-shrink-0">🧴</div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium tracking-widest text-soft">{entry.brand}</div>
                      <div className="text-sm font-medium text-ink truncate">{entry.name || entry.productId}</div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    {prob !== null && (
                      <div className="font-serif text-xl font-light text-rose-primary">{prob}%</div>
                    )}
                    <div className={`text-xs px-2 py-0.5 rounded font-medium ${badgeClass}`}>
                      {entry.finalVerdict}
                    </div>
                  </div>
                  <div className="text-xs text-muted">
                    {new Date(entry.createdAt).toLocaleDateString('ro-RO', { day: 'numeric', month: 'long' })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* CTA neautentificat */}
      {!isAuthenticated && (
        <div className="px-9 py-12 bg-cream-warm border-b border-rose-border flex flex-col items-center gap-5 text-center">
          <h2 className="font-serif text-3xl font-light text-ink">
            Evaluări personalizate,<br />
            <em className="italic text-rose-primary">gratuit.</em>
          </h2>
          <p className="text-sm text-muted max-w-md leading-relaxed">
            Creează-ți contul și primești recomandări adaptate tipului tău de ten, preocupărilor principale și bugetului.
          </p>
          <div className="flex gap-4">
            <Link to="/register" className="btn-primary px-8 py-3">creează cont gratuit</Link>
            <button onClick={() => setShowAuthModal(true)} className="btn-outline px-8 py-3">
              am deja cont
            </button>
          </div>
        </div>
      )}

      {/* BANNER DE CE SKINIQ */}
      <div className="grid grid-cols-2 border-t border-rose-border">
        <div className="bg-rose-border px-10 py-12 flex flex-col gap-5 justify-center">
          <div className="text-xs tracking-widest text-rose-dark font-medium">DE CE SKINIQ</div>
          <h2 className="font-serif text-3xl font-light text-ink leading-tight">
            Nu marketing.<br />
            <em className="italic text-rose-dark">Știință.</em>
          </h2>
          <p className="text-sm text-rose-dark leading-relaxed max-w-xs opacity-80">
            Evaluăm produsele pe baza datelor reale — recenzii, popularitate,
            raport calitate-preț — nu pe baza sponsorizărilor.
          </p>
          {!isAuthenticated ? (
            <Link to="/register" className="btn-primary w-fit">creează cont gratuit →</Link>
          ) : (
            <Link to="/evaluate" className="btn-primary w-fit">evaluează un produs →</Link>
          )}
        </div>
        <div className="bg-rose-mid flex items-center justify-center gap-6 px-10 py-12">
          {[
            { num: '9.2k', label: 'produse evaluate' },
            { num: '91%', label: 'acuratețe model ML' },
            { num: '0', label: 'sponsorizări' },
          ].map(s => (
            <div key={s.label} className="bg-white rounded-2xl px-6 py-5 text-center min-w-[100px]">
              <div className="font-serif text-3xl font-light text-rose-primary">{s.num}</div>
              <div className="text-xs text-soft mt-1">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function TrustItem({ icon, text }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted">
      <span>{icon}</span>
      <span>{text}</span>
    </div>
  )
}

function ProfileTag({ icon, text }) {
  return (
    <div className="flex items-center gap-1.5 text-xs bg-white border border-rose-border rounded-full px-3 py-1 text-rose-dark">
      <span>{icon}</span>
      {text}
    </div>
  )
}