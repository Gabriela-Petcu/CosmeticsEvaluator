import { Link, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { useAuth } from '../context/AuthContext'

const CATEGORIES = [
  { icon: '💧', name: 'serumuri', count: 214 },
  { icon: '🌸', name: 'hidratare', count: 318 },
  { icon: '☀️', name: 'SPF', count: 97 },
  { icon: '🌿', name: 'curățare', count: 186 },
  { icon: '👁️', name: 'ochi', count: 74 },
]

const FEATURED_PRODUCTS = [
  { brand: 'LA MER', name: 'Moisturizing Soft Cream', price: '$95', verdict: 'Recomandat', score: 88.2, icon: '🌊' },
  { brand: 'THE ORDINARY', name: 'Hyaluronic Acid 2% + B5', price: '$9', verdict: 'Recomandat', score: 79.4, icon: '🔬' },
  { brand: 'SUPERGOOP', name: 'Unseen Sunscreen SPF 40', price: '$36', verdict: 'Verifică', score: 71.0, icon: '☀️' },
  { brand: 'SULWHASOO', name: 'First Care Activating Serum', price: '$82', verdict: 'Recomandat', score: 84.7, icon: '✨' },
]

const VERDICT_BADGE = {
  'Recomandat': 'bg-green-100 text-green-800',
  'Nerecomandat': 'bg-red-100 text-red-800',
  'Verifică': 'bg-amber-100 text-amber-800',
}

const SKIN_LABEL = {
  oily: 'ten gras', dry: 'ten uscat', combination: 'ten mixt',
  sensitive: 'ten sensibil', normal: 'ten normal',
}

const CONCERN_LABEL = {
  acne: 'acnee', dehydration: 'deshidratare', anti_aging: 'anti-aging',
  dark_spots: 'pete', redness: 'roșeață', dullness: 'ten tern',
}

export default function HomePage() {
  const { isAuthenticated, user } = useAuth()
  const navigate = useNavigate()
  const [search, setSearch] = useState('')

  const handleSearch = (e) => {
    e.preventDefault()
    if (search.trim()) navigate(`/evaluate?q=${encodeURIComponent(search)}`)
    else navigate('/evaluate')
  }

  return (
    <div>

      {/* HERO */}
      <div className="grid grid-cols-2 border-b border-rose-border">
        <div className="bg-cream-warm px-10 py-14 flex flex-col justify-center gap-6 border-r border-rose-border">
          <div className="flex items-center gap-2">
            <div className="w-5 h-px bg-rose-primary" />
            <span className="text-xs tracking-widest text-rose-primary font-medium">EVALUARE INTELIGENTĂ</span>
          </div>

          <h1 className="font-serif text-5xl font-light leading-tight text-ink">
            Descoperă ce<br />
            merită<br />
            <em className="italic text-rose-primary">pielea ta.</em>
          </h1>

          <p className="text-sm text-muted leading-relaxed max-w-sm">
            Introduci un produs de skincare. Algoritmul analizează recenzii, popularitate
            și compatibilitatea cu profilul tău de ten.
          </p>

          <form onSubmit={handleSearch} className="flex border border-rose-mid rounded overflow-hidden max-w-sm">
            <input
              className="flex-1 px-4 py-3 text-sm outline-none bg-white placeholder-soft italic"
              placeholder="Caută un produs sau brand…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <button type="submit" className="bg-rose-primary text-white px-5 text-xs tracking-widest hover:bg-rose-deeper transition-colors">
              analizează
            </button>
          </form>

          <div className="flex gap-5">
            <TrustItem icon="🗄️" text="9.000+ produse" />
            <TrustItem icon="🧠" text="model ML antrenat" />
            <TrustItem icon="✅" text="personalizat pentru tine" />
          </div>
        </div>

        {/* Dreapta hero */}
        <div className="bg-rose-light flex items-end justify-center px-10 pt-10 relative overflow-hidden">
          {/* Placeholder foto model */}
          <div className="w-52 h-72 bg-rose-border rounded-t-lg flex flex-col items-center justify-center gap-2 opacity-60">
            <span className="text-5xl">🧖‍♀️</span>
            <span className="text-xs text-rose-dark tracking-widest">foto model</span>
          </div>

          {/* Card floating scor */}
          <div className="absolute top-8 right-8 bg-white border border-rose-border rounded-2xl p-4 w-40 shadow-sm">
            <div className="text-xs tracking-widest text-soft mb-1">scor final</div>
            <div className="font-serif text-4xl font-light text-rose-primary leading-none">83.4</div>
            <div className="text-xs text-green-700 font-medium mt-2">✓ recomandat</div>
            <div className="text-xs text-soft mt-2 border-t border-rose-border pt-2 leading-tight">
              Drunk Elephant<br />Marula Oil
            </div>
          </div>
        </div>
      </div>

      {/* PROFILE STRIP — doar dacă e autentificat */}
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
        <div className="grid grid-cols-5 gap-4">
          {CATEGORIES.map(cat => (
            <Link to="/evaluate" key={cat.name}
              className="card flex flex-col items-center gap-3 py-5 hover:border-rose-mid transition-colors group">
              <div className="w-12 h-12 rounded-full bg-rose-light flex items-center justify-center text-2xl group-hover:bg-rose-border transition-colors">
                {cat.icon}
              </div>
              <div className="text-xs text-gray-600 font-medium">{cat.name}</div>
              <div className="text-xs text-soft">{cat.count} produse</div>
            </Link>
          ))}
        </div>
      </div>

      {/* PRODUSE RECOMANDATE */}
      <div className="px-9 py-10 bg-cream-warm border-b border-rose-border">
        <div className="flex items-baseline justify-between mb-7">
          <h2 className="font-serif text-2xl font-light text-ink">
            {isAuthenticated ? 'Recomandate pentru tine' : 'Produse populare'}
          </h2>
          <Link to="/evaluate" className="text-xs tracking-widest text-rose-primary hover:underline">
            vezi toate →
          </Link>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {FEATURED_PRODUCTS.map(p => (
            <Link to="/evaluate" key={p.name}
              className="card flex flex-col gap-3 hover:border-rose-mid transition-colors group">
              <div className="h-16 bg-rose-light rounded-lg flex items-center justify-center text-3xl group-hover:bg-rose-border transition-colors">
                {p.icon}
              </div>
              <div className="text-xs font-medium tracking-widest text-soft">{p.brand}</div>
              <div className="text-sm font-medium text-ink leading-tight">{p.name}</div>
              <div className="flex items-center justify-between mt-auto">
                <span className="text-xs text-muted">{p.price}</span>
                <span className={`text-xs px-2 py-0.5 rounded font-medium ${VERDICT_BADGE[p.verdict]}`}>
                  {p.verdict}
                </span>
              </div>
              <div className="text-xs text-rose-primary font-medium">scor {p.score}</div>
            </Link>
          ))}
        </div>
      </div>

      {/* BANNER — de ce SkinIQ */}
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
            <Link to="/register" className="btn-primary w-fit flex items-center gap-2">
              creează cont gratuit →
            </Link>
          ) : (
            <Link to="/evaluate" className="btn-primary w-fit flex items-center gap-2">
              evaluează un produs →
            </Link>
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
    <div className="flex items-center gap-2 text-xs text-soft">
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