import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const VERDICT_CONFIG = {
  'Recomandat': {
    bg: 'bg-green-50', border: 'border-green-200',
    badge: 'bg-green-100 text-green-800 border-green-300',
    icon: '✅', color: 'text-green-700',
  },
  'Nerecomandat': {
    bg: 'bg-red-50', border: 'border-red-200',
    badge: 'bg-red-100 text-red-800 border-red-300',
    icon: '❌', color: 'text-red-700',
  },
  default: {
    bg: 'bg-amber-50', border: 'border-amber-200',
    badge: 'bg-amber-100 text-amber-800 border-amber-300',
    icon: '⚠️', color: 'text-amber-700',
  },
}

function getVerdictConfig(verdict) {
  if (!verdict) return VERDICT_CONFIG.default
  if (verdict === 'Recomandat') return VERDICT_CONFIG['Recomandat']
  if (verdict === 'Nerecomandat') return VERDICT_CONFIG['Nerecomandat']
  return VERDICT_CONFIG.default
}

// Helper — citește câmpul indiferent de majuscule
function g(obj, ...keys) {
  for (const key of keys) {
    if (obj[key] !== undefined && obj[key] !== null) return obj[key]
  }
  return null
}

export default function ResultPage() {
  const { isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const [result, setResult] = useState(null)

  useEffect(() => {
    const saved = sessionStorage.getItem('skiniq_result')
    if (!saved) { navigate('/evaluate'); return }
    try {
      const parsed = JSON.parse(saved)
      // Normalizăm — extragem originalResult dacă există
      const raw = parsed.originalResult || parsed.OriginalResult || parsed
      const productInfo = parsed.productInfo || parsed.ProductInfo || {}

      const normalized = {
        ScorFinal: g(raw, 'scorFinal', 'ScorFinal') ?? 0,
        Merita: g(raw, 'merita', 'Merita') ?? 0,
        MeritaML: g(raw, 'meritaML', 'MeritaML') ?? 0,
        ProbabilitateML: g(raw, 'probabilitateML', 'ProbabilitateML') ?? 0,
        FitScore: g(raw, 'fitScore', 'FitScore') ?? 0,
        SePotriveste: g(raw, 'sePotriveste', 'SePotriveste') ?? 0,
        VerdictFinal: g(raw, 'verdictFinal', 'VerdictFinal') || '',
        ExplicatieFinala: g(raw, 'explicatieFinala', 'ExplicatieFinala') || '',
        MotivePozitive: g(raw, 'motivePozitive', 'MotivePozitive') || [],
        MotiveNegative: g(raw, 'motiveNegative', 'MotiveNegative') || [],
        TopFactoriML: g(raw, 'topFactoriML', 'TopFactoriML') || [],
        productName: g(productInfo, 'name', 'Name') || g(raw, 'productId') || 'Produs evaluat',
        brand: g(productInfo, 'brand', 'Brand') || '',
        price: g(productInfo, 'price', 'Price') || null,
      }
      setResult(normalized)
    } catch {
      navigate('/evaluate')
    }
  }, [])

  if (!result) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="font-serif text-2xl text-rose-primary animate-pulse">se încarcă rezultatul...</div>
      </div>
    )
  }

  const vc = getVerdictConfig(result.VerdictFinal)
  const scorePercent = Math.min(100, Math.max(0, result.ScorFinal))
  const mlPercent = Math.round(result.ProbabilitateML * 100)
  const fitPercent = Math.min(100, Math.max(0, result.FitScore))

  return (
    <div>
      {/* BREADCRUMB */}
      <div className="px-9 py-3 border-b border-rose-border bg-cream text-xs text-soft flex items-center gap-2">
        <Link to="/evaluate" className="hover:text-rose-primary transition-colors">evaluează</Link>
        <span>›</span>
        <span className="text-ink">rezultat evaluare</span>
      </div>

      {/* HERO */}
      <div className={`px-9 py-7 border-b ${vc.border} ${vc.bg} flex items-center gap-6`}>
        <div className="w-16 h-20 rounded-xl bg-white border border-rose-border flex items-center justify-center text-3xl flex-shrink-0">
          🧴
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium tracking-widest text-soft mb-1">PRODUS EVALUAT</div>
          {result.brand && (
            <div className="text-xs font-medium tracking-widest text-soft mb-1">{result.brand.toUpperCase()}</div>
          )}
          <div className="font-serif text-2xl font-light text-ink leading-tight">
            {result.productName}
          </div>
          <div className="text-xs text-muted mt-2 flex items-center gap-3">
            {result.price && <span>${result.price}</span>}
          </div>
        </div>
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <div className={`flex items-center gap-2 px-5 py-2.5 rounded-full border font-medium text-sm ${vc.badge}`}>
            <span>{vc.icon}</span>
            {result.VerdictFinal}
          </div>
          <div className={`text-xs max-w-[220px] text-right leading-relaxed ${vc.color}`}>
            {result.ExplicatieFinala}
          </div>
        </div>
      </div>

      {/* BODY */}
      <div className="grid grid-cols-2">

        {/* STÂNGA */}
        <div className="border-r border-rose-border px-9 py-7 flex flex-col gap-8">

          {/* Scoruri */}
          <div>
            <h2 className="section-title mb-5">Scorurile evaluării</h2>
            <div className="grid grid-cols-2 gap-3">
              <div className="col-span-2 bg-rose-light border border-rose-border rounded-xl p-5">
                <div className="text-xs font-medium tracking-widest text-soft mb-2">SCOR FINAL BASELINE</div>
                <div className="font-serif text-5xl font-light text-rose-primary leading-none">
                  {result.ScorFinal.toFixed(1)}
                </div>
                <div className="mt-3 h-1.5 bg-rose-border rounded-full overflow-hidden">
                  <div className="h-full bg-rose-primary rounded-full transition-all"
                    style={{ width: `${scorePercent}%` }} />
                </div>
                <div className="text-xs text-rose-deeper mt-2">din 100 · prag recomandare = percentila 75</div>
              </div>

              <div className="card">
                <div className="text-xs font-medium tracking-widest text-soft mb-2">PROBABILITATE ML</div>
                <div className={`font-serif text-3xl font-light ${mlPercent >= 75 ? 'text-green-600' : mlPercent >= 50 ? 'text-amber-600' : 'text-red-500'}`}>
                  {mlPercent}%
                </div>
                <div className="mt-2 h-1 bg-gray-100 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${mlPercent >= 75 ? 'bg-green-500' : mlPercent >= 50 ? 'bg-amber-400' : 'bg-red-400'}`}
                    style={{ width: `${mlPercent}%` }} />
                </div>
                <div className="text-xs text-muted mt-1.5">model logistic regression</div>
              </div>

              <div className="card">
                <div className="text-xs font-medium tracking-widest text-soft mb-2">COMPATIBILITATE PROFIL</div>
                <div className={`font-serif text-3xl font-light ${fitPercent >= 60 ? 'text-green-600' : 'text-amber-600'}`}>
                  {fitPercent}
                </div>
                <div className="mt-2 h-1 bg-gray-100 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${fitPercent >= 60 ? 'bg-green-500' : 'bg-amber-400'}`}
                    style={{ width: `${fitPercent}%` }} />
                </div>
                <div className="text-xs text-muted mt-1.5">FitScore · prag = 60</div>
              </div>
            </div>
          </div>

          {/* Componentele verdictului */}
          <div>
            <h2 className="section-title mb-4">Componentele verdictului</h2>
            <div className="flex flex-col divide-y divide-rose-border">
              <ComponentRow
                icon="📊" iconBg="bg-green-50"
                title="Scor baseline"
                desc="rating, recenzii, loves, preț/oz"
                value={result.Merita === 1 ? 'merită' : 'nu merită'}
                positive={result.Merita === 1}
              />
              <ComponentRow
                icon="🧠" iconBg="bg-blue-50"
                title="Model ML"
                desc="logistic regression · 9.000+ produse"
                value={result.MeritaML === 1 ? 'merită' : 'nu merită'}
                positive={result.MeritaML === 1}
              />
              <ComponentRow
                icon="👤" iconBg="bg-rose-light"
                title="Compatibilitate profil"
                desc="tip ten · preocupare · buget"
                value={`${fitPercent} / 100`}
                positive={fitPercent >= 60}
                neutral
              />
            </div>
          </div>

          {/* SHAP */}
          {result.TopFactoriML && result.TopFactoriML.length > 0 && (
            <div>
              <h2 className="section-title mb-4">Top factori ML (SHAP)</h2>
              <div className="flex flex-col gap-2">
                {result.TopFactoriML.map((factor, i) => {
                  const shapVal = g(factor, 'shap_value', 'shapValue') || 0
                  const featVal = g(factor, 'feature_value', 'featureValue')
                  const featName = g(factor, 'feature') || ''
                  const direction = g(factor, 'direction') || ''
                  const isPositive = direction === 'creste_probabilitatea'
                  const barWidth = Math.min(100, Math.abs(shapVal) * 300)
                  return (
                    <div key={i} className="flex items-center gap-3 p-3 border border-rose-border rounded-xl bg-white">
                      <div className="font-serif text-lg font-light text-rose-primary w-5 text-center flex-shrink-0">
                        {i + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium font-mono text-ink">{featName}</div>
                        <div className="text-xs text-muted mt-0.5">
                          valoare: {featVal !== null && featVal !== undefined ? Number(featVal).toFixed(2) : '—'}
                          · SHAP: {shapVal > 0 ? '+' : ''}{shapVal.toFixed(4)}
                        </div>
                        <div className="mt-1.5 h-1 bg-gray-100 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${isPositive ? 'bg-green-400' : 'bg-red-400'}`}
                            style={{ width: `${barWidth}%` }} />
                        </div>
                      </div>
                      <div className={`text-xs px-2 py-1 rounded font-medium flex-shrink-0
                        ${isPositive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {isPositive ? '↑ crește prob.' : '↓ scade prob.'}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* DREAPTA */}
        <div className="px-9 py-7 flex flex-col gap-8">

          {/* Semnale */}
          <div>
            <h2 className="section-title mb-4">Semnale de compatibilitate</h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-2">
                <div className="text-xs font-medium tracking-widest text-green-700 mb-1">SEMNALE POZITIVE</div>
                {result.MotivePozitive && result.MotivePozitive.length > 0 ? (
                  result.MotivePozitive.map((m, i) => (
                    <div key={i} className="flex items-start gap-2 bg-green-50 rounded-lg p-2.5 text-xs text-green-800 leading-relaxed">
                      <span className="flex-shrink-0 mt-0.5">✓</span>{m}
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-muted">Niciun semnal pozitiv.</div>
                )}
              </div>
              <div className="flex flex-col gap-2">
                <div className="text-xs font-medium tracking-widest text-red-600 mb-1">SEMNALE NEGATIVE</div>
                {result.MotiveNegative && result.MotiveNegative.length > 0 ? (
                  result.MotiveNegative.map((m, i) => (
                    <div key={i} className="flex items-start gap-2 bg-red-50 rounded-lg p-2.5 text-xs text-red-800 leading-relaxed">
                      <span className="flex-shrink-0 mt-0.5">✗</span>{m}
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-muted">Niciun semnal negativ.</div>
                )}
              </div>
            </div>
          </div>

          {/* Acțiuni */}
          <div>
            <h2 className="section-title mb-4">Acțiuni</h2>
            <div className="grid grid-cols-3 gap-3">
              <button className="btn-primary flex items-center justify-center gap-2 py-3 col-span-3">
                🛒 unde cumpăr
              </button>
              <button className="btn-outline flex items-center justify-center gap-2 py-2.5">
                🔖 salvează
              </button>
              <button className="btn-outline flex items-center justify-center gap-2 py-2.5">
                📤 distribuie
              </button>
              <Link to="/evaluate"
                className="btn-outline flex items-center justify-center gap-2 py-2.5 text-xs text-center">
                ← evaluează alt produs
              </Link>
            </div>
          </div>

          {/* Despre scor */}
          <div className="card bg-cream-warm">
            <div className="text-xs font-medium tracking-widest text-soft mb-3">DESPRE ACEST SCOR</div>
            <div className="flex flex-col gap-2 text-xs text-muted leading-relaxed">
              <p><strong className="text-ink">Scorul baseline</strong> combină rating-ul, numărul de recenzii, loves și prețul/oz cu ponderi fixe: 50% rating, 20% recenzii, 20% loves, 10% preț.</p>
              <p><strong className="text-ink">Modelul ML</strong> este un clasificator de regresie logistică antrenat pe 9.000+ produse Sephora.</p>
              <p><strong className="text-ink">Compatibilitatea</strong> este calculată euristic pe baza tipului tău de ten, preocupării principale și bugetului.</p>
            </div>
          </div>

          <Link to="/history" className="flex items-center justify-center gap-2 text-xs text-rose-primary hover:underline">
            📋 vezi toate evaluările tale →
          </Link>
        </div>
      </div>
    </div>
  )
}

function ComponentRow({ icon, iconBg, title, desc, value, positive, neutral }) {
  return (
    <div className="flex items-center justify-between py-3">
      <div className="flex items-center gap-3">
        <div className={`w-8 h-8 rounded-lg ${iconBg} flex items-center justify-center flex-shrink-0 text-sm`}>
          {icon}
        </div>
        <div>
          <div className="text-sm font-medium text-ink">{title}</div>
          <div className="text-xs text-muted mt-0.5">{desc}</div>
        </div>
      </div>
      <div className={`text-xs font-medium px-3 py-1 rounded-full
        ${neutral ? 'bg-rose-light text-rose-dark'
          : positive ? 'bg-green-100 text-green-700'
          : 'bg-red-100 text-red-700'}`}>
        {value}
      </div>
    </div>
  )
}