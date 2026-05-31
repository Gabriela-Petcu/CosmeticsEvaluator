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
      const raw = parsed.originalResult || parsed.OriginalResult || parsed
      const productInfo = parsed.productInfo || parsed.ProductInfo || {}

      const normalized = {
        FinalScore:      g(raw, 'finalScore',      'FinalScore')      ?? 0,
        IsRecommended:   g(raw, 'isRecommended',   'IsRecommended')   ?? 0,
        IsRecommendedML: g(raw, 'isRecommendedML', 'IsRecommendedML') ?? 0,
        MLProbability:   g(raw, 'mlProbability',   'MLProbability')   ?? 0,
        FitScore:        g(raw, 'fitScore',         'FitScore')        ?? 0,
        IsCompatible:    g(raw, 'isCompatible',    'IsCompatible')    ?? 0,
        FinalVerdict:    g(raw, 'finalVerdict',    'FinalVerdict')    || '',
        FinalExplanation:g(raw, 'finalExplanation','FinalExplanation')|| '',
        PositiveSignals: g(raw, 'positiveSignals', 'PositiveSignals') || [],
        NegativeSignals: g(raw, 'negativeSignals', 'NegativeSignals') || [],
        TopMLFactors:    g(raw, 'topMLFactors',    'TopMLFactors')    || [],
        productName: g(raw, 'productName') || g(productInfo, 'name', 'Name') || g(raw, 'productId') || 'Produs evaluat',
        brand: g(raw, 'brand') || g(productInfo, 'brand', 'Brand') || '',
        price: g(raw, 'price') || g(productInfo, 'price', 'Price') || null,
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

  const vc = getVerdictConfig(result.FinalVerdict)
  const scorePercent = Math.min(100, Math.max(0, result.FinalScore))
  const mlPercent    = Math.round(result.MLProbability * 100)
  const fitPercent   = Math.min(100, Math.max(0, result.FitScore))

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
            {result.FinalVerdict}
          </div>
          <div className={`text-xs max-w-[220px] text-right leading-relaxed ${vc.color}`}>
            {result.FinalExplanation}
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
                  {result.FinalScore.toFixed(1)}
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
                value={result.IsRecommended === 1 ? 'merită' : 'nu merită'}
                positive={result.IsRecommended === 1}
              />
              <ComponentRow
                icon="🧠" iconBg="bg-blue-50"
                title="Model ML"
                desc="logistic regression · 9.000+ produse"
                value={result.IsRecommendedML === 1 ? 'merită' : 'nu merită'}
                positive={result.IsRecommendedML === 1}
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
          {result.TopMLFactors && result.TopMLFactors.length > 0 && (
            <div>
              <h2 className="section-title mb-4">Top factori ML (SHAP)</h2>
              <div className="flex flex-col gap-2">
                {result.TopMLFactors.map((factor, i) => {
                  const shapVal  = g(factor, 'shap_value', 'shapValue') || 0
                  const featVal  = g(factor, 'feature_value', 'featureValue')
                  const featName = g(factor, 'feature') || ''
                  const direction = g(factor, 'direction') || ''
                  const isPositive = direction === 'increases_probability'
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
                {result.PositiveSignals && result.PositiveSignals.length > 0 ? (
                  result.PositiveSignals.map((m, i) => (
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
                {result.NegativeSignals && result.NegativeSignals.length > 0 ? (
                  result.NegativeSignals.map((m, i) => (
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
            <div className="flex flex-col gap-3">
              <a
                href={`https://www.google.com/search?q=${encodeURIComponent(`buy ${result.brand || ''} ${result.productName || ''}`)}&tbm=shop`}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary flex items-center justify-center gap-2 py-3 text-xs">
                🔍 unde cumpăr — Google Shopping
              </a>
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
              <p><strong className="text-ink">Scorul baseline</strong> 50% rating · 20% recenzii · 20% loves · 10% preț/oz.</p>
              <p><strong className="text-ink">Modelul ML</strong> regresie logistică antrenată pe ~9.000 produse Sephora, 
cu threshold la percentila 75 a scorului baseline.</p>
              <p><strong className="text-ink">Compatibilitatea</strong> sistem euristic bazat pe reguli · tip ten, preocupare, buget.</p>
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