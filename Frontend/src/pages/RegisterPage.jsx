import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { register as registerApi } from '../api/auth'

const SKIN_TYPES = [
  { value: 'oily', label: 'gras', icon: '💧' },
  { value: 'dry', label: 'uscat', icon: '☀️' },
  { value: 'combination', label: 'mixt', icon: '⚖️' },
  { value: 'sensitive', label: 'sensibil', icon: '🌸' },
  { value: 'normal', label: 'normal', icon: '✓' },
]

const CONCERNS = [
  { value: 'acne', label: 'acnee', icon: '🔬' },
  { value: 'dehydration', label: 'deshidratare', icon: '💦' },
  { value: 'anti_aging', label: 'anti-aging', icon: '✨' },
  { value: 'dark_spots', label: 'pete', icon: '🎯' },
  { value: 'redness', label: 'roșeață', icon: '🌿' },
  { value: 'dullness', label: 'ten tern', icon: '🌙' },
]

const BUDGETS = [
  { value: 'low', label: 'redus', desc: 'sub $30' },
  { value: 'medium', label: 'mediu', desc: '$30–$80' },
  { value: 'high', label: 'ridicat', desc: 'peste $80' },
]

export default function RegisterPage() {
  const { login } = useAuth()
  const navigate = useNavigate()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [skinType, setSkinType] = useState('dry')
  const [mainConcern, setMainConcern] = useState('anti_aging')
  const [budgetLevel, setBudgetLevel] = useState('medium')
  const [agreed, setAgreed] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const passwordStrength = () => {
    if (password.length === 0) return 0
    let score = 0
    if (password.length >= 8) score++
    if (/[A-Z]/.test(password)) score++
    if (/[0-9]/.test(password)) score++
    if (/[^A-Za-z0-9]/.test(password)) score++
    return score
  }

  const strengthLabel = ['', 'slabă', 'medie', 'bună', 'excelentă']
  const strengthColor = ['', 'bg-red-400', 'bg-orange-400', 'bg-yellow-400', 'bg-green-500']

  const handleSubmit = async (e) => {
  e.preventDefault()
  setError('')

  if (password !== confirmPassword) {
    setError('Parolele nu coincid.')
    return
  }
  if (!agreed) {
    setError('Trebuie să fii de acord cu termenii.')
    return
  }
  if (passwordStrength() < 2) {
    setError('Parola este prea slabă. Adaugă litere mari și cifre.')
    return
  }

  setLoading(true)
  try {
    const res = await registerApi(email, password, skinType, mainConcern, budgetLevel)
    const { token, email: userEmail, role } = res.data
    login(token, { email: userEmail, role, skinType, mainConcern, budgetLevel })
    navigate('/')
  } catch (err) {
    console.error('Register error:', err)
    const msg = err.response?.data
    if (typeof msg === 'string') setError(msg)
    else if (msg?.errors) setError(Object.values(msg.errors).flat().join(' '))
    else setError('Eroare la înregistrare. Încearcă din nou.')
  } finally {
    setLoading(false)
  }
}

  const strength = passwordStrength()

  return (
    <div className="grid grid-cols-2 min-h-[calc(100vh-65px)]">

      {/* STÂNGA */}
      <div className="bg-rose-light border-r border-rose-border px-11 py-12 flex flex-col justify-between">
        <div className="flex flex-col gap-8">
          <div>
            <h1 className="font-serif text-4xl font-light leading-tight text-ink">
              Începe rutina<br />
              ta <em className="italic text-rose-primary">inteligentă.</em>
            </h1>
            <p className="text-sm text-muted mt-4 leading-relaxed max-w-xs">
              Creează-ți contul în 2 minute și primești recomandări personalizate instant.
            </p>
          </div>

          {/* PAȘI */}
          <div className="flex flex-col gap-0">
            <Step number="1" title="Creezi contul" desc="Email + parolă sau rapid cu Google." last={false} />
            <Step number="2" title="Îți definești profilul de ten" desc="Tip de ten, preocupare principală, buget." last={false} />
            <Step number="3" title="Primești recomandări" desc="Algoritmul evaluează produsele pentru tine." last={true} />
          </div>
        </div>

        <p className="text-xs text-soft">
          Ai deja cont?{' '}
          <Link to="/login" className="text-rose-primary hover:underline">
            Autentifică-te →
          </Link>
        </p>
      </div>

      {/* DREAPTA */}
      <div className="bg-cream px-11 py-10 flex flex-col gap-5 overflow-y-auto">
        <div>
          <h2 className="text-xl font-medium tracking-tight">Creează cont gratuit</h2>
          <p className="text-sm text-muted mt-1">
            Ai deja cont?{' '}
            <Link to="/login" className="text-rose-primary hover:underline">
              Autentifică-te
            </Link>
          </p>
        </div>

        {/* GOOGLE */}
        <button className="flex items-center justify-center gap-3 w-full py-3 border border-gray-200 rounded-lg bg-white text-sm font-medium hover:border-rose-mid transition-colors">
          <GoogleIcon />
          Înregistrează-te cu Google
        </button>

        <Divider />

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {error && (
            <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
              {error}
            </div>
          )}

          {/* EMAIL */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium tracking-widest text-gray-500">ADRESĂ EMAIL <span className="text-rose-primary">*</span></label>
            <input
              type="email"
              className="input-field"
              placeholder="ana@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          {/* PAROLE */}
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium tracking-widest text-gray-500">PAROLĂ <span className="text-rose-primary">*</span></label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  className="input-field pr-10"
                  placeholder="min. 8 caractere"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button type="button" onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-soft text-sm">
                  {showPassword ? '🙈' : '👁️'}
                </button>
              </div>
              {password.length > 0 && (
                <div>
                  <div className="flex gap-1 mt-1">
                    {[1,2,3,4].map(i => (
                      <div key={i} className={`flex-1 h-1 rounded-full ${i <= strength ? strengthColor[strength] : 'bg-rose-border'}`} />
                    ))}
                  </div>
                  <div className="text-xs text-soft mt-1">putere: {strengthLabel[strength]}</div>
                </div>
              )}
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium tracking-widest text-gray-500">CONFIRMĂ PAROLA <span className="text-rose-primary">*</span></label>
              <input
                type="password"
                className="input-field"
                placeholder="repetă parola"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
              {confirmPassword.length > 0 && password !== confirmPassword && (
                <div className="text-xs text-red-500 mt-1">parolele nu coincid</div>
              )}
            </div>
          </div>

          {/* TIP TEN */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-medium tracking-widest text-gray-500">TIPUL TĂU DE TEN <span className="text-rose-primary">*</span></label>
            <div className="grid grid-cols-5 gap-2">
              {SKIN_TYPES.map(s => (
                <button key={s.value} type="button"
                  onClick={() => setSkinType(s.value)}
                  className={`flex flex-col items-center gap-1 py-2 px-1 rounded-lg border text-xs transition-all cursor-pointer
                    ${skinType === s.value
                      ? 'border-rose-primary bg-rose-light text-rose-dark font-medium'
                      : 'border-gray-200 bg-white text-muted hover:border-rose-mid'}`}>
                  <span className="text-base">{s.icon}</span>
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* PREOCUPARE */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-medium tracking-widest text-gray-500">PREOCUPARE PRINCIPALĂ <span className="text-rose-primary">*</span></label>
            <div className="grid grid-cols-3 gap-2">
              {CONCERNS.map(c => (
                <button key={c.value} type="button"
                  onClick={() => setMainConcern(c.value)}
                  className={`flex items-center gap-2 py-2 px-3 rounded-lg border text-xs transition-all cursor-pointer
                    ${mainConcern === c.value
                      ? 'border-rose-primary bg-rose-light text-rose-dark font-medium'
                      : 'border-gray-200 bg-white text-muted hover:border-rose-mid'}`}>
                  <span>{c.icon}</span>
                  {c.label}
                </button>
              ))}
            </div>
          </div>

          {/* BUGET */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-medium tracking-widest text-gray-500">NIVEL BUGET</label>
            <div className="grid grid-cols-3 gap-2">
              {BUDGETS.map(b => (
                <button key={b.value} type="button"
                  onClick={() => setBudgetLevel(b.value)}
                  className={`flex flex-col items-center py-2.5 px-3 rounded-lg border text-xs transition-all cursor-pointer
                    ${budgetLevel === b.value
                      ? 'border-rose-primary bg-rose-light text-rose-dark font-medium'
                      : 'border-gray-200 bg-white text-muted hover:border-rose-mid'}`}>
                  <span className="font-medium">{b.label}</span>
                  <span className="text-soft mt-0.5">{b.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* TERMENI */}
          <label className="flex items-start gap-2 cursor-pointer">
            <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)}
              className="accent-rose-primary mt-0.5 flex-shrink-0" />
            <span className="text-xs text-muted leading-relaxed">
              Sunt de acord cu{' '}
              <span className="text-rose-primary cursor-pointer">termenii și condițiile</span> și{' '}
              <span className="text-rose-primary cursor-pointer">politica de confidențialitate</span> SkinIQ.
            </span>
          </label>

          <button
            type="submit"
            disabled={loading}
            className="btn-primary disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {loading ? 'se creează contul...' : 'creează cont'}
          </button>
        </form>
      </div>
    </div>
  )
}

function Step({ number, title, desc, last }) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className="w-7 h-7 rounded-full bg-rose-primary text-white text-xs font-medium flex items-center justify-center flex-shrink-0">
          {number}
        </div>
        {!last && <div className="w-px flex-1 bg-rose-border my-1 min-h-[20px]" />}
      </div>
      <div className="pb-5">
        <div className="text-sm font-medium text-ink">{title}</div>
        <div className="text-xs text-muted mt-0.5">{desc}</div>
      </div>
    </div>
  )
}

function Divider() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-px bg-rose-border" />
      <span className="text-xs text-soft tracking-widest">sau cu email</span>
      <div className="flex-1 h-px bg-rose-border" />
    </div>
  )
}

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
    </svg>
  )
}