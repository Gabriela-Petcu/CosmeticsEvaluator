import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { login as loginApi, googleLogin, getProfile } from '../api/auth'
import { useGoogleLogin } from '@react-oauth/google'

export default function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [remember, setRemember] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
  e.preventDefault()
  setError('')
  setLoading(true)
  try {
    const res = await loginApi(email, password)
    const { token, email: userEmail, role } = res.data

    // Salvăm token-ul înainte de getProfile
    localStorage.setItem('skiniq_token', token)

    let skinType = 'normal'
    let mainConcern = 'anti_aging'
    let budgetLevel = 'medium'

    try {
      const profileRes = await getProfile()
      skinType = profileRes.data.skinType || skinType
      mainConcern = profileRes.data.mainConcern || mainConcern
      budgetLevel = profileRes.data.budgetLevel || budgetLevel
    } catch {
      // Profilul nu e critic — continuăm cu valorile default
    }

    login(token, {
      email: userEmail,
      role,
      skinType,
      mainConcern,
      budgetLevel,
    })

    navigate('/')
  } catch (err) {
    console.error('Login error:', err)
    const msg = err.response?.data
    if (typeof msg === 'string') setError(msg)
    else setError('Email sau parolă incorectă.')
  } finally {
    setLoading(false)
  }
}
const handleGoogleLogin = useGoogleLogin({
  onSuccess: async (tokenResponse) => {
    setLoading(true)
    setError('')
    try {
      // Obținem datele userului de la Google
      const userInfo = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
        headers: { Authorization: `Bearer ${tokenResponse.access_token}` }
      }).then(r => r.json())

      // Trimitem id_token la backend
      const res = await googleLogin(tokenResponse.access_token)
      const { token, email: userEmail, role } = res.data

      localStorage.setItem('skiniq_token', token)

      let skinType = 'normal'
      let mainConcern = 'anti_aging'
      let budgetLevel = 'medium'

      try {
        const profileRes = await getProfile()
        skinType = profileRes.data.skinType || skinType
        mainConcern = profileRes.data.mainConcern || mainConcern
        budgetLevel = profileRes.data.budgetLevel || budgetLevel
      } catch {}

      login(token, { email: userEmail, role, skinType, mainConcern, budgetLevel })
      navigate('/')
    } catch (err) {
      console.error('Google login error:', err)
      setError('Autentificarea cu Google a eșuat. Încearcă din nou.')
    } finally {
      setLoading(false)
    }
  },
  onError: () => setError('Autentificarea cu Google a fost anulată.')
})

  return (
    <div className="grid grid-cols-2 min-h-[calc(100vh-65px)]">

      {/* STÂNGA */}
      <div className="bg-rose-light border-r border-rose-border px-11 py-12 flex flex-col justify-between">
        <div className="flex flex-col gap-8">
          <div>
            <h1 className="font-serif text-4xl font-light leading-tight text-ink">
              Bine ai revenit.<br />
              Pielea ta<br />
              <em className="italic text-rose-primary">te-a așteptat.</em>
            </h1>
            <p className="text-sm text-muted mt-4 leading-relaxed max-w-xs">
              Intră în cont și descoperă produsele potrivite pentru profilul tău de ten.
            </p>
          </div>

          <div className="flex flex-col gap-5">
            <Feature icon="🧠" title="Evaluare bazată pe ML" desc="Modelul nostru analizează 9.000+ produse în timp real." />
            <Feature icon="✨" title="Personalizat pentru tine" desc="Recomandări adaptate tipului tău de ten și bugetului." />
            <Feature icon="🛡️" title="Fără sponsorizări" desc="Nicio marcă nu plătește pentru a apărea recomandată." />
          </div>
        </div>

        <p className="text-xs text-soft">
          Nu ai cont?{' '}
          <Link to="/register" className="text-rose-primary hover:underline">
            Creează unul gratuit →
          </Link>
        </p>
      </div>

      {/* DREAPTA */}
      <div className="bg-cream px-11 py-12 flex flex-col justify-center gap-7">
        <div>
          <h2 className="text-xl font-medium tracking-tight">Autentificare</h2>
          <p className="text-sm text-muted mt-1">
            Nu ai cont?{' '}
            <Link to="/register" className="text-rose-primary hover:underline">
              Înregistrează-te gratuit
            </Link>
          </p>
        </div>

        {/* GOOGLE */}
        <button
  onClick={() => handleGoogleLogin()}
  disabled={loading}
  className="flex items-center justify-center gap-3 w-full py-3 border border-gray-200 rounded-lg bg-white text-sm font-medium hover:border-rose-mid transition-colors disabled:opacity-60">
  <GoogleIcon />
  Continuă cu Google
</button>

        <Divider />

        {/* FORMULAR */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {error && (
            <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
              {error}
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium tracking-widest text-gray-500">ADRESĂ EMAIL</label>
            <input
              type="email"
              className="input-field"
              placeholder="ana@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium tracking-widest text-gray-500">PAROLĂ</label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                className="input-field pr-10"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-soft hover:text-muted text-sm"
              >
                {showPassword ? '🙈' : '👁️'}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 text-xs text-muted cursor-pointer">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
                className="accent-rose-primary"
              />
              ține-mă conectată
            </label>
            <Link to="/forgot-password" className="text-xs text-rose-primary hover:underline">
  ai uitat parola?
</Link>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn-primary mt-1 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {loading ? 'se verifică...' : 'intră în cont'}
          </button>
        </form>

        <p className="text-xs text-soft text-center leading-relaxed">
          Prin autentificare, ești de acord cu{' '}
          <span className="text-rose-primary cursor-pointer">termenii</span> și{' '}
          <span className="text-rose-primary cursor-pointer">politica de confidențialitate</span> SkinIQ.
        </p>
      </div>
    </div>
  )
}

function Feature({ icon, title, desc }) {
  return (
    <div className="flex items-start gap-3">
      <div className="w-9 h-9 rounded-full bg-rose-border flex items-center justify-center flex-shrink-0 text-base">
        {icon}
      </div>
      <div>
        <div className="text-sm font-medium text-ink">{title}</div>
        <div className="text-xs text-muted mt-0.5 leading-relaxed">{desc}</div>
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