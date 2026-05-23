import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { resetPassword } from '../api/auth'

export default function ResetPasswordPage() {
  const navigate = useNavigate()
  const [token, setToken] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const t = params.get('token')
    const e = params.get('email')
    if (!t || !e) {
      setError('Link invalid sau expirat. Solicită un nou link de resetare.')
      return
    }
    setToken(t)
    setEmail(e)
  }, [])

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
  const strength = passwordStrength()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Parolele nu coincid.')
      return
    }
    if (strength < 2) {
      setError('Parola este prea slabă.')
      return
    }

    setLoading(true)
    try {
      await resetPassword(token, email, password)
      setSuccess(true)
      setTimeout(() => navigate('/login'), 3000)
    } catch (err) {
      const msg = err.response?.data
      if (typeof msg === 'string') setError(msg)
      else setError('A apărut o eroare. Link-ul poate fi expirat.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-2 min-h-[calc(100vh-65px)]">

      {/* STÂNGA */}
      <div className="bg-rose-light border-r border-rose-border px-11 py-12 flex flex-col justify-between">
        <div className="flex flex-col gap-6">
          <div>
            <h1 className="font-serif text-4xl font-light leading-tight text-ink">
              Parolă<br />
              <em className="italic text-rose-primary">nouă.</em>
            </h1>
            <p className="text-sm text-muted mt-4 leading-relaxed max-w-xs">
              Alege o parolă sigură pentru contul tău SkinIQ.
            </p>
          </div>

          <div className="bg-white border border-rose-border rounded-xl p-5 flex flex-col gap-3">
            <div className="text-xs font-medium tracking-widest text-soft">CERINȚE PAROLĂ</div>
            {[
              { text: 'Minim 8 caractere', ok: password.length >= 8 },
              { text: 'O literă mare (A-Z)', ok: /[A-Z]/.test(password) },
              { text: 'O literă mică (a-z)', ok: /[a-z]/.test(password) },
              { text: 'O cifră (0-9)', ok: /[0-9]/.test(password) },
            ].map(req => (
              <div key={req.text} className="flex items-center gap-2 text-xs">
                <span className={req.ok ? 'text-green-500' : 'text-soft'}>
                  {req.ok ? '✓' : '○'}
                </span>
                <span className={req.ok ? 'text-green-700' : 'text-muted'}>
                  {req.text}
                </span>
              </div>
            ))}
          </div>
        </div>

        <p className="text-xs text-soft">
          <Link to="/login" className="text-rose-primary hover:underline">
            ← înapoi la autentificare
          </Link>
        </p>
      </div>

      {/* DREAPTA */}
      <div className="bg-cream px-11 py-12 flex flex-col justify-center gap-7">
        {success ? (
          <div className="flex flex-col items-center gap-6 text-center">
            <div className="text-6xl">🎉</div>
            <div>
              <h2 className="font-serif text-2xl font-light text-ink">
                Parolă resetată!
              </h2>
              <p className="text-sm text-muted mt-3 leading-relaxed">
                Parola ta a fost schimbată cu succes. Ești redirecționată la autentificare...
              </p>
            </div>
            <Link to="/login" className="btn-primary px-8 py-3">
              intră în cont
            </Link>
          </div>
        ) : (
          <>
            <div>
              <h2 className="text-xl font-medium tracking-tight">Setează parola nouă</h2>
              <p className="text-sm text-muted mt-1">
                Pentru contul <strong>{email}</strong>
              </p>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              {error && (
                <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
                  {error}
                  {error.includes('expirat') && (
                    <span> <Link to="/forgot-password" className="underline">Solicită un nou link</Link>.</span>
                  )}
                </div>
              )}

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium tracking-widest text-gray-500">
                  PAROLĂ NOUĂ
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    className="input-field pr-10"
                    placeholder="min. 8 caractere"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                  />
                  <button type="button"
                    onClick={() => setShowPassword(!showPassword)}
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
                <label className="text-xs font-medium tracking-widest text-gray-500">
                  CONFIRMĂ PAROLA
                </label>
                <input
                  type="password"
                  className="input-field"
                  placeholder="repetă parola"
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  required
                />
                {confirmPassword.length > 0 && password !== confirmPassword && (
                  <div className="text-xs text-red-500">parolele nu coincid</div>
                )}
              </div>

              <button
                type="submit"
                disabled={loading || strength < 2}
                className="btn-primary disabled:opacity-60 disabled:cursor-not-allowed">
                {loading ? 'se salvează...' : 'salvează parola nouă'}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  )
}