import { useState } from 'react'
import { Link } from 'react-router-dom'
import { forgotPassword } from '../api/auth'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [sent, setSent] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await forgotPassword(email)
      setSent(true)
    } catch (err) {
      const msg = err.response?.data
      if (typeof msg === 'string') setError(msg)
      else setError('A apărut o eroare. Încearcă din nou.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-2 min-h-[calc(100vh-65px)]">

      {/* STÂNGA */}
      <div className="bg-rose-light border-r border-rose-border px-11 py-12 flex flex-col justify-between">
        <div className="flex flex-col gap-8">
          <div>
            <h1 className="font-serif text-4xl font-light leading-tight text-ink">
              Ai uitat<br />
              <em className="italic text-rose-primary">parola?</em>
            </h1>
            <p className="text-sm text-muted mt-4 leading-relaxed max-w-xs">
              Nicio problemă. Introduci emailul cu care te-ai înregistrat și îți trimitem un link de resetare.
            </p>
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-full bg-rose-border flex items-center justify-center flex-shrink-0 text-base">
                📧
              </div>
              <div>
                <div className="text-sm font-medium text-ink">Trimitem un email</div>
                <div className="text-xs text-muted mt-0.5 leading-relaxed">
                  Vei primi un link valabil 1 oră pentru resetarea parolei.
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-full bg-rose-border flex items-center justify-center flex-shrink-0 text-base">
                🔐
              </div>
              <div>
                <div className="text-sm font-medium text-ink">Setezi o parolă nouă</div>
                <div className="text-xs text-muted mt-0.5 leading-relaxed">
                  Accesezi link-ul din email și introduci o parolă nouă sigură.
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-full bg-rose-border flex items-center justify-center flex-shrink-0 text-base">
                ✅
              </div>
              <div>
                <div className="text-sm font-medium text-ink">Te autentifici normal</div>
                <div className="text-xs text-muted mt-0.5 leading-relaxed">
                  Folosești noua parolă pentru a intra în cont.
                </div>
              </div>
            </div>
          </div>
        </div>

        <p className="text-xs text-soft">
          Ți-ai amintit parola?{' '}
          <Link to="/login" className="text-rose-primary hover:underline">
            Întoarce-te la autentificare →
          </Link>
        </p>
      </div>

      {/* DREAPTA */}
      <div className="bg-cream px-11 py-12 flex flex-col justify-center gap-7">
        {!sent ? (
          <>
            <div>
              <h2 className="text-xl font-medium tracking-tight">Resetare parolă</h2>
              <p className="text-sm text-muted mt-1">
                Introdu emailul contului tău SkinIQ.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              {error && (
                <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
                  {error}
                </div>
              )}

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium tracking-widest text-gray-500">
                  ADRESĂ EMAIL
                </label>
                <input
                  type="email"
                  className="input-field"
                  placeholder="ana@example.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="btn-primary disabled:opacity-60 disabled:cursor-not-allowed">
                {loading ? 'se trimite...' : 'trimite link de resetare'}
              </button>
            </form>

            <div className="text-xs text-soft text-center leading-relaxed">
              Dacă emailul există în sistem, vei primi un link în câteva secunde.
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-6 text-center">
            <div className="text-6xl">📬</div>
            <div>
              <h2 className="font-serif text-2xl font-light text-ink">
                Email trimis!
              </h2>
              <p className="text-sm text-muted mt-3 leading-relaxed max-w-xs">
                Verifică inbox-ul pentru <strong>{email}</strong>. Link-ul este valabil 1 oră.
              </p>
            </div>
            <div className="text-xs text-soft leading-relaxed max-w-xs">
              Nu ai primit emailul? Verifică folderul Spam sau{' '}
              <button onClick={() => setSent(false)} className="text-rose-primary hover:underline cursor-pointer">
                încearcă din nou
              </button>.
            </div>
            <Link to="/login" className="btn-outline px-8 py-3">
              înapoi la autentificare
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}