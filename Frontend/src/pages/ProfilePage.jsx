import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { getProfile, updateProfile } from '../api/auth'
import { getHistory, deleteEvaluation } from '../api/evaluate'

const SKIN_TYPES = [
  { value: 'oily',        label: 'gras',     icon: '💧' },
  { value: 'dry',         label: 'uscat',    icon: '☀️' },
  { value: 'combination', label: 'mixt',     icon: '⚖️' },
  { value: 'sensitive',   label: 'sensibil', icon: '🌸' },
  { value: 'normal',      label: 'normal',   icon: '✓'  },
]

const CONCERNS = [
  { value: 'acne',        label: 'acnee',        icon: '🔬' },
  { value: 'dehydration', label: 'deshidratare', icon: '💦' },
  { value: 'anti_aging',  label: 'anti-aging',   icon: '✨' },
  { value: 'dark_spots',  label: 'pete',         icon: '🎯' },
  { value: 'redness',     label: 'roșeată',      icon: '🌿' },
  { value: 'dullness',    label: 'ten tern',     icon: '🌙' },
]

const BUDGETS = [
  { value: 'low',    label: 'redus',   desc: 'sub $30'    },
  { value: 'medium', label: 'mediu',   desc: '$30–$80'    },
  { value: 'high',   label: 'ridicat', desc: 'peste $80'  },
]

const VERDICT_STYLES = {
  'Recomandat':   { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-800', badge: 'bg-green-100 text-green-800' },
  'Nerecomandat': { bg: 'bg-red-50',   border: 'border-red-200',   text: 'text-red-800',   badge: 'bg-red-100 text-red-800'   },
  default:        { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-800', badge: 'bg-amber-100 text-amber-800' },
}

function getVerdictStyle(verdict) {
  return VERDICT_STYLES[verdict] || VERDICT_STYLES.default
}

export default function ProfilePage() {
  const { user, logout, login, token } = useAuth()
  const navigate = useNavigate()

  const [activeTab, setActiveTab]           = useState('profil')
  const [profile, setProfile]               = useState(null)
  const [history, setHistory]               = useState([])
  const [loadingProfile, setLoadingProfile] = useState(true)
  const [loadingHistory, setLoadingHistory] = useState(true)

  const [editingSkin, setEditingSkin]   = useState(false)
  const [skinType, setSkinType]         = useState('')
  const [mainConcern, setMainConcern]   = useState('')
  const [budgetLevel, setBudgetLevel]   = useState('')
  const [savingSkin, setSavingSkin]     = useState(false)
  const [skinSuccess, setSkinSuccess]   = useState(false)

  useEffect(() => {
    if (!user) { navigate('/login'); return }
    getProfile()
      .then(res => {
        setProfile(res.data)
        setSkinType(res.data.skinType || 'dry')
        setMainConcern(res.data.mainConcern || 'anti_aging')
        setBudgetLevel(res.data.budgetLevel || 'medium')
      })
      .catch(() => navigate('/login'))
      .finally(() => setLoadingProfile(false))

    getHistory()
      .then(res => setHistory(res.data || []))
      .catch(() => setHistory([]))
      .finally(() => setLoadingHistory(false))
  }, [])

  const handleSaveSkin = async () => {
    setSavingSkin(true)
    try {
      await updateProfile(skinType, mainConcern, budgetLevel)
      setProfile(p => ({ ...p, skinType, mainConcern, budgetLevel }))
      login(token, { ...user, skinType, mainConcern, budgetLevel })
      setSkinSuccess(true)
      setTimeout(() => setSkinSuccess(false), 3000)
      setEditingSkin(false)
    } catch {
      alert('Eroare la salvare. Încearcă din nou.')
    } finally {
      setSavingSkin(false)
    }
  }

  const handleDelete = async (id) => {
    if (!window.confirm('Ștergi această evaluare?')) return
    try {
      await deleteEvaluation(id)
      setHistory(h => h.filter(e => e.id !== id))
    } catch {
      alert('Eroare la ștergere.')
    }
  }

  const handleLogout = () => { logout(); navigate('/') }

  const initials  = user?.email?.slice(0, 2).toUpperCase() || 'AN'
  const recCount  = history.filter(h => h.finalVerdict === 'Recomandat').length
  const noCount   = history.filter(h => h.finalVerdict === 'Nerecomandat').length

  const isGoogleAccount = user?.email && !profile?.createdWithPassword

  if (loadingProfile) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="font-serif text-2xl text-rose-primary animate-pulse">se încarcă...</div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-[240px_1fr] min-h-[calc(100vh-65px)]">

      {/* SIDEBAR */}
      <div className="bg-cream-warm border-r border-rose-border px-5 py-7 flex flex-col gap-7">
        <div className="flex flex-col items-center gap-2">
          <div className="w-16 h-16 rounded-full bg-rose-border flex items-center justify-center font-serif text-2xl font-light text-rose-dark border-2 border-rose-mid">
            {initials}
          </div>
          <div className="text-sm font-medium text-ink">{user?.email}</div>
          <div className="text-xs bg-rose-light border border-rose-border rounded-full px-3 py-1 text-rose-dark">
            cont activ
          </div>
        </div>

        <div className="flex flex-col gap-1">
          {[
            { id: 'profil',     label: 'profilul meu',      icon: '👤' },
            { id: 'istoric',    label: 'istoric evaluări',  icon: '📋' },
            { id: 'securitate', label: 'securitate',        icon: '🔒' },
          ].map(item => (
            <button key={item.id} onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-left transition-colors cursor-pointer
                ${activeTab === item.id
                  ? 'bg-rose-light text-rose-primary font-medium'
                  : 'text-muted hover:bg-gray-50 hover:text-ink'}`}>
              <span>{item.icon}</span>
              {item.label}
            </button>
          ))}
        </div>

        <button onClick={handleLogout}
          className="flex items-center gap-2 px-3 py-2 text-xs text-soft hover:text-red-500 transition-colors mt-auto cursor-pointer">
          🚪 deconectare
        </button>
      </div>

      {/* CONȚINUT */}
      <div className="px-9 py-8 flex flex-col gap-8">

        {/* TAB: PROFIL */}
        {activeTab === 'profil' && (
          <>
            {skinSuccess && (
              <div className="text-xs text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-3">
                ✓ Profilul a fost actualizat cu succes!
              </div>
            )}

            <Section title="Informații cont">
              <div className="grid grid-cols-2 gap-5">
                <InfoField label="EMAIL" value={profile?.email} />
                <InfoField label="CONT CREAT" value={new Date(profile?.createdAt || Date.now()).toLocaleDateString('ro-RO', { day: 'numeric', month: 'long', year: 'numeric' })} />
                <InfoField label="ROL" value={profile?.role === 'Admin' ? 'administrator' : 'utilizator'} />
              </div>
            </Section>

            <Section
              title="Profilul de ten"
              action={
                editingSkin
                  ? <button onClick={handleSaveSkin} disabled={savingSkin}
                      className="btn-primary text-xs py-1.5 px-4 disabled:opacity-60">
                      {savingSkin ? 'se salvează...' : '✓ salvează'}
                    </button>
                  : <button onClick={() => setEditingSkin(true)}
                      className="btn-outline text-xs py-1.5 px-4">
                      ✏️ editează
                    </button>
              }
            >
              {!editingSkin ? (
                <div className="grid grid-cols-3 gap-4">
                  <ProfileCard
                    label="TIP TEN"
                    value={SKIN_TYPES.find(s => s.value === (profile?.skinType || skinType))?.label || skinType}
                    icon={SKIN_TYPES.find(s => s.value === (profile?.skinType || skinType))?.icon || '✨'}
                  />
                  <ProfileCard
                    label="PREOCUPARE"
                    value={CONCERNS.find(c => c.value === (profile?.mainConcern || mainConcern))?.label || mainConcern}
                    icon={CONCERNS.find(c => c.value === (profile?.mainConcern || mainConcern))?.icon || '🎯'}
                  />
                  <ProfileCard
                    label="BUGET"
                    value={BUDGETS.find(b => b.value === (profile?.budgetLevel || budgetLevel))?.label || budgetLevel}
                    icon="💰"
                  />
                </div>
              ) : (
                <div className="flex flex-col gap-5">
                  <div>
                    <div className="text-xs font-medium tracking-widest text-gray-500 mb-2">TIP TEN</div>
                    <div className="grid grid-cols-5 gap-2">
                      {SKIN_TYPES.map(s => (
                        <button key={s.value} type="button" onClick={() => setSkinType(s.value)}
                          className={`flex flex-col items-center gap-1 py-2 rounded-lg border text-xs transition-all cursor-pointer
                            ${skinType === s.value ? 'border-rose-primary bg-rose-light text-rose-dark font-medium' : 'border-gray-200 bg-white text-muted hover:border-rose-mid'}`}>
                          <span>{s.icon}</span>{s.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs font-medium tracking-widest text-gray-500 mb-2">PREOCUPARE PRINCIPALĂ</div>
                    <div className="grid grid-cols-3 gap-2">
                      {CONCERNS.map(c => (
                        <button key={c.value} type="button" onClick={() => setMainConcern(c.value)}
                          className={`flex items-center gap-2 py-2 px-3 rounded-lg border text-xs transition-all cursor-pointer
                            ${mainConcern === c.value ? 'border-rose-primary bg-rose-light text-rose-dark font-medium' : 'border-gray-200 bg-white text-muted hover:border-rose-mid'}`}>
                          <span>{c.icon}</span>{c.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs font-medium tracking-widest text-gray-500 mb-2">NIVEL BUGET</div>
                    <div className="grid grid-cols-3 gap-2">
                      {BUDGETS.map(b => (
                        <button key={b.value} type="button" onClick={() => setBudgetLevel(b.value)}
                          className={`flex flex-col items-center py-2.5 rounded-lg border text-xs transition-all cursor-pointer
                            ${budgetLevel === b.value ? 'border-rose-primary bg-rose-light text-rose-dark font-medium' : 'border-gray-200 bg-white text-muted hover:border-rose-mid'}`}>
                          <span className="font-medium">{b.label}</span>
                          <span className="text-soft">{b.desc}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </Section>

            <Section title="Activitatea mea">
              <div className="grid grid-cols-3 gap-4">
                <StatCard num={history.length} label="produse evaluate" />
                <StatCard num={recCount} label="recomandate" color="text-green-600" />
                <StatCard num={noCount} label="nerecomandate" color="text-red-500" />
              </div>
            </Section>
          </>
        )}

        {/* TAB: ISTORIC */}
        {activeTab === 'istoric' && (
          <Section title="Istoricul evaluărilor">
            {loadingHistory ? (
              <div className="text-sm text-muted animate-pulse">se încarcă...</div>
            ) : history.length === 0 ? (
              <div className="text-center py-12 text-muted">
                <div className="text-4xl mb-3">📋</div>
                <div className="text-sm">Nu ai nicio evaluare încă.</div>
                <Link to="/evaluate" className="text-xs text-rose-primary hover:underline mt-2 inline-block">
                  Evaluează primul tău produs →
                </Link>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {history.map(entry => {
                  const style = getVerdictStyle(entry.finalVerdict)
                  return (
                    <div key={entry.id}
                      className={`flex items-center gap-4 p-4 rounded-xl border ${style.border} ${style.bg}`}>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium tracking-widest text-soft">{entry.brand}</div>
                        <div className="text-sm font-medium text-ink mt-0.5 truncate">{entry.name || entry.productId}</div>
                        <div className="text-xs text-muted mt-1">
                          {new Date(entry.createdAt).toLocaleDateString('ro-RO', { day: 'numeric', month: 'long', year: 'numeric' })}
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-1.5 flex-shrink-0">
                        <div className="font-serif text-xl font-light text-rose-primary">
                          {entry.mlProbability ? `${Math.round(entry.mlProbability * 100)}%` : '—'}
                        </div>
                        <div className={`text-xs px-2 py-0.5 rounded font-medium ${style.badge}`}>
                          {entry.finalVerdict}
                        </div>
                      </div>
                      <button onClick={() => handleDelete(entry.id)}
                        className="text-soft hover:text-red-500 transition-colors text-sm cursor-pointer flex-shrink-0"
                        title="șterge evaluare">
                        🗑️
                      </button>
                    </div>
                  )
                })}
              </div>
            )}
          </Section>
        )}

        {/* TAB: SECURITATE */}
        {activeTab === 'securitate' && (
          <Section title="Securitate">
            <div className="flex flex-col gap-4">
              <div className="p-4 bg-rose-light border border-rose-border rounded-xl text-sm text-rose-dark">
                <div className="font-medium mb-2">🔐 Resetare parolă</div>
                <div className="text-xs text-muted mb-3">
                  Poți reseta parola prin email. Vei primi un link valabil 1 oră.
                </div>
                <button
                  onClick={() => window.location.href = '/forgot-password'}
                  className="btn-outline text-xs py-2 px-4">
                  trimite link de resetare
                </button>
              </div>
              <div className="p-4 bg-cream-warm border border-rose-border rounded-xl text-sm">
                <div className="font-medium mb-1 text-ink">📧 Email înregistrat</div>
                <div className="text-xs text-muted">{user?.email}</div>
              </div>
            </div>
          </Section>
        )}

      </div>
    </div>
  )
}

function Section({ title, children, action }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between pb-3 border-b border-rose-border">
        <h2 className="section-title">{title}</h2>
        {action}
      </div>
      {children}
    </div>
  )
}

function InfoField({ label, value }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-xs font-medium tracking-widest text-soft">{label}</div>
      <div className="text-sm text-ink">{value || '—'}</div>
    </div>
  )
}

function ProfileCard({ label, value, icon }) {
  return (
    <div className="card flex flex-col gap-2">
      <div className="text-xs font-medium tracking-widest text-soft">{label}</div>
      <div className="flex items-center gap-2 text-sm font-medium text-ink">
        <span className="text-lg">{icon}</span>
        {value}
      </div>
    </div>
  )
}

function StatCard({ num, label, color = 'text-rose-primary' }) {
  return (
    <div className="card text-center">
      <div className={`font-serif text-3xl font-light ${color}`}>{num}</div>
      <div className="text-xs text-muted mt-1">{label}</div>
    </div>
  )
}