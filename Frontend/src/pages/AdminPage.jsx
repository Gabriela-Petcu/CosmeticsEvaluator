import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  getAdminStats, getAdminUsers, updateUserRole, deleteUser,
  getAdminProducts, addProduct, deleteProduct
} from '../api/admin'

export default function AdminPage() {
  const { user, isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('stats')

  const [stats, setStats] = useState(null)
  const [users, setUsers] = useState([])
  const [products, setProducts] = useState([])
  const [totalProducts, setTotalProducts] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)

  const [deleteModal, setDeleteModal] = useState(null)
  const [deleteType, setDeleteType] = useState('')

  const [showAddProduct, setShowAddProduct] = useState(false)
  const [newProduct, setNewProduct] = useState({
    brand: '', name: '', price: 0, nOfReviews: 0,
    nOfLoves: 0, reviewScore: 0, pricePerOunce: 0
  })

  useEffect(() => {
    if (!isAuthenticated || user?.role !== 'Admin') {
      navigate('/')
      return
    }
    loadData()
  }, [])

  useEffect(() => {
    if (activeTab === 'products') loadProducts()
  }, [page, search, activeTab])

  const loadData = async () => {
    setLoading(true)
    try {
      const [statsRes, usersRes] = await Promise.all([
        getAdminStats(),
        getAdminUsers()
      ])
      setStats(statsRes.data)
      setUsers(usersRes.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadProducts = async () => {
    try {
      const res = await getAdminProducts(page, search)
      setProducts(res.data.products || res.data.Products || [])
      setTotalProducts(res.data.total || res.data.Total || 0)
    } catch (err) {
      console.error(err)
    }
  }

  const handleRoleChange = async (id, newRole) => {
    try {
      await updateUserRole(id, newRole)
      setUsers(prev => prev.map(u => u.id === id ? { ...u, role: newRole } : u))
    } catch (err) {
      alert('Eroare la schimbarea rolului.')
    }
  }

  const handleDeleteConfirm = async () => {
    try {
      if (deleteType === 'user') {
        await deleteUser(deleteModal)
        setUsers(prev => prev.filter(u => u.id !== deleteModal))
      } else {
        await deleteProduct(deleteModal)
        setProducts(prev => prev.filter(p => p.id !== deleteModal))
        setTotalProducts(prev => prev - 1)
      }
    } catch (err) {
      alert('Eroare la ștergere.')
    } finally {
      setDeleteModal(null)
      setDeleteType('')
    }
  }

  const handleAddProduct = async () => {
    try {
      await addProduct(newProduct)
      setShowAddProduct(false)
      setNewProduct({ brand: '', name: '', price: 0, nOfReviews: 0, nOfLoves: 0, reviewScore: 0, pricePerOunce: 0 })
      loadProducts()
    } catch (err) {
      alert('Eroare la adăugarea produsului.')
    }
  }

  const VERDICT_COLOR = {
    'Recomandat': 'text-green-600',
    'Nerecomandat': 'text-red-500',
    default: 'text-amber-600'
  }

  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="font-serif text-2xl text-rose-primary animate-pulse">se încarcă panoul admin...</div>
    </div>
  )

  return (
    <div>
      {/* MODAL ȘTERGERE */}
      {deleteModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center px-4"
          onClick={() => setDeleteModal(null)}>
          <div className="bg-white rounded-2xl p-8 max-w-sm w-full shadow-xl text-center flex flex-col gap-5"
            onClick={e => e.stopPropagation()}>
            <div className="text-4xl">⚠️</div>
            <h2 className="font-serif text-2xl font-light text-ink">
              Confirmi ștergerea?
            </h2>
            <p className="text-sm text-muted leading-relaxed">
              Această acțiune este permanentă și nu poate fi anulată.
            </p>
            <div className="flex flex-col gap-2">
              <button onClick={handleDeleteConfirm}
                className="py-3 px-6 bg-red-500 text-white rounded-lg text-sm font-medium hover:bg-red-600 transition-colors cursor-pointer">
                da, șterge
              </button>
              <button onClick={() => setDeleteModal(null)} className="btn-outline py-3">
                anulează
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL ADAUGĂ PRODUS */}
      {showAddProduct && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center px-4"
          onClick={() => setShowAddProduct(false)}>
          <div className="bg-white rounded-2xl p-8 max-w-lg w-full shadow-xl flex flex-col gap-5"
            onClick={e => e.stopPropagation()}>
            <h2 className="font-serif text-2xl font-light text-ink">Adaugă produs nou</h2>
            <div className="grid grid-cols-2 gap-3">
              {[
                { key: 'brand', label: 'Brand', type: 'text' },
                { key: 'name', label: 'Nume produs', type: 'text' },
                { key: 'price', label: 'Preț ($)', type: 'number' },
                { key: 'pricePerOunce', label: 'Preț/oz', type: 'number' },
                { key: 'reviewScore', label: 'Rating (0-5)', type: 'number' },
                { key: 'nOfReviews', label: 'Nr. recenzii', type: 'number' },
                { key: 'nOfLoves', label: 'Nr. loves', type: 'number' },
              ].map(f => (
                <div key={f.key} className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium tracking-widest text-gray-500">{f.label.toUpperCase()}</label>
                  <input
                    type={f.type}
                    className="input-field"
                    value={newProduct[f.key]}
                    onChange={e => setNewProduct(prev => ({ ...prev, [f.key]: f.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value }))}
                  />
                </div>
              ))}
            </div>
            <div className="flex gap-3">
              <button onClick={handleAddProduct} className="btn-primary flex-1 py-3">
                adaugă produsul
              </button>
              <button onClick={() => setShowAddProduct(false)} className="btn-outline flex-1 py-3">
                anulează
              </button>
            </div>
          </div>
        </div>
      )}

      {/* HEADER */}
      <div className="bg-cream-warm border-b border-rose-border px-9 py-6">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-5 h-px bg-rose-primary" />
          <span className="text-xs tracking-widest text-rose-primary font-medium">PANOU ADMINISTRARE</span>
        </div>
        <h1 className="font-serif text-3xl font-light text-ink">
          Admin <em className="italic text-rose-primary">SkinIQ</em>
        </h1>
      </div>

      {/* TABS */}
      <div className="border-b border-rose-border bg-cream px-9 flex gap-1">
        {[
          { id: 'stats', label: '📊 statistici' },
          { id: 'users', label: '👥 utilizatori' },
          { id: 'products', label: '🧴 catalog produse' },
        ].map(tab => (
          <button key={tab.id} onClick={() => { setActiveTab(tab.id); if (tab.id === 'products') loadProducts() }}
            className={`text-xs px-5 py-3 border-b-2 transition-colors cursor-pointer
              ${activeTab === tab.id
                ? 'border-rose-primary text-rose-primary font-medium'
                : 'border-transparent text-muted hover:text-ink'}`}>
            {tab.label}
          </button>
        ))}
      </div>

      <div className="px-9 py-8">

        {/* TAB STATISTICI */}
        {activeTab === 'stats' && stats && (
          <div className="flex flex-col gap-8">
            <div className="grid grid-cols-3 gap-4">
              {[
                { num: stats.totalUsers || stats.TotalUsers, label: 'utilizatori înregistrați', icon: '👥' },
                { num: stats.totalEvaluations || stats.TotalEvaluations, label: 'evaluări totale', icon: '📋' },
                { num: stats.totalProducts || stats.TotalProducts, label: 'produse în catalog', icon: '🧴' },
              ].map(s => (
                <div key={s.label} className="card flex items-center gap-4">
                  <div className="text-3xl">{s.icon}</div>
                  <div>
                    <div className="font-serif text-3xl font-light text-rose-primary">{s.num}</div>
                    <div className="text-xs text-muted mt-1">{s.label}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="card">
                <h3 className="section-title mb-4">Distribuție verdicte</h3>
                {(stats.verdictCounts || stats.VerdictCounts || []).map(v => (
                  <div key={v.verdict || v.Verdict} className="flex items-center justify-between py-2 border-b border-rose-border last:border-0">
                    <span className={`text-sm font-medium ${VERDICT_COLOR[v.verdict || v.Verdict] || VERDICT_COLOR.default}`}>
                      {v.verdict || v.Verdict || 'Necunoscut'}
                    </span>
                    <span className="font-serif text-xl text-rose-primary">{v.count || v.Count}</span>
                  </div>
                ))}
              </div>

              <div className="card">
                <h3 className="section-title mb-4">Produse cel mai evaluate</h3>
                {(stats.topProducts || stats.TopProducts || []).map((p, i) => (
                  <div key={i} className="flex items-center gap-3 py-2 border-b border-rose-border last:border-0">
                    <div className="font-serif text-lg text-rose-primary w-5">{i + 1}</div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-soft">{p.brand || p.Brand}</div>
                      <div className="text-sm font-medium text-ink truncate">{p.name || p.Name}</div>
                    </div>
                    <div className="text-xs text-muted">{p.count || p.Count}x</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <h3 className="section-title mb-4">Evaluări recente</h3>
              <div className="flex flex-col gap-2">
                {(stats.recentEvaluations || stats.RecentEvaluations || []).map(e => (
                  <div key={e.id || e.Id} className="flex items-center gap-4 py-2 border-b border-rose-border last:border-0">
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-soft">{e.brand || e.Brand}</div>
                      <div className="text-sm font-medium text-ink truncate">{e.name || e.Name}</div>
                    </div>
                    <div className={`text-xs font-medium ${VERDICT_COLOR[e.finalVerdict || e.FinalVerdict] || VERDICT_COLOR.default}`}>
                      {e.finalVerdict || e.FinalVerdict}
                    </div>
                    <div className="text-xs text-muted">
                      {new Date(e.createdAt || e.CreatedAt).toLocaleDateString('ro-RO')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB UTILIZATORI */}
        {activeTab === 'users' && (
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="section-title">Utilizatori ({users.length})</h2>
            </div>
            <div className="card p-0 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-cream-warm border-b border-rose-border">
                  <tr>
                    {['ID', 'Email', 'Rol', 'Tip ten', 'Preocupare', 'Evaluări', 'Înregistrat', 'Acțiuni'].map(h => (
                      <th key={h} className="text-left text-xs font-medium tracking-widest text-soft px-4 py-3">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {users.map(u => (
                    <tr key={u.id} className="border-b border-rose-border hover:bg-cream-warm transition-colors">
                      <td className="px-4 py-3 text-xs text-soft">{u.id}</td>
                      <td className="px-4 py-3 text-sm font-medium text-ink">{u.email}</td>
                      <td className="px-4 py-3">
                        <select
                          value={u.role}
                          onChange={e => handleRoleChange(u.id, e.target.value)}
                          className={`text-xs px-2 py-1 rounded border outline-none cursor-pointer
                            ${u.role === 'Admin'
                              ? 'bg-rose-light border-rose-mid text-rose-dark'
                              : 'bg-gray-50 border-gray-200 text-muted'}`}>
                          <option value="User">User</option>
                          <option value="Admin">Admin</option>
                        </select>
                      </td>
                      <td className="px-4 py-3 text-xs text-muted">{u.skinType}</td>
                      <td className="px-4 py-3 text-xs text-muted">{u.mainConcern}</td>
                      <td className="px-4 py-3 text-xs text-muted">{u.evaluationCount}</td>
                      <td className="px-4 py-3 text-xs text-muted">
                        {new Date(u.createdAt).toLocaleDateString('ro-RO')}
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => { setDeleteModal(u.id); setDeleteType('user') }}
                          className="text-xs text-red-400 hover:text-red-600 cursor-pointer">
                          🗑️ șterge
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB PRODUSE */}
        {activeTab === 'products' && (
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="section-title">Catalog produse ({totalProducts})</h2>
              <button onClick={() => setShowAddProduct(true)} className="btn-primary px-6 py-2 text-xs">
                + adaugă produs
              </button>
            </div>
            <div className="flex gap-3">
              <input
                className="input-field max-w-sm"
                placeholder="Caută brand sau produs..."
                value={search}
                onChange={e => { setSearch(e.target.value); setPage(1) }}
              />
            </div>
            <div className="card p-0 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-cream-warm border-b border-rose-border">
                  <tr>
                    {['ID', 'Brand', 'Nume', 'Preț', 'Rating', 'Recenzii', 'Loves', 'Acțiuni'].map(h => (
                      <th key={h} className="text-left text-xs font-medium tracking-widest text-soft px-4 py-3">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {products.map(p => (
                    <tr key={p.id} className="border-b border-rose-border hover:bg-cream-warm transition-colors">
                      <td className="px-4 py-3 text-xs text-soft">{p.id}</td>
                      <td className="px-4 py-3 text-xs text-soft">{p.brand}</td>
                      <td className="px-4 py-3 text-sm font-medium text-ink max-w-[200px] truncate">{p.name}</td>
                      <td className="px-4 py-3 text-xs text-muted">${p.price}</td>
                      <td className="px-4 py-3 text-xs text-muted">{p.reviewScore}</td>
                      <td className="px-4 py-3 text-xs text-muted">{p.nOfReviews?.toLocaleString()}</td>
                      <td className="px-4 py-3 text-xs text-muted">{p.nOfLoves?.toLocaleString()}</td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => { setDeleteModal(p.id); setDeleteType('product') }}
                          className="text-xs text-red-400 hover:text-red-600 cursor-pointer">
                          🗑️
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center gap-3 justify-center">
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                className="btn-outline px-4 py-2 text-xs disabled:opacity-40">← anterior</button>
              <span className="text-xs text-muted">pagina {page} din {Math.ceil(totalProducts / 20)}</span>
              <button onClick={() => setPage(p => p + 1)} disabled={page >= Math.ceil(totalProducts / 20)}
                className="btn-outline px-4 py-2 text-xs disabled:opacity-40">următor →</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}