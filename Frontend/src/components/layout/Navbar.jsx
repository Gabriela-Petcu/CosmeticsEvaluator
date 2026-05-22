import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

export default function Navbar() {
  const { isAuthenticated, logout, user } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  const isActive = (path) => location.pathname === path

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <nav className="flex items-center justify-between px-9 py-4 border-b border-rose-border bg-cream sticky top-0 z-50">
      <Link to="/" className="font-serif text-xl font-light tracking-widest text-rose-dark">
        Skin<em className="italic text-rose-primary">IQ</em>
      </Link>

      <div className="flex gap-7 text-xs tracking-wide text-muted">
        <Link to="/"
          className={isActive('/') ? 'text-rose-primary font-medium' : 'hover:text-rose-primary transition-colors'}>
          acasă
        </Link>
        {isAuthenticated && (
          <>
            <Link to="/evaluate"
              className={isActive('/evaluate') ? 'text-rose-primary font-medium' : 'hover:text-rose-primary transition-colors'}>
              evaluează
            </Link>
            <Link to="/history"
              className={isActive('/history') ? 'text-rose-primary font-medium' : 'hover:text-rose-primary transition-colors'}>
              istoric
            </Link>
            <Link to="/profile"
              className={isActive('/profile') ? 'text-rose-primary font-medium' : 'hover:text-rose-primary transition-colors'}>
              profilul meu
            </Link>
          </>
        )}
      </div>

      <div className="flex items-center gap-4">
        {isAuthenticated ? (
          <>
            <span className="text-xs text-muted">{user?.email}</span>
            <button onClick={handleLogout}
              className="text-xs tracking-widest px-4 py-2 border border-rose-border rounded text-muted hover:text-rose-primary hover:border-rose-mid transition-colors">
              ieși din cont
            </button>
          </>
        ) : (
          <>
            <Link to="/login"
              className="text-xs tracking-widest text-muted hover:text-rose-primary transition-colors">
              autentificare
            </Link>
            <Link to="/register"
              className="text-xs tracking-widest px-4 py-2 border border-rose-mid rounded text-rose-primary hover:bg-rose-light transition-colors">
              cont nou
            </Link>
          </>
        )}
      </div>
    </nav>
  )
}