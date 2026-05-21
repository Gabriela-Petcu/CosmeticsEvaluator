import axios from 'axios'

const API_BASE = 'http://localhost:5200'

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

// Atașează token-ul JWT automat la fiecare request
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('skiniq_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Dacă token-ul e expirat, deloghez automat
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('skiniq_token')
      localStorage.removeItem('skiniq_user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default client