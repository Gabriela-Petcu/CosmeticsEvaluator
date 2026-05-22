import client from './client'

export const register = async (email, password, skinType, mainConcern, budgetLevel) => {
  // 1. Creăm contul
  await client.post('/auth/register', { email, password })

  // 2. Logăm imediat
  const loginRes = await client.post('/auth/login', { email, password })

  // 3. Setăm profilul de ten — token-ul e deja în localStorage via AuthContext
  const token = loginRes.data.token
  await client.put('/auth/profile', {
    skinType,
    mainConcern,
    budgetLevel,
  }, {
    headers: { Authorization: `Bearer ${token}` }
  })

  return loginRes
}

export const login = (email, password) =>
  client.post('/auth/login', { email, password })

export const googleLogin = (idToken) =>
  client.post('/auth/google-login', JSON.stringify(idToken), {
    headers: { 'Content-Type': 'application/json' }
  })

export const getProfile = () =>
  client.get('/auth/profile')

export const updateProfile = (skinType, mainConcern, budgetLevel) =>
  client.put('/auth/profile', { skinType, mainConcern, budgetLevel })