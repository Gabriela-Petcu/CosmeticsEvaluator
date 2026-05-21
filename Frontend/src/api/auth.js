import client from './client'

export const register = (email, password, skinType, mainConcern, budgetLevel) =>
  client.post('/auth/register', { email, password })
    .then(async (res) => {
      // După register, setăm profilul
      const loginRes = await login(email, password)
      await updateProfile(skinType, mainConcern, budgetLevel)
      return loginRes
    })

export const login = (email, password) =>
  client.post('/auth/login', { email, password })

export const googleLogin = (idToken) =>
  client.post('/auth/google-login', JSON.stringify(idToken), {
    headers: { 'Content-Type': 'application/json' }
  })

export const getProfile = () =>
  client.get('/auth/profile')

export const updateProfile = (skinType, mainConcern, budgetLevel) =>
  client.put('/auth/profile', {
    skinType,
    mainConcern,
    budgetLevel,
  })