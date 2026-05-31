import client from './client'

export const register = async (email, password, skinType, mainConcern, budgetLevel) => {
  await client.post('/auth/register', { email, password })

  const loginRes = await client.post('/auth/login', { email, password })

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

export const googleLogin = (accessToken) =>
  client.post('/auth/google-login', JSON.stringify(accessToken), {
    headers: { 'Content-Type': 'application/json' }
  })

export const getProfile = () =>
  client.get('/auth/profile')

export const updateProfile = (skinType, mainConcern, budgetLevel) =>
  client.put('/auth/profile', { skinType, mainConcern, budgetLevel })

export const forgotPassword = (email) =>
  client.post('/auth/forgot-password', { email })

export const resetPassword = (token, email, newPassword) =>
  client.post('/auth/reset-password', { token, email, newPassword })