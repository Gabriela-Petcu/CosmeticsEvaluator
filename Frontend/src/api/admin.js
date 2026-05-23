import client from './client'

export const getAdminStats = () =>
  client.get('/admin/stats')

export const getAdminUsers = () =>
  client.get('/admin/users')

export const updateUserRole = (id, role) =>
  client.put(`/admin/users/${id}/role`, { role })

export const deleteUser = (id) =>
  client.delete(`/admin/users/${id}`)

export const getAdminProducts = (page = 1, search = '') =>
  client.get(`/admin/products?page=${page}&pageSize=20&search=${search}`)

export const addProduct = (product) =>
  client.post('/admin/products', product)

export const updateProduct = (id, product) =>
  client.put(`/admin/products/${id}`, product)

export const deleteProduct = (id) =>
  client.delete(`/admin/products/${id}`)