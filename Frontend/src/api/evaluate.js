import client from './client'

export const getProducts = () =>
  client.get('/evaluate/products')

export const evaluateById = (productId) =>
  client.post(`/evaluate/evaluate-by-id/${productId}`)

export const evaluateManual = (productData) =>
  client.post('/evaluate', {
    product_id: `manual_${Date.now()}`,
    data: productData,
  })

export const getHistory = () =>
  client.get('/evaluate/history')

export const deleteEvaluation = (id) =>
  client.delete(`/evaluations/${id}`)