import client from './client'

export const getProducts = () =>
  client.get('/evaluate/products')

export const evaluateById = async (productId) => {
  const res = await client.post(`/evaluate/evaluate-by-id/${productId}`)
  const data = res.data
  const result = data.originalResult || data.OriginalResult || data

  const productInfo = data.productInfo || data.ProductInfo || {}
  result.productName = productInfo.name || productInfo.Name || result.productId || 'Produs evaluat'
  result.brand = productInfo.brand || productInfo.Brand || ''
  result.price = productInfo.price || productInfo.Price || null

  return { data: result }
}

export const evaluateManual = async (productData) => {
  const res = await client.post('/evaluate', {
    product_id: `manual_${Date.now()}`,
    data: productData,
  })
  const data = res.data
  const result = data.originalResult || data.OriginalResult || data
  return { data: result }
}

export const getHistory = () =>
  client.get('/evaluate/history')

export const deleteEvaluation = (id) =>
  client.delete(`/evaluations/${id}`)