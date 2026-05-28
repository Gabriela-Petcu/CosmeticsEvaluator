import client from './client'

export const getProducts = () =>
  client.get('/evaluate/products')

export const evaluateById = async (productId) => {
  const res = await client.post(`/evaluate/evaluate-by-id/${productId}`)
  const data = res.data

  const result = data.originalResult || data.OriginalResult || data
  const productInfo = data.productInfo || data.ProductInfo || {}

  // Îmbogățim result cu info produs
  result.productName = productInfo.name || productInfo.Name || result.productName || 'Produs evaluat'
  result.brand = productInfo.brand || productInfo.Brand || result.brand || ''
  result.price = productInfo.price || productInfo.Price || result.price || null

  // Păstrăm și productInfo separat pentru ResultPage
  result.productInfo = productInfo

  return { data: result }
}


export const getHistory = () =>
  client.get('/evaluate/history')

export const deleteEvaluation = (id) =>
  client.delete(`/evaluations/${id}`)