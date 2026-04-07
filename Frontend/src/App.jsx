import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Beaker, History, CheckCircle, XCircle, Sparkles, Search } from 'lucide-react';

function App() {
  const [history, setHistory] = useState([]);
  const [products, setProducts] = useState([]); // Lista celor 1784 produse
  const [searchTerm, setSearchTerm] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // 1. Încărcăm istoricul și lista de produse la pornire
  useEffect(() => {
    const init = async () => {
      try {
        const historyRes = await axios.get('http://localhost:5200/Evaluate/history');
        setHistory(historyRes.data);
        
        const productsRes = await axios.get('http://localhost:5200/Evaluate/products');
        setProducts(productsRes.data);
      } catch (err) {
        console.error("Eroare la inițializare", err);
      }
    };
    init();
  }, []);

  // 2. Filtrarea produselor
  const filteredProducts = searchTerm.length > 1 
    ? products.filter(p => 
        p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.brand.toLowerCase().includes(searchTerm.toLowerCase())
      ).slice(0, 10) 
    : [];

  // 3. Evaluarea prin ID (când dai click pe un produs)
  const handleSelectProduct = async (id) => {
    setLoading(true);
    setSearchTerm(""); // Închidem lista
    try {
      const res = await axios.post(`http://localhost:5200/Evaluate/evaluate-by-id/${id}`);
      setResult(res.data);
      // Reîmprospătăm istoricul
      const historyRes = await axios.get('http://localhost:5200/Evaluate/history');
      setHistory(historyRes.data);
    } catch (err) {
      alert("Eroare la evaluare!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'Segoe UI, sans-serif', maxWidth: '900px', margin: '0 auto', backgroundColor: '#fdf2f8', minHeight: '100vh' }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ color: '#be185d', fontSize: '2.5rem' }}>
          <Sparkles /> Cosmetics AI Evaluator
        </h1>
        <p style={{ color: '#86198f' }}>Alege un produs din cele 1784 disponibile pentru analiză</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
        {/* SECȚIUNEA CĂUTARE */}
        <section style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', position: 'relative' }}>
          <h2 style={{ fontSize: '1.2rem', marginBottom: '20px' }}><Search /> Caută Produs</h2>
          <input 
  type="text" 
  placeholder="Scrie brand sau nume (ex: Drunk Elephant)..."
  value={searchTerm}
  onChange={(e) => setSearchTerm(e.target.value)}
  style={{ 
    width: '100%', 
    padding: '12px', 
    borderRadius: '8px', 
    border: '1px solid #ddd', 
    fontSize: '1rem',
    backgroundColor: 'white', // ADAUGĂ ASTA
    color: 'black'            // ADAUGĂ ASTA
  }}
/>
          
          {filteredProducts.length > 0 && (
            <div style={{ position: 'absolute', background: 'white', width: '90%', zIndex: 10, boxShadow: '0 10px 15px rgba(0,0,0,0.1)', borderRadius: '8px', marginTop: '5px', border: '1px solid #eee' }}>
              {filteredProducts.map(p => (
                <div 
                  key={p.id} 
                  onClick={() => handleSelectProduct(p.id)}
                  style={{ padding: '10px', cursor: 'pointer', borderBottom: '1px solid #f9f9f9' }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = '#fce7f3'}
                  onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}
                >
                  <strong>{p.brand}</strong> - {p.name}
                </div>
              ))}
            </div>
          )}
          {loading && <p style={{ marginTop: '10px', color: '#be185d' }}>Se analizează datele din baza de date...</p>}
        </section>

        {/* SECȚIUNEA REZULTAT */}
        <section style={{ background: 'white', padding: '25px', borderRadius: '15px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          {!result ? (
            <p style={{ color: '#9ca3af' }}>Selectează un produs din listă pentru verdict.</p>
          ) : (
            <div>
              <div style={{ fontSize: '3rem', marginBottom: '10px' }}>
                {result.originalResult.ml.merita_ml ? <CheckCircle color="green" size={60} /> : <XCircle color="red" size={60} />}
              </div>
              <h3 style={{ fontSize: '1.5rem', color: '#1f2937' }}>{result.finalVerdict}</h3>
              <p>Siguranță AI: <strong>{(result.originalResult.ml.probability * 100).toFixed(2)}%</strong></p>
            </div>
          )}
        </section>
      </div>

      {/* SECȚIUNEA ISTORIC */}
      <section style={{ marginTop: '40px' }}>
        <h2 style={{ fontSize: '1.3rem', color: '#be185d' }}><History /> Istoric Evaluări Recente</h2>
        <div style={{ background: 'white', borderRadius: '10px', marginTop: '15px', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ background: '#fce7f3' }}>
              <tr>
                <th style={{ padding: '12px', textAlign: 'left' }}>Produs</th>
                <th style={{ padding: '12px', textAlign: 'left' }}>Verdict</th>
                <th style={{ padding: '12px', textAlign: 'left' }}>Dată</th>
              </tr>
            </thead>
            <tbody>
              {history.map(item => (
                <tr key={item.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: '12px' }}>{item.productId}</td>
                  <td style={{ padding: '12px' }}>{item.finalVerdict}</td>
                  <td style={{ padding: '12px', color: '#6b7280' }}>{new Date(item.createdAt).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default App;