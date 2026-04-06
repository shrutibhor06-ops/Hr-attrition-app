document.addEventListener('DOMContentLoaded', () => {
  initAuth();
  
  // Set up export CSV handler
  document.getElementById('exportCsvBtn').addEventListener('click', () => {
    const query = document.getElementById('searchInput').value;
    window.location.href = `/api/export?search=${encodeURIComponent(query)}`;
  });

  // Set up search with debouncing
  let debounceTimeout;
  document.getElementById('searchInput').addEventListener('input', (e) => {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      loadEmployeeData(e.target.value);
    }, 400);
  });
  
  document.getElementById('logoutBtn').addEventListener('click', async () => {
    await fetch('/api/logout', { method: 'POST' });
    window.location.reload();
  });
  
  document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    btn.innerText = 'Verifying...';
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        username: document.getElementById('loginUser').value,
        password: document.getElementById('loginPass').value
      })
    });
    if (res.ok) {
        window.location.reload();
    } else {
        document.getElementById('loginError').style.display = 'block';
        btn.innerText = 'Sign In';
    }
  });
});

async function initAuth() {
  const res = await fetch('/api/me');
  const session = await res.json();
  if (session.logged_in) {
    document.getElementById('loginOverlay').style.display = 'none';
    initNavigation();
    loadDashboardData();
    loadEmployeeData();
    generatePredictionForm();
    loadModelInsights();
    loadHistory();
    loadAdmin();
  }
}

function initNavigation() {
  const items = document.querySelectorAll('.nav-item');
  const sections = document.querySelectorAll('.section');
  
  items.forEach(item => {
    item.addEventListener('click', () => {
      items.forEach(nav => nav.classList.remove('active'));
      sections.forEach(sec => sec.classList.remove('active'));
      
      item.classList.add('active');
      const target = item.getAttribute('data-target');
      document.getElementById(target).classList.add('active');
    });
  });
}

async function loadDashboardData() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    
    // Animate numbers
    document.getElementById('stat-total').innerText = data.totalEmployees || 0;
    document.getElementById('stat-attrition-count').innerText = data.attritionCount || 0;
    document.getElementById('stat-attrition-rate').innerText = (data.attritionRate || 0) + '%';
    
    if (data.departmentDistribution) {
      renderChart(data.departmentDistribution);
    }
  } catch (err) {
    console.error("Failed to load stats", err);
  }
}

function renderChart(deptData) {
  const ctx = document.getElementById('attritionChart').getContext('2d');
  const labels = Object.keys(deptData);
  const values = Object.values(deptData);
  
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Employees by Department',
        data: values,
        backgroundColor: 'rgba(59, 130, 246, 0.4)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#f8fafc' } }
      },
      scales: {
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
      }
    }
  });
}

async function loadEmployeeData(search = '') {
  try {
    const res = await fetch(`/api/employees?search=${encodeURIComponent(search)}`);
    const data = await res.json();
    
    const thead = document.getElementById('tableHeader');
    const tbody = document.getElementById('tableBody');
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    if (data.length === 0) return;
    
    // Generate headers based on first object keys
    const priorityCols = ['EmployeeNumber', 'Age', 'Department', 'JobRole', 'Attrition'];
    const otherCols = Object.keys(data[0]).filter(k => !priorityCols.includes(k));
    const allCols = [...priorityCols.filter(k => Object.hasOwn(data[0], k)), ...otherCols];
    
    allCols.slice(0, 8).forEach(col => { // Show max 8 columns for UI neatness
      const th = document.createElement('th');
      th.innerText = col.replace(/([A-Z])/g, ' $1').trim(); // split camel case
      thead.appendChild(th);
    });
    
    // Generate rows
    data.slice(0, 100).forEach(row => { // Render top 100 for performance
      const tr = document.createElement('tr');
      allCols.slice(0, 8).forEach(col => {
        const td = document.createElement('td');
        let val = row[col];
        if (col === 'Attrition' && val) {
          const cls = val.toLowerCase() === 'yes' || val === 1 ? 'badge-yes' : 'badge-no';
          td.innerHTML = `<span class="badge ${cls}">${val}</span>`;
        } else {
          td.innerText = val !== null ? val : '-';
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load employees", err);
  }
}

async function loadModelInsights() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    if (!data.models) return;
    
    // Model Comparison Chart
    const mcCtx = document.getElementById('modelComparisonChart').getContext('2d');
    new Chart(mcCtx, {
      type: 'bar',
      data: {
        labels: data.models.map(m => m.name),
        datasets: [
          { label: 'Accuracy', data: data.models.map(m => m.accuracy), backgroundColor: '#3b82f6' },
          { label: 'F1 Score', data: data.models.map(m => m.f1_score), backgroundColor: '#8b5cf6' }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#f8fafc' } } }, scales: { y: { ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } } }
    });

    // Feature Importances Chart
    if (data.feature_importances) {
      const fCtx = document.getElementById('featureChart').getContext('2d');
      new Chart(fCtx, {
        type: 'bar',
        data: {
          labels: Object.keys(data.feature_importances).map(lbl => lbl.substring(0, 15)),
          datasets: [{
            label: 'Importance Weighting',
            data: Object.values(data.feature_importances),
            backgroundColor: 'rgba(16, 185, 129, 0.6)'
          }]
        },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#f8fafc' } } }, scales: { y: { ticks: { color: '#94a3b8', font: {size: 10} } }, x: { ticks: { color: '#94a3b8' } } } }
      });
    }
  } catch(e) { console.error('Failed to load model insights', e); }
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    const tbody = document.getElementById('historyBody');
    tbody.innerHTML = '';
    data.forEach(row => {
      const tr = document.createElement('tr');
      const cls = row.prediction.toLowerCase().includes('high') ? 'danger-color' : 'success-color';
      
      const details = document.createElement('td');
      details.style.maxWidth = '300px';
      details.style.overflow = 'hidden';
      details.style.textOverflow = 'ellipsis';
      details.innerText = row.inputs;
      
      tr.innerHTML = `
        <td>${row.timestamp}<br><span style="color:var(--text-secondary);font-size:0.8rem;">by @${row.username}</span></td>
        <td style="color:var(--${cls}); font-weight:600;">${row.prediction}</td>
        <td>${row.confidence ? row.confidence.toFixed(1) + '%' : '-'}</td>
      `;
      tr.appendChild(details);
      tbody.appendChild(tr);
    });
  } catch(e) { console.error('Failed to load history', e); }
}

async function loadAdmin() {
  try {
    const res = await fetch('/api/users');
    const data = await res.json();
    const tbody = document.getElementById('adminBody');
    tbody.innerHTML = '';
    data.forEach(row => {
      const tr = document.createElement('tr');
      const statColor = row.is_logged_in ? 'success-color' : 'text-secondary';
      const statText = row.is_logged_in ? 'Online' : 'Offline';
      tr.innerHTML = `
        <td style="font-weight: 600;">@${row.username}</td>
        <td><span style="color: var(--${statColor});"><i class="ph ph-circle"></i> ${statText}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch(e) { console.error('Failed to load admin', e); }
}

async function generatePredictionForm() {
  try {
    const res = await fetch('/api/features');
    const schema = await res.json();
    const container = document.getElementById('dynamicFormContainer');
    container.innerHTML = '';
    
    Object.keys(schema).forEach(key => {
      const field = schema[key];
      const div = document.createElement('div');
      div.className = 'form-group';
      
      const label = document.createElement('label');
      label.innerText = key.replace(/([A-Z])/g, ' $1').trim();
      div.appendChild(label);
      
      if (field.type === 'categorical') {
        const select = document.createElement('select');
        select.className = 'form-control';
        select.name = key;
        select.required = true;
        field.options.forEach(opt => {
          const option = document.createElement('option');
          option.value = opt;
          option.innerText = opt;
          select.appendChild(option);
        });
        div.appendChild(select);
      } else {
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.className = 'form-control';
        input.name = key;
        input.required = true;
        input.placeholder = `avg: ${Math.round(field.mean * 10) / 10}`;
        input.value = Math.round(field.mean * 10) / 10;
        div.appendChild(input);
      }
      container.appendChild(div);
    });
  } catch (err) {
    console.error("Failed to load features for predictor component", err);
  }
}

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const jsonPayload = {};
  formData.forEach((val, key) => {
    // If it looks like a number, parse it
    jsonPayload[key] = isNaN(val) ? val : parseFloat(val);
  });
  
  const predictBtn = document.getElementById('predictBtn');
  const resultDiv = document.getElementById('predictionResult');
  const resultText = document.getElementById('resultText');
  const resultProb = document.getElementById('resultProb');
  
  predictBtn.classList.add('loading');
  resultDiv.style.display = 'none';
  resultDiv.className = 'prediction-result'; // reset class
  
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(jsonPayload)
    });
    
    const data = await res.json();
    resultDiv.style.display = 'block';
    
    if (data.error) {
      resultText.innerText = "Error predicting risk";
      resultProb.innerText = data.error;
    } else {
      resultText.innerText = data.prediction;
      if (data.prediction.toLowerCase().includes('high')) {
        resultDiv.classList.add('high-risk');
        resultText.innerHTML = '<i class="ph ph-warning"></i> High Flight Risk';
      } else {
        resultDiv.classList.add('low-risk');
        resultText.innerHTML = '<i class="ph ph-check-circle"></i> Low Flight Risk';
      }
      
      if (data.probabilities && data.probabilities.length > 0) {
        // Find max prob
        const confidence = Math.max(...data.probabilities) * 100;
        resultProb.innerText = `Confidence: ${confidence.toFixed(1)}%`;
      }
    }
  } catch (err) {
    console.error("Prediction failed", err);
  } finally {
    predictBtn.classList.remove('loading');
  }
});
