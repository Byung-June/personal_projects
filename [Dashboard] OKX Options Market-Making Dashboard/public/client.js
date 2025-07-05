const socket = io();

// Balance 렌더링
const balanceTbody = document
  .getElementById('balance-table')
  .querySelector('tbody');
function renderBalance(bal) {
  balanceTbody.innerHTML = '';
  Object.entries(bal).forEach(([asset, val]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td style="text-align:left">${asset}</td><td>${val}</td>`;
    balanceTbody.append(tr);
  });
}

// Futures Info 렌더링
const futInfoTbody = document
  .getElementById('fut-info-table')
  .querySelector('tbody');
function renderFutInfo(fut, futInfo) {
  futInfoTbody.innerHTML = '';
  const [avgPx = '-', upl = '-'] = futInfo || [];
  const uplClass = upl > 0 ? 'positive' : (upl < 0 ? 'negative' : '');
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${fut}</td>
    <td>${avgPx}</td>
    <td class="${uplClass}">${upl}</td>
  `;
  futInfoTbody.append(tr);
}

// Option Details 렌더링
const optDetailsTbody = document
  .getElementById('opt-details-table')
  .querySelector('tbody');
function renderOptDetails(opt, delta, optInfo) {
  optDetailsTbody.innerHTML = '';
  Object.keys(opt).sort().forEach(base => {
    const [crossC, isoC, crossP, isoP] = opt[base];
    const [sumΔ = 0, ΔC = 0, ΔP = 0] = delta[base] || [];
    const [avgC = '-', uplC = '-', avgP = '-', uplP = '-'] = optInfo[base] || [];
    const sumClass = sumΔ > 0 ? 'positive' : (sumΔ < 0 ? 'negative' : '');
    const cClass   = ΔC   > 0 ? 'positive' : (ΔC   < 0 ? 'negative' : '');
    const pClass   = ΔP   > 0 ? 'positive' : (ΔP   < 0 ? 'negative' : '');
    const uplCClass= uplC  > 0 ? 'positive' : (uplC  < 0 ? 'negative' : '');
    const uplPClass= uplP  > 0 ? 'positive' : (uplP  < 0 ? 'negative' : '');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="text-align:left">${base}</td>
      <td>${crossC}</td><td>${isoC}</td><td>${crossP}</td><td>${isoP}</td>
      <td class="${sumClass}">${sumΔ}</td>
      <td class="${cClass}">${ΔC}</td>
      <td class="${pClass}">${ΔP}</td>
      <td>${avgC}</td>
      <td class="${uplCClass}">${uplC}</td>
      <td>${avgP}</td>
      <td class="${uplPClass}">${uplP}</td>
    `;
    optDetailsTbody.append(tr);
  });
}

// Open Orders 렌더링
const openOrdersTbody = document
  .getElementById('open-orders-table')
  .querySelector('tbody');
function renderOpenOrders(openOrders) {
  openOrdersTbody.innerHTML = '';
  Object.values(openOrders)
    .flat()
    .forEach(o => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${o.time}</td>
        <td style="text-align:left">${o.instId}</td>
        <td>${o.ordId}</td>
        <td>${o.clOrdId}</td>
        <td>${o.side}</td>
        <td>${o.px}</td>
        <td>${o.sz}</td>
        <td>${o.accFillSz}</td>
        <td>${o.tdMode}</td>
      `;
      openOrdersTbody.append(tr);
    });
}

// 이벤트 바인딩
socket.on('balance',      renderBalance);
socket.on('positions',    ({ opt, fut, delta, optInfo, futInfo }) => {
  renderFutInfo(fut, futInfo);
  renderOptDetails(opt, delta, optInfo);
});
socket.on('openOrders',   renderOpenOrders);