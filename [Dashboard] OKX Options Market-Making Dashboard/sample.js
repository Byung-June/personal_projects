const express = require('express');
const http    = require('http');
const os      = require('os');
const { Server } = require('socket.io');

function getLocalExternalIPv4() {
  const nets = os.networkInterfaces();
  for (const name in nets) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return '127.0.0.1';
}

// Express & Socket.IO 초기화
const gApp    = express();
const gServer = http.createServer(gApp);
const gIO     = new Server(gServer);

// static 폴더 설정 (public 내 HTML/CSS/JS 제공)
gApp.use(express.static('public'));

// --------------------------- 글로벌 상태 --------------------------- //
const gBalance = { BTC: 0.04558, USDT: 192107, Equity: 197073 };
let   gPosOpt  = {
  '250711-100000': [0, 3, -3, 0],
  '250704-100800': [-1, 0, 0, 1]
};
let   gPosFut  = 23;
let   gPosOptDelta = {
  '250711-100000': [0,  0.015, -0.015],
  '250704-100800': [-0.002, -0.006, 0.004]
};
let   gPosOptInfo  = {
  '250711-100000': [0.03, 123.45, 0.04, -10.2],
  '250704-100800': [0.006, 0,     0.1,   -0.004]
};
let   gPosFutInfo  = [107680.1, 1.7];

// Open Orders 예시 데이터
let   gOpenOrders  = {
  'BTC-USD-SWAP': [
    { time: '20250703 12:00:00', instId: 'BTC-USD-SWAP', ordId: '1001', clOrdId: 'px0x1', side: 'buy',  px: 109680.9, sz: 7,  accFillSz: 0, tdMode: 'cross' },
    { time: '20250703 12:05:00', instId: 'BTC-USD-SWAP', ordId: '1002', clOrdId: 'px0x2', side: 'sell', px: 109700.5, sz: 5,  accFillSz: 2, tdMode: 'isolated' }
  ],
  'BTC-USD-250711-100000-C': [
    { time: '20250703 12:10:00', instId: 'BTC-USD-250711-100000-C', ordId: '2001', clOrdId: 'px0x3', side: 'buy', px: 0.005, sz: 40, accFillSz: 0, tdMode: 'isolated' }
  ]
};

// --------------------------- 방송 함수 --------------------------- //
function broadcastBalance() {
  gIO.emit('balance', gBalance);
}
function broadcastPositions() {
  gIO.emit('positions', {
    opt:     gPosOpt,
    fut:     gPosFut,
    delta:   gPosOptDelta,
    optInfo: gPosOptInfo,
    futInfo: gPosFutInfo
  });
}
function broadcastOpenOrders() {
  gIO.emit('openOrders', gOpenOrders);
}

// --------------------------- 라우팅 --------------------------- //
gApp.get('/', (req, res) => {
  res.sendFile(__dirname + '/public/index.html');
});

// --------------------------- Socket.IO 연결 --------------------------- //
gIO.on('connection', socket => {
  console.log('Client connected:', socket.id);
  
  // 신규 방식: 함수 호출
  broadcastBalance();
  broadcastPositions();
  broadcastOpenOrders();
});

// --------------------------- 주기적 업데이트 --------------------------- //
setInterval(() => {
  // demo용 랜덤 변화
  gBalance.BTC    = +(gBalance.BTC    + (Math.random() - 0.5) * 0.0001).toFixed(5);
  gBalance.USDT   = +(gBalance.USDT   + (Math.random() - 0.5) * 10).toFixed(0);
  gBalance.Equity = +(gBalance.Equity + (Math.random() - 0.5) * 50).toFixed(0);

  // 실제 fetch 함수를 호출해 데이터를 갱신할 때는 getPositionsH2(), getOpenOrdersH2() 등을 실행
  // getPositionsH2();
  // getOpenOrdersH2();

  // 방송
  broadcastBalance();
  broadcastPositions();
  broadcastOpenOrders();
}, 5000);

// --------------------------- 서버 가동 --------------------------- //
const PORT = process.env.PORT || 3000;
gServer.listen(PORT, '0.0.0.0', () => {
  const ip = getLocalExternalIPv4();
  console.log(`🚀 Server running at: http://localhost:${PORT}  or  http://${ip}:${PORT}`);
});