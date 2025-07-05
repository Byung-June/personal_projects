const WebSocket = require('ws');
const fs = require('fs');
const http2 = require('http2');
const util = require('util');
const crypto = require('crypto');

const datetime = require('node-datetime');
const path = require('path');

const express = require('express');
const http    = require('http');
const { Server } = require('socket.io');
const os = require('os');

// ========================= Logging ========================= //
// datetime.setOffsetInHours(9);
let currentTime = new Date();
let times = datetime.create(currentTime);
let date = times.format('Y-m-d');
let index = 0;
let fileName = `./log_opt_${date}_${index}.txt`;
// Increase index until we find a filename that does not exist
while (fs.existsSync(fileName)) {
    index++;
    fileName = `./log_opt_${date}_${index}.txt`;
}

let logStdout = process.stdout;
console.log = function () {
    let currentTime = new Date();
    let times = datetime.create(currentTime);
    let date = times.format('Y-m-d');
    let time = times.format('H:M:S:N');
    let logFile = fs.createWriteStream(`./log_opt_${date}_${index}.txt`, { flags: 'a' });
    logFile.write(date + ' ' + time + ' : ' + util.format.apply(null, arguments) + '\r\n');
    logStdout.write(time + ' : ' + util.format.apply(null, arguments) + '\r\n');
    logFile.end();
};

// ========================= App ========================= //

// ── Express + Socket.IO 세팅 ──
const gApp = express();
const gServer = http.createServer(gApp);
const gIO = new Server(gServer);

//// ========================= Global Variables ========================= ////

// --------------------------- Global Variables for Private --------------------------- //
const keysPath = path.join(__dirname, 'keys.txt');
let keys = {};

// You can parse the file synchronously at startup.
try {
    const data = fs.readFileSync(keysPath, 'utf8');
    keys = JSON.parse(data);
} catch (error) {
    console.error("Error loading keys:", error);
}

const API_KEY = keys.API_KEY;
const API_SECRET = keys.API_SECRET;
const PASSPHRASE = keys.PASSPHRASE;

let gClient;
let gClientReconnect = true;  // true일 때만 재접속 시도
let gClientPing;

// --------------------------- Global Variables for Private --------------------------- //
const gBalance = {
    'BTC': null,
    'USDT': null,
    'Equity': null,
}

let gPosOpt = {};
let gPosFut = 0;
let gPosOptDelta = {};

let gPosOptInfo = {};
let gPosFutInfo = {};

let gOpenOrders = {};


// --------------------------- Global Variables for Fee/Threshold --------------------------- //




// --------------------------- Process --------------------------- //
process.env.TZ = 'Asia/Seoul';

process.on('uncaughtException', (err, origin) => {
    if (err.code === 'ECONNRESET') {
        console.log('connection reset', err);
    } else {
        console.log('unknown error: ', err, origin);
    }
});

process.on('beforeExit', () => {
    gClient.close();
});

process.on('SIGINT', () => {
    console.log('🛑 Exiting, will not reconnect.');
    gClientReconnect = false;
    gClient && gClient.close();
    process.exit();
});

// ========================= Dashboard Helper Functions ========================= //

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





// ========================= Helper Functions ========================= //
function getSign(timestamp, method, requestPath, body = '') {
  const prehash = timestamp + method + requestPath + body;
  return crypto.createHmac('sha256', API_SECRET)
               .update(prehash)
               .digest('base64');
}

function getTimestamp() {
  return new Date().toISOString();
}

/**
 * Remove any keys of the form "YYMMDD-..." 
 * whose date portion is before today.
 *
 * @param {Object} bucketObj — e.g. gOpenPos or gOpenPosDelta
 */
function pruneExpiredBuckets(bucketObj) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  Object.keys(bucketObj)
    .filter(key => /^\d{6}-/.test(key))    // only things like "230628-100000"
    .forEach(key => {
      const dateStr = key.split('-')[0];   // "230628"
      const yy = 2000 + Number(dateStr.slice(0, 2));
      const mm = Number(dateStr.slice(2, 4)) - 1; 
      const dd = Number(dateStr.slice(4, 6));
      const expiry = new Date(yy, mm, dd);

      if (expiry < today) {
        delete bucketObj[key];
      }
    });
}

/**
 * base: "YYMMDD-..." 형식 문자열 (예: "230628-100000")
 * nowMs: 현재 시각의 밀리초 타임스탬프
 *
 * 반환값:
 *  - nowMs ≤ startMs      → 1
 *  - nowMs ≥ startMs + 1h → 0
 *  - 그 외(16:00~17:00 사이) → 1에서 0으로 선형 보간된 값
 */
function calcMultiplierFromBase(base, nowMs) {
  // YYYY-MM-DD 파싱
  const [dateStr] = base.split('-');
  const year  = 2000 + Number(dateStr.slice(0, 2));
  const month = Number(dateStr.slice(2, 4)) - 1; // JS Date month는 0~11
  const day   = Number(dateStr.slice(4, 6));

  // 기준 시각: 해당 날짜 16:00
  const startMs  = new Date(year, month, day, 16, 0, 0).getTime();
  // 만료 시각: 해당 날짜 17:00
  const expireMs = startMs + 60 * 60 * 1000;

  // nowMs 대비 경과 시간(ms), 0~1h 범위로 클램핑
  const elapsed = Math.min(Math.max(nowMs - startMs, 0), expireMs - startMs);

  // 1에서 0으로 선형 보간
  return 1 - elapsed / (expireMs - startMs);
}
    

// ========================= H2 Functions ========================= //
function connect() {
    gClient = http2.connect('https://www.okx.com');

    gClient.on('connect', () => {
        console.log('✅ HTTP/2 session established');
        // ping 대신 더미 request로 goaway 방지
        gClientPing = setInterval(() => {
            if (gClient && !gClient.destroyed) {
                const req = gClient.request({
                ':method': 'GET',
                ':path': '/api/v5/system/time'
                });
                req.on('response', () => req.close());
                req.on('error', () => {/* 무시 */});
                req.end();
            }
        }, 4000);
    });

    gClient.on('error', err => {
        console.log('HTTP2 Session Error:', err);
        clearInterval(gClientPing);
        gClient.close();                  // 기존 세션 닫고
    });

    // 서버가 GOAWAY를 보내는 경우에도 재접속  
    gClient.on('goaway', (errorCode, lastStreamID, opaqueData) => {
        console.log('HTTP2 Goaway received:', { errorCode, lastStreamID });
        clearInterval(gClientPing);
        gClient.close();
    });

    gClient.on('close', () => {
        console.log('🔒 HTTP/2 session closed');
        clearInterval(gClientPing);
        if (gClientReconnect) {
            console.log('⏳ Reconnecting in 1s…');
            setTimeout(connect, 1000);
        }
    });

    return gClient;
}


function getBalanceH2(ccy) {
    const basePath = '/api/v5/account/balance';
    const requestPath = ccy
        ? `${basePath}?ccy=${encodeURIComponent(ccy)}`
        : basePath;

    const method    = 'GET';
    const timestamp = getTimestamp();
    const body      = '';  // GET 이므로 빈 문자열
    const sign      = getSign(timestamp, method, requestPath, body);

    // 3) HTTP/2 전용 pseudo-header 와 일반 헤더 섞어서 설정
    const headers = {
        ':method':    method,
        ':scheme':    'https',
        ':authority': 'www.okx.com',
        ':path':      requestPath,
        'content-type':         'application/json',
        'OK-ACCESS-KEY':        API_KEY,
        'OK-ACCESS-SIGN':       sign,
        'OK-ACCESS-TIMESTAMP':  timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE
    };

    // 4) 세션에서 스트림(request) 생성
    const req = gClient.request(headers);

    let data = '';
    req.setEncoding('utf8');
    req.on('data', chunk => { data += chunk; });
    req.on('end', () => {
        try {
            const json = JSON.parse(data);
            console.log("Get Balance Response:", json.data);

            const details = json.data[0].details;
            const btcDetail = details.find(d => d.ccy === 'BTC');
            const currBTC = btcDetail ? Number(btcDetail.eq) : 0;
            
            const usdtDetail = details.find(d => d.ccy === 'USDT');
            const currUSDT = usdtDetail ? Number(usdtDetail.eq) : 0;

            gBalance['BTC'] = currBTC
            gBalance['USDT'] = currUSDT
            gBalance['Equity'] = json.data[0].totalEq
            
            console.log("gBalance: ", gBalance)
        } catch (err) {
        console.error("Error parsing response:", err);
        }
    });
    req.on('error', err => {
        console.error("Request stream error:", err);
    });
    req.end();  // GET 이므로 본문 없이 바로 종료
}


function getPositionsH2(instType = '', instId = '') {
    // 1) 쿼리 스트링 빌드
    let query = '';
    if (instType || instId) {
        query = '?';
        if (instType) query += `instType=${encodeURIComponent(instType)}`;
        if (instId)    query += `${instType ? '&' : ''}instId=${encodeURIComponent(instId)}`;
    }

    const requestPath = '/api/v5/account/positions' + query;
    const method      = 'GET';
    const timestamp   = getTimestamp();
    const body        = '';  // GET 요청이므로 빈 문자열
    const sign        = getSign(timestamp, method, requestPath, body);

    // 2) HTTP/2 pseudo-header + 일반 헤더 설정
    const headers = {
        ':method':    method,
        ':scheme':    'https',
        ':authority': 'www.okx.com',
        ':path':      requestPath,
        'content-type':         'application/json',
        'OK-ACCESS-KEY':        API_KEY,
        'OK-ACCESS-SIGN':       sign,
        'OK-ACCESS-TIMESTAMP':  timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE
    };

    // 3) 스트림(request) 생성
    const req = gClient.request(headers);

    // 디버깅: 상태 코드 & 콘텐츠 타입 로그
    req.on('response', (resHeaders) => {
        console.log('getOpenOrders status:', resHeaders[':status']);
        // console.log('getOpenOrders content-type:', resHeaders['content-type']);
        if (resHeaders[':status'] != 200) {
            gClient.close();
        }
    });

    let data = '';
    req.setEncoding('utf8');
    req.on('data', chunk => { data += chunk; });

    req.on('end', () => {
        try {
            const json = JSON.parse(data);
            // console.log(`Get Positions Response (${instType || 'all'}):`, json.data);

            if (json.code !== '0' || !Array.isArray(json.data)) {
                console.log('Unexpected positions response:', json);
                return;
            }

            const posOpt = {};      // OPTION 포지션 임시 저장소
            let posFut = 0;         // SWAP 포지션
            const posOptDelta = {};  // OPTION 포지션 delta 임시 저장소
            const posOptInfo = {};  // OPTION 포지션 info 임시 저장소
            let posFutInfo = null;  // SWAP 포지션 info
            const nowMs = Date.now();

            json.data.forEach(pos => {
                const instId = pos.instId;

                if (pos.instType === 'OPTION') {
                    const base = instId.substring(8, instId.length - 2);
                    const multiplier = calcMultiplierFromBase(base, nowMs);
                    if (multiplier <= 0) {
                        delete posOpt[base]; 
                        delete posOptDelta[base]; 
                        delete posOptInfo[base]; 
                        return; // 다음 loop로 패스
                    }

                    // 해당 base 초기화
                    if (!posOpt[base]) {
                        posOpt[base] = [0, 0, 0, 0]; // [crossCall, isolatedCall, crossPut, isolatedPut]
                        posOptDelta[base] = [0, 0, 0]; // [sum, posDeltaCall, posDeltaPut]
                        posOptInfo[base] = [null, null, null, null] // [avgPriceCall, uplCall, avgPricePut, uplPut]
                    }

                    if (pos.mgnMode === 'isolated') {
                        if (instId.endsWith('C')) {
                            posOpt[base][1] = +pos.pos;  // isolated call
                            posOptDelta[base][0] += +pos.deltaBS * multiplier
                            posOptDelta[base][1] += +pos.deltaBS * multiplier
                            posOptInfo[base][0] = +pos.avgPx
                            posOptInfo[base][1] = +pos.upl
                        } else {
                            posOpt[base][3] = +pos.pos;  // isolated put
                            posOptDelta[base][0] += +pos.deltaBS * multiplier
                            posOptDelta[base][2] += +pos.deltaBS * multiplier
                            posOptInfo[base][2] = +pos.avgPx
                            posOptInfo[base][3] = +pos.upl
                        }
                    } else {
                        if (instId.endsWith('C')) {
                            posOpt[base][0] = +pos.pos;  // cross call
                            posOptDelta[base][0] += +pos.deltaBS * multiplier
                            posOptDelta[base][1] += +pos.deltaBS * multiplier
                            posOptInfo[base][0] = +pos.avgPx
                            posOptInfo[base][1] = +pos.upl
                        } else {
                            posOpt[base][2] = +pos.pos;  // cross put
                            posOptDelta[base][0] += +pos.deltaBS * multiplier
                            posOptDelta[base][2] += +pos.deltaBS * multiplier
                            posOptInfo[base][2] = +pos.avgPx
                            posOptInfo[base][3] = +pos.upl
                        }
                    }

                } else if (pos.instType === 'SWAP') {
                    posFut += +pos.pos;
                    posFutInfo = [pos.avgPx, pos.upl]
                }
            });
            gPosOpt = posOpt;
            gPosFut = posFut;
            gPosOptDelta = posOptDelta;
            gPosOptInfo = posOptInfo;
            gPosFutInfo = posFutInfo;
            
            console.log('gPosOpt:', gPosOpt);
            console.log('gPosFut:', gPosFut);
            console.log('gPosOptDelta:', gPosOptDelta);
            console.log('gPosOptInfo:', gPosOptInfo);
            console.log('gPosFutInfo:', gPosFutInfo);

        } catch (err) {
            console.error("Error parsing getPositions response:", err);
        }
    });

    req.on('error', err => {
        console.error("Get Positions request error:", err);
    });

    // 4) 스트림 종료 (GET 이므로 본문 없음)
    req.end();
}


function getOpenOrdersH2(instType = '', instId = '') {
    // 1) 쿼리 스트링 빌드
    let query = '';
    if (instType || instId) {
        query = '?';
        if (instType) query += `instType=${encodeURIComponent(instType)}`;
        if (instId)    query += `${instType ? '&' : ''}instId=${encodeURIComponent(instId)}`;
    }

    const requestPath = '/api/v5/trade/orders-pending' + query;
    const method      = 'GET';
    const timestamp   = getTimestamp();
    const body        = '';  // GET 요청이므로 빈 문자열
    const sign        = getSign(timestamp, method, requestPath, body);

    // 2) HTTP/2 pseudo-header + 일반 헤더 설정
    const headers = {
        ':method':    method,
        ':scheme':    'https',
        ':authority': 'www.okx.com',
        ':path':      requestPath,
        'content-type':         'application/json',
        'OK-ACCESS-KEY':        API_KEY,
        'OK-ACCESS-SIGN':       sign,
        'OK-ACCESS-TIMESTAMP':  timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE
    };

    // 3) 스트림(request) 생성
    const req = gClient.request(headers);

    // 디버깅: 상태 코드 & 콘텐츠 타입 로그
    req.on('response', (resHeaders) => {
        console.log('getOpenOrders status:', resHeaders[':status']);
        // console.log('getOpenOrders content-type:', resHeaders['content-type']);
        if (resHeaders[':status'] != 200) {
            gClient.close();
        }
    });

    let data = '';
    req.setEncoding('utf8');
    req.on('data', chunk => { data += chunk; });

    req.on('end', () => {
        try {
            const json = JSON.parse(data);
            console.log(`Get OpenOrders Response (${instType || 'all'}):`, json.data);

            if (json.code !== '0' || !Array.isArray(json.data)) {
                console.log('Unexpected OpenOrders response:', json);
                return;
            }

            const openOrders = {};      // open Orders 임시 저장소
            json.data.forEach(order => {
                    const {
                    uTime: time,
                    instId,
                    ordId,
                    clOrdId,
                    side,
                    px,
                    sz,
                    accFillSz,
                    tdMode
                } = order;

                if (!openOrders[instId]) openOrders[instId] = [];
                openOrders[instId].push({ time, instId, ordId, clOrdId, side, px, sz, accFillSz, tdMode });
            });
            gOpenOrders = openOrders;
            
            console.log('gOpenOrders:', gOpenOrders);
        } catch (err) {
            console.error("Error parsing getOpenOrders response:", err);
        }
    });

    req.on('error', err => {
        console.error("Get OpenOrders request error:", err);
    });

    // 4) 스트림 종료 (GET 이므로 본문 없음)
    req.end();
}

// ========================= Execution ========================= //

connect()

function main() {
    // 정적 파일 (public/index.html 등) 제공
    gApp.use(express.static('public'));

    // --------------------------- Socket.IO 연결 --------------------------- //
    gIO.on('connection', socket => {
        console.log('Client connected:', socket.id);
        broadcastBalance();
        broadcastPositions();
        broadcastOpenOrders();
    });

    // 2초마다 OKX fetch → 상태 갱신 후 이벤트 방출
    setInterval(async () => {
        getBalanceH2();
        getPositionsH2();
        getOpenOrdersH2();

        broadcastBalance();
        broadcastPositions();
        broadcastOpenOrders();
    }, 2000);

    // 서버 시작
    const PORT = process.env.PORT || 3000;
    gServer.listen(PORT, '0.0.0.0', () => {
        const ip = getLocalExternalIPv4();
        console.log(`🚀 Server running at: http://localhost:${PORT}  or  http://${ip}:${PORT}`);
    });
}

setTimeout(()=>{
    main()
}, 1000)


    