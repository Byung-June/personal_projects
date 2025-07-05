# OKX Options Market-Making Dashboard

A real-time dashboard for tracking OKX account balances, option positions, and open orders, built with Node.js, HTTP/2, and Socket.IO.

---

## 📋 Overview

This project provides a web-based monitor for your OKX options market-making activities. It fetches account data directly from OKX using HTTP/2, processes balances, option chain positions (including delta exposures), and open orders, and broadcasts updates to a client frontend in real time via Socket.IO.

Key capabilities:

- **Balances**: Display BTC, USDT, and total equity.
- **Futures & Option Positions**: Show futures PnL and detailed option positions by base expiry with delta calculations.
- **Open Orders**: List all pending orders with timestamp, instrument, side, price, size, and margin mode.
- **Delta Adjustment**: Automatically scale option deltas based on an expiry-based time-decay multiplier.

---

### 🧰 Tech Stack

- **Runtime**: Node.js (v14+)
- **Server**: Express
- **Real-Time**: Socket.IO
- **HTTP**: HTTP/2 client (`http2` module)
- **Frontend**: HTML, CSS (Grid Layout), Vanilla JavaScript
- **Styling**: CSS3 (Flexbox & Grid)
- **Logging**: Built-in `fs` module for file-based logs
- **Package Management**: npm
- **Version Control**: Git

---

## ⚙️ Features

- **HTTP/2 API Integration**: Connects to OKX's HTTP/2 API for low-latency data retrieval.
- **Real-Time Updates**: Uses Socket.IO to push data to the browser every 2 seconds.
- **Delta Decay Function**: Adjusts option deltas linearly from 16:00 to 17:00 on expiry day.
- **Auto-Pruning**: Removes expired option buckets after their expiry date.
- **Responsive Frontend**: Simple HTML/CSS grid layout for clear visualization.

---

## 🚀 Getting Started

### Prerequisites

- Node.js (v14+)
- npm
- OKX API credentials (API Key, Secret, Passphrase)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/youruser/okx-opt-mm-dashboard.git
   cd okx-opt-mm-dashboard
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Configure API keys** Create a `keys.txt` file in the project root with your credentials in JSON format:

   ```json
   {
     "API_KEY": "your_api_key",
     "API_SECRET": "your_api_secret",
     "PASSPHRASE": "your_passphrase"
   }
   ```

4. **Run the dashboard**

   ```bash
   node dashboard.js
   ```

5. **Open in browser** Navigate to `http://localhost:3000` to view the dashboard.

---

## 📁 Project Structure

```
├── dashboard.js       # Main server: HTTP/2 client & Socket.IO broadcaster
├── sample.js          # Demo broadcaster with random data
├── public/            # Static assets
│   ├── index.html     # Dashboard markup
│   ├── style.css      # Layout and styles
│   └── client.js      # Served to browser for rendering data
├── keys.txt           # JSON file with OKX API credentials (not committed)
├── package.json       # npm dependencies and scripts
└── README.md          # Project documentation
```

---

## 🛠 How It Works

1. **Connection Setup** (`dashboard.js`)

   - Establishes an HTTP/2 session to `https://www.okx.com`.
   - Sets up Express server to serve static files from `/public`.
   - Initializes Socket.IO for real-time communication.

2. **Data Fetching**

   - **Balances**: `getBalanceH2()` retrieves `/api/v5/account/balance` and updates `gBalance`.
   - **Positions**: `getPositionsH2()` retrieves `/api/v5/account/positions`, categorizes by `OPTION` vs `SWAP`, computes per-expiry delta exposures, average prices, and unrealized PnL.
   - **Open Orders**: `getOpenOrdersH2()` retrieves `/api/v5/trade/orders-pending` and stores into `gOpenOrders`.

3. **Delta Decay & Pruning**

   - ``: For each option expiry bucket (YYMMDD-XXX), linearly decays delta from 1→0 between 16:00 and 17:00 local time on expiry day.
   - ``: Automatically removes any expiry buckets older than today.

4. **Broadcasting**

   - Every 2 seconds, the server calls all fetch functions and emits updates via Socket.IO events: `balance`, `positions`, `openOrders`.
   - The browser client listens for these events and updates HTML tables accordingly.

---

## 🎨 Frontend Layout

- **Grid-Based Dashboard** (`style.css`): Arranges four main cards:

  1. Balance & Greeks (top-left)
  2. Statistics (top-right)
  3. Positions (bottom-left)
     - Futures summary
     - Options detail grid
  4. Open Orders (bottom-right)

- **Color Coding**: Positive values in green (`.positive`), negative in red (`.negative`).

---

## 🔧 Customization

- **Update Interval**: Modify the `setInterval` duration in `dashboard.js` (default: 2000ms).
- **Logging**: Logs are written to `log_opt_YYYY-MM-DD_N.txt` in the project root.
- **Styling**: Adjust `style.css` to change layout, colors, or fonts.

---

## 📜 License

This project is licensed under the MIT License.

---

## ✍️ Contributing

Contributions are welcome! Please open issues or pull requests on GitHub.

---

*Last updated: 2025-07-05*

