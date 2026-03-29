# 🦢 Black Swan Agent
### Adversarial Robustness Testing for Algorithmic Trading Systems

**Black Swan** is a headless, institutional-grade adversarial testing agent for algorithmic trading strategies.

It does not predict markets.  
It **attempts to break your strategy**.

By combining:
- Walk-Forward Optimization (WFO)
- Monte Carlo adversarial simulations
- Strict anti-lookahead backtesting
- High-performance Numba-accelerated JIT kernels

Black Swan produces **mathematically defensible robustness reports**, not heuristic opinions.

> If your strategy survives Black Swan, it’s worth considering. If it doesn’t, it never was.

---

Black Swan is not a backtesting tool.  
It is a **strategy adversary**.

---

## 🎯 Why Black Swan Exists

Most retail and even semi-professional backtesting systems suffer from:
- **Lookahead bias**
- **Overfitting** via static optimization over the entire dataset
- **Ignoring execution uncertainty** (assuming perfect fills)
- **Ignoring tax drag** (which can destroy many "profitable" strategies)

Black Swan addresses these by:
- Enforcing strict `t+1` execution semantics.
- Using continuous Walk-Forward Optimization instead of global curve-fitting.
- Stress-testing strategies across multiple adversarial Monte Carlo regimes.
- **Accurate execution modeling:** Native support for per-trade fees and slippage.
- **Tax-Aware Outcomes:** Real-world capital gains tracking (STCG/LTCG) for US, UK, and India.

---

## 🏗 Architecture Overview

```text
User JSON-RPC Request
          ↓
A2A HTTP Server (__main__.py)
          ↓
OpenAI Agent Executor (with session memory)
          ↓
LLM Confirmation Flow (Fees/Slippage/Tax)
          ↓
QuantToolset (Numba JIT Engine)
          ↓
[WFO + Monte Carlo + Tax Engine]
          ↓
LLM Summary Formulation -> User
```

---

## 🔬 Quantitative Engine

### Walk-Forward Optimization (WFO)
Rather than optimizing parameters over the entire dataset, Black Swan perpetually trains and validates parameters forward through time via Optuna.

- **Train/Test Windows:** Rolling out-of-sample validation.
- **Engine:** Numba-accelerated `@njit` kernels for near-C execution speeds.
- **Costs:** Variable fees and slippage applied at each position change.

### Tax Engine (Holding-Period Aware)
Black Swan tracks every discrete trade to determine holding duration. It applies country-specific tax regimes:
- **India:** STCG (20%) / LTCG (12.5%) after 365 days.
- **USA:** STCG (37% max) / LTCG (20%) after 365 days.
- **UK:** 20% Capital Gains.

---

## 🎲 Adversarial Monte Carlo Lab

Black Swan applies multiple stress regimes to the Out-of-Sample equity curve:

1. **Execution Failure Simulation (Connectivity)**
   - Randomly drops 20% of trades to simulate latency or broker disconnects.
2. **Trade Order Randomization**
   - Tests if survival was dependent on a specific historical chronological sequence.
3. **Temporal Disorder Simulation**
   - Shuffles market days individually to destroy trend dependency.
4. **Synthetic Market Generation (GBM)**
   - Generates large-scale Gaussian GBM price paths calibrated to observed volatility.

---

## 🚀 Usage (A2A Confirmation Flow)

Black Swan requires a two-step confirmation for institutional safety.

**1. Initial Request:**
```bash
curl -X POST http://localhost:5000/ \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "msg-01",
      "timestamp": "2026-03-29T00:00:00Z",
      "role": "user",
      "parts": [{
        "text": "Run a 2y SMA suite on AAPL. Apply 0.1% fees and Indian tax regime."
      }]
    }
  }
}'
```

**2. Confirmation (Reply 'yes' with contextId):**
The agent will present a summary of the execution costs. Reply "yes" mapped to the new contextId to trigger the compute-intensive Numba simulation.

```bash
curl -X POST http://localhost:5000/ \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "msg-02",
      "contextId": "YOUR_CONTEXT_ID_HERE",
      "timestamp": "2026-03-29T00:01:00Z",
      "role": "user",
      "parts": [{
        "text": "yes"
      }]
    }
  }
}'
```

---

## 🔮 Future Work

- [ ] Position sizing (Kelly Criterion, ATR-based risk targeting).
- [ ] Multi-asset portfolio simulation and co-integration testing.
- [ ] Market regime detection (bull/bear/sideways segregation).
- [ ] GPU-accelerated Monte Carlo execution paths.
