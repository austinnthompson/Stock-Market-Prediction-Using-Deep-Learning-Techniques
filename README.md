# Stock Market Prediction Using Deep Learning Techniques  

## Live Prediction Example (as of December 5, 2025)
- **Model predicted SPY closing price:** `$685.84`  
- **Actual SPY closing price:** `$685.69`  
- **Difference:** **-$0.15** (0.02% error)

## Project Overview
This project builds a **hybrid deep learning model** to predict the **weekly closing price** of the SPY ETF (S&P 500) by combining:
- Historical price + volume data
- Technical indicators (RSI, MACD, moving averages, momentum)
- Financial news sentiment embeddings (via pre-trained **FinBERT-Tone**)

The model uses a **CNN-GRU** architecture and outputs both a point prediction and a probabilistic forecast via Monte Carlo simulation.

### Key Results
- Mean test MSE: **0.00078** → ~**2.80% average error** in weekly log-return prediction
- Includes 80% prediction interval and full distribution visualization

## Methodology Summary

| Component                  | Description                                                                                   |
|---------------------------|-------------------------------------------------------------------------------------------------|
| **Data Sources**          | `yfinance` (10 years of weekly SPY data)<br>`Finnhub API` (market news)                        |
| **Technical Features**    | 2/4-week MA, momentum (1/2/4w), RSI, MACD                                                      |
| **Sentiment Analysis**    | FinBERT-Tone (Hugging Face) → 768-dim embeddings → compressed to 64-dim                       |
| **Model Architecture**    | 1D CNN → extracts local patterns from price sequences<br>Linear compressor for news embeddings<br>GRU (128 hidden units) → final prediction of next week’s log return |
| **Sequence Length**       | 4 weeks (tunable)                                                                              |
| **Training**              | PyTorch, Adam optimizer, MSE loss, 2000 epochs, batch size 32, learning rate 1e-6             |
| **Uncertainty Modeling**  | Monte Carlo simulation (10,000 samples) using test RMSE as σ                                   |
