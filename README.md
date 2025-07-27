# Compound V2/V3 Wallet Risk Scoring

This project computes a risk score (0–1000) for a list of Ethereum wallet addresses, based on their Compound V2/V3 protocol usage. The score reflects responsible usage versus risky or bot-like behavior, helping assess wallet reliability in DeFi.

## Features

- Takes a list of wallet addresses in `Wallet.csv`
- For each wallet, gathers transaction data (mocked by default; swap for real queries)
- Engineers DeFi risk features (activity, repayment, liquidations, diversity)
- Outputs a CSV with `wallet_id,score`
- Scalable for 100s or 1000s of wallets

## Usage

Install dependencies:

pip install pandas

Run scoring:
python score_compound_wallets.py --wallets Wallet.csv --output risk_scores.csv


## How it Works

1. **Input**: `Wallet.csv` — a CSV file with one wallet address per line (with or without a header).
2. **Fetch transactions**: The script collects all Compound protocol events for each wallet. *(By default, this is mocked; for production, integrate a data source/API in `fetch_compound_transactions`.)*
3. **Feature Engineering**: For each wallet, the following are calculated:
    - Number of deposits, borrows, repays, liquidations
    - Asset diversity
    - Days active
    - Collateral ratio and repayment ratio
    - Transaction pattern timing
4. **Scoring**: Each wallet receives a risk score (0–1000). High scores = reliable. Low scores = risk.

## Output

- `risk_scores.csv`:
    | wallet_id | score |
    | --------- | ----- |
    | 0xabc...  | 732   |

## To Use Real Data

Replace the logic in `fetch_compound_transactions` with on-chain queries or subgraph lookups, so each wallet’s Compound actions are accurately gathered.

## License

MIT

## Author

Srinath
srinathselvakumar1505@gmail.com
