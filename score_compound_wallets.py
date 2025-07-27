#!/usr/bin/env python3
"""
Compound V2/V3 Wallet Risk Scoring (from Wallet List)

Given a CSV of wallet addresses, fetch Compound transactions,
engineer risk features, and output wallet scores (0-1000).

Usage:
    python compound_risk_score_from_wallets.py --wallets Wallet.csv --output risk_scores.csv

Author: [Your Name]
"""

import argparse
import pandas as pd
import numpy as np
import sys
import time
from typing import List

def read_wallet_list(wallet_file: str) -> List[str]:
    """Read list of wallet addresses from a CSV file (single column, with or without header)."""
    wallets = pd.read_csv(wallet_file, header=None, dtype=str)
    possible_wallets = set()
    for col in wallets.columns:
        for w in wallets[col]:
            if isinstance(w, str) and w.lower().startswith('0x') and len(w) == 42:
                possible_wallets.add(w.lower())
    if not possible_wallets:
        raise ValueError("No wallet addresses found in the wallet file.")
    return sorted(possible_wallets)

def fetch_compound_transactions(wallet_id: str) -> pd.DataFrame:
    """
    MOCK: Replace this with your API, on-chain, or subgraph query to fetch all Compound
    transactions for the wallet_id (across relevant actions: supply/mint, borrow, repay, redeem, liquidate).
    
    Returns a DataFrame with columns:
    - wallet_id
    - timestamp
    - action
    - amount
    - asset
    - (optionally asset_price_usd)
    """
    # ---- Replace this block with a real fetch for each wallet ----
    # Here we generate some mock random data for demonstration.
    np.random.seed(int(wallet_id[-6:], 16) % 1000000)  # Seed for reproducibility by wallet
    n = np.random.randint(5, 30)
    actions = np.random.choice(
        ['supply', 'borrow', 'repay', 'redeem', 'liquidate'],
        size=n,
        p=[0.35, 0.2, 0.2, 0.2, 0.05]
    )
    now = int(time.time())
    timestamps = np.sort(np.random.randint(now - 86400*365, now, size=n))
    amounts = np.abs(np.random.normal(loc=1000, scale=500, size=n))
    assets = np.random.choice(['USDC', 'DAI', 'ETH', 'WBTC', 'COMP'], size=n)
    df = pd.DataFrame({
        'wallet_id': wallet_id,
        'timestamp': timestamps,
        'action': actions,
        'amount': amounts,
        'asset': assets,
        'asset_price_usd': np.random.uniform(0.99, 1.05, size=n)  # For stablecoins, mock USD price
    })
    return df
    # ---- End of mock ----

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['action'] = df['action'].str.lower().str.strip()
    df['asset'] = df['asset'].str.upper()
    if 'asset_price_usd' in df.columns:
        df['usd_value'] = df['amount'] * pd.to_numeric(df['asset_price_usd'], errors='coerce').fillna(1)
    else:
        df['usd_value'] = df['amount']
    df['date'] = df['timestamp'].dt.date
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    wallets = []
    for wallet, group in df.groupby('wallet_id'):
        deposit_mask = group['action'].isin(['supply', 'mint', 'deposit'])
        borrow_mask = group['action'].isin(['borrow'])
        repay_mask = group['action'].isin(['repay', 'repayborrow'])
        withdraw_mask = group['action'].isin(['redeem', 'withdraw'])
        liquidation_mask = group['action'].isin(['liquidate', 'liquidateborrow'])

        total_deposit = group[deposit_mask]['usd_value'].sum()
        total_borrow = group[borrow_mask]['usd_value'].sum()
        total_repay = group[repay_mask]['usd_value'].sum()

        f = {
            'wallet_id': wallet,
            'n_deposits': deposit_mask.sum(),
            'n_borrows': borrow_mask.sum(),
            'n_repays': repay_mask.sum(),
            'n_withdraws': withdraw_mask.sum(),
            'n_liquidations': liquidation_mask.sum(),
            'unique_assets': group['asset'].nunique(),
            'active_days': group['date'].nunique(),
            'total_deposit': total_deposit,
            'total_borrow': total_borrow,
            'total_repay': total_repay,
        }
        f['repay_ratio'] = f['total_repay'] / f['total_borrow'] if f['total_borrow'] > 0 else 0
        f['collateral_ratio'] = f['total_deposit'] / f['total_borrow'] if f['total_borrow'] > 0 else np.inf

        if len(group) > 1:
            gsorted = group.sort_values('timestamp')
            diffs = gsorted['timestamp'].diff().dt.total_seconds().dropna()
            f['std_time_between'] = diffs.std() if not diffs.empty else 0
            f['mean_time_between'] = diffs.mean() if not diffs.empty else 0
        else:
            f['std_time_between'] = 0
            f['mean_time_between'] = 0
        wallets.append(f)
    return pd.DataFrame(wallets)

def score_wallet(features: pd.Series) -> float:
    score = 500
    score -= features['n_liquidations'] * 150
    if features['repay_ratio'] > 1.0:
        score += 100
    elif features['repay_ratio'] > 0.8:
        score += 60
    elif features['repay_ratio'] > 0.5:
        score += 25
    elif features['repay_ratio'] > 0.2:
        score -= 25
    else:
        score -= 50
    if features['collateral_ratio'] > 2.0:
        score += 50
    elif features['collateral_ratio'] > 1.2:
        score += 25
    elif features['collateral_ratio'] < 0.8:
        score -= 25
    score += min(features['active_days'] * 4, 60)
    score += min(features['unique_assets'] * 10, 40)
    score += min(features['total_deposit'] / 1e5, 40)
    score += min(features['total_borrow'] / 1e5, 40)
    if features['std_time_between'] < 3600 and features['n_borrows'] + features['n_deposits'] > 5:
        score -= 60
    return float(max(0, min(score, 1000)))

def main():
    parser = argparse.ArgumentParser(description="Compound Wallet Risk Scoring (from Wallet List)")
    parser.add_argument('--wallets', required=True, help="CSV file with wallet addresses")
    parser.add_argument('--output', default="risk_scores.csv", help="Output CSV file")
    args = parser.parse_args()

    try:
        wallet_ids = read_wallet_list(args.wallets)
        all_txs = []
        print(f"Found {len(wallet_ids)} wallets. Fetching transaction data...")
        for i, wallet in enumerate(wallet_ids, 1):
            print(f"Fetching transactions for wallet {i}/{len(wallet_ids)}: {wallet}")
            # You must replace this with actual data fetching in production!
            txs = fetch_compound_transactions(wallet)
            all_txs.append(txs)
        df = pd.concat(all_txs, ignore_index=True)
        df = preprocess_data(df)
        features_df = engineer_features(df)
        features_df['score'] = features_df.apply(score_wallet, axis=1)
        out_df = features_df[['wallet_id', 'score']].sort_values('score', ascending=False)
        out_df.to_csv(args.output, index=False)
        print(f"Scoring complete. Results written to {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
