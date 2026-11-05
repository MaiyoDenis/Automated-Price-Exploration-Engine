#!/usr/bin/env python3
"""
Comprehensive Analysis of Live VPS Database for Project APEX
Analyzes tick data, sampling rates, digit distributions, Markov dynamics,
and simulates the Differs Strategy on real VPS production data.
"""

import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from project_apex.indicators.digit_analyzer import DigitAnalyzer
from project_apex.strategies.differs_strategy import DiffersStrategy
from project_apex.models.tick import Tick


def run_analysis(db_path: str = "datasets/vps_apex_backup.db"):
    print(f"============================================================")
    print(f"       PROJECT APEX — LIVE VPS DATABASE ANALYSIS")
    print(f"============================================================\n")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, symbol, timestamp, price FROM ticks ORDER BY timestamp ASC", conn)
    candles_df = pd.read_sql_query("SELECT * FROM candles", conn)
    conn.close()

    total_ticks = len(df)
    print(f"[*] Loaded {total_ticks:,} ticks from {db_path}")
    print(f"[*] Loaded {len(candles_df):,} candles from {db_path}\n")

    if total_ticks == 0:
        print("[!] No ticks found in database.")
        return

    # 1. TIME PROFILE & SAMPLING HEALTH
    # Timestamps are in milliseconds
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    t_min = df['dt'].iloc[0]
    t_max = df['dt'].iloc[-1]
    duration_sec = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000.0
    duration_hrs = duration_sec / 3600.0

    print("------------------------------------------------------------")
    print("1. DATASET TIMEFRAME & SAMPLING INTEGRITY")
    print("------------------------------------------------------------")
    print(f"Symbol:               {df['symbol'].iloc[0]}")
    print(f"Total Ticks:          {total_ticks:,}")
    print(f"Start Time (UTC):     {t_min.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time (UTC):       {t_max.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration:       {duration_hrs:.2f} hours ({duration_sec/60:.1f} minutes)")

    # Tick intervals
    df['interval_sec'] = df['timestamp'].diff() / 1000.0
    intervals = df['interval_sec'].dropna()

    median_interval = intervals.median()
    mean_interval = intervals.mean()
    expected_interval = 2.0  # R_25 typically ticks every 2.0 seconds
    expected_ticks = int(duration_sec / expected_interval)
    coverage_pct = (total_ticks / expected_ticks * 100) if expected_ticks > 0 else 100

    gaps_gt_3s = df[df['interval_sec'] > 3.0]
    gaps_gt_10s = df[df['interval_sec'] > 10.0]
    max_gap = intervals.max()

    print(f"Nominal Tick Rate:    Every {median_interval:.1f}s (Mean: {mean_interval:.2f}s)")
    print(f"Expected Ticks:       ~{expected_ticks:,} (based on 2s interval)")
    print(f"Data Coverage:        {coverage_pct:.2f}% of expected ticks captured")
    print(f"Minor Gaps (>3s):     {len(gaps_gt_3s)} occurrences ({len(gaps_gt_3s)/total_ticks*100:.2f}%)")
    print(f"Major Gaps (>10s):    {len(gaps_gt_10s)} occurrences")
    print(f"Maximum Gap:          {max_gap:.1f} seconds")

    # 2. PRICE ACTION & MICROSTRUCTURE
    prices = df['price'].values
    p_open = prices[0]
    p_close = prices[-1]
    p_high = np.max(prices)
    p_low = np.min(prices)
    p_change = p_close - p_open
    p_pct_change = (p_change / p_open) * 100

    price_diffs = np.diff(prices)
    up_ticks = np.sum(price_diffs > 0)
    down_ticks = np.sum(price_diffs < 0)
    flat_ticks = np.sum(price_diffs == 0)

    print("\n------------------------------------------------------------")
    print("2. PRICE ACTION & VOLATILITY (OHLC)")
    print("------------------------------------------------------------")
    print(f"Open:                 {p_open:.3f}")
    print(f"High:                 {p_high:.3f}")
    print(f"Low:                  {p_low:.3f}")
    print(f"Close:                {p_close:.3f}")
    print(f"Range:                {p_high - p_low:.3f} points ({(p_high - p_low)/p_low*100:.2f}%)")
    print(f"Net Change:           {p_change:+.3f} points ({p_pct_change:+.2f}%)")
    print(f"Up Ticks:             {up_ticks:,} ({up_ticks/len(price_diffs)*100:.2f}%)")
    print(f"Down Ticks:           {down_ticks:,} ({down_ticks/len(price_diffs)*100:.2f}%)")
    print(f"Flat Ticks:           {flat_ticks:,} ({flat_ticks/len(price_diffs)*100:.2f}%)")
    print(f"Avg Absolute Delta:   {np.mean(np.abs(price_diffs)):.4f}")
    print(f"Max Single Up-Tick:   {np.max(price_diffs):+.4f}")
    print(f"Max Single Down-Tick: {np.min(price_diffs):+.4f}")

    # 3. LAST DIGIT (LDP) DISTRIBUTION & UNIFORMITY
    # Determine precision:
    decimals = 3
    for p in prices[:100]:
        s = str(p)
        if '.' in s:
            decimals = max(decimals, len(s.split('.')[1]))

    # Extract last digit according to bot's exact logic
    digits = [int(f"{p:.{decimals}f}"[-1]) for p in prices]
    df['digit'] = digits
    digit_counts = Counter(digits)

    # Chi-Square Test for uniformity
    observed = [digit_counts[d] for d in range(10)]
    expected = [total_ticks / 10.0] * 10
    chi2_stat, chi2_p = stats.chisquare(f_obs=observed, f_exp=expected)

    # Shannon Entropy
    probs = [c / total_ticks for c in observed]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(10)  # ~3.321928 bits
    entropy_ratio = (entropy / max_entropy) * 100

    # Even / Odd & Over / Under
    evens = sum(digit_counts[d] for d in [0, 2, 4, 6, 8])
    odds = sum(digit_counts[d] for d in [1, 3, 5, 7, 9])
    unders = sum(digit_counts[d] for d in range(5))
    overs = sum(digit_counts[d] for d in range(5, 10))

    sorted_digits = sorted(digit_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n------------------------------------------------------------")
    print("3. LAST DIGIT (LDP) DISTRIBUTION ANALYSIS")
    print("------------------------------------------------------------")
    print(f"Price Precision:      {decimals} decimal places")
    print("Digit Frequency Breakdown (Target: 10.00% each):")
    for d in range(10):
        cnt = digit_counts[d]
        pct = (cnt / total_ticks) * 100
        dev = pct - 10.0
        bar = "█" * int(round(pct * 2))
        print(f"  Digit {d}: {cnt:5d} ({pct:5.2f}%) [Dev: {dev:+5.2f}%]  {bar}")

    print(f"\nHottest Digit:        Digit {sorted_digits[0][0]} ({sorted_digits[0][1]:,} ticks, {sorted_digits[0][1]/total_ticks*100:.2f}%)")
    print(f"Coldest Digit:        Digit {sorted_digits[-1][0]} ({sorted_digits[-1][1]:,} ticks, {sorted_digits[-1][1]/total_ticks*100:.2f}%)")
    print(f"Even vs Odd:          Even: {evens:,} ({evens/total_ticks*100:.2f}%) | Odd: {odds:,} ({odds/total_ticks*100:.2f}%)")
    print(f"Under(0-4) vs Over:   Under: {unders:,} ({unders/total_ticks*100:.2f}%) | Over: {overs:,} ({overs/total_ticks*100:.2f}%)")
    print(f"Shannon Entropy:      {entropy:.4f} / {max_entropy:.4f} bits ({entropy_ratio:.2f}% efficiency)")
    print(f"Chi-Square Test:      χ² = {chi2_stat:.3f}, p-value = {chi2_p:.4f}")
    if chi2_p > 0.05:
        print("  -> Verdict: Uniform Random (Fail to reject H0). No statistically significant digit bias overall.")
    else:
        print("  -> Verdict: Non-Uniform / Statistically Significant Deviation from pure randomness (p < 0.05).")

    # 4. STREAKS & DROUGHTS (CONSECUTIVE RUNS & ABSENCES)
    max_streak = {d: 0 for d in range(10)}
    cur_streak_d = None
    cur_streak_len = 0

    droughts = {d: [] for d in range(10)}
    last_seen = {d: 0 for d in range(10)}

    for idx, d in enumerate(digits):
        # Drought tracking
        for other in range(10):
            if other == d:
                drought_len = idx - last_seen[other]
                droughts[other].append(drought_len)
                last_seen[other] = idx

        # Streak tracking
        if d == cur_streak_d:
            cur_streak_len += 1
        else:
            if cur_streak_d is not None:
                max_streak[cur_streak_d] = max(max_streak[cur_streak_d], cur_streak_len)
            cur_streak_d = d
            cur_streak_len = 1
    if cur_streak_d is not None:
        max_streak[cur_streak_d] = max(max_streak[cur_streak_d], cur_streak_len)

    print("\n------------------------------------------------------------")
    print("4. STREAK AND DROUGHT ANALYSIS (Critical for Differs)")
    print("------------------------------------------------------------")
    print("Digit | Max Consecutive Repeat | Max Drought (Ticks) | Avg Drought (Ticks)")
    print("------+------------------------+--------------------+--------------------")
    for d in range(10):
        d_list = droughts[d]
        max_dr = max(d_list) if d_list else 0
        avg_dr = np.mean(d_list) if d_list else 0
        print(f"  {d}   |           {max_streak[d]}            |        {max_dr:3d}         |       {avg_dr:5.1f}")

    # Self-transition probability (P(D_t = d | D_{t-1} = d))
    repeat_counts = sum(1 for i in range(1, len(digits)) if digits[i] == digits[i-1])
    print(f"\nOverall Digit Repetition Rate: {repeat_counts:,} / {len(digits)-1:,} ({repeat_counts/(len(digits)-1)*100:.2f}% vs expected 10.00%)")

    # 5. FIRST-ORDER MARKOV TRANSITION MATRIX
    matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(digits) - 1):
        matrix[digits[i], digits[i+1]] += 1

    trans_probs = matrix / matrix.sum(axis=1, keepdims=True)

    print("\n------------------------------------------------------------")
    print("5. TOP 5 MOST FREQUENT & LEAST FREQUENT TRANSITIONS")
    print("------------------------------------------------------------")
    pairs = []
    for i in range(10):
        for j in range(10):
            pairs.append((i, j, matrix[i, j], trans_probs[i, j]))

    pairs.sort(key=lambda x: x[3], reverse=True)
    print("Highest Transition Probabilities P(next | current):")
    for src, dst, cnt, prob in pairs[:5]:
        print(f"  {src} -> {dst}: {prob*100:5.2f}% ({cnt} occurrences)")

    print("\nLowest Transition Probabilities P(next | current):")
    for src, dst, cnt, prob in pairs[-5:]:
        print(f"  {src} -> {dst}: {prob*100:5.2f}% ({cnt} occurrences)")

    # 6. SIMULATION: APEX DIFFERS STRATEGY ON LIVE VPS TICKS
    print("\n------------------------------------------------------------")
    print("6. BOT APEX STRATEGY SIMULATION ON REAL VPS TICKS")
    print("------------------------------------------------------------")
    print("[*] Running DiffersStrategy backtest with base_stake=$2.00, recovery=$22.00, payout=9.7%...")

    from loguru import logger
    logger.remove()  # Suppress debug logs during backtest

    strategy = DiffersStrategy()
    strategy.initialize({"base_confidence": 0.70, "duration_ticks": 1})

    trades = []
    payout_ratio = 0.097  # 9.7% payout on differs
    base_stake = 2.0
    recovery_stake = 22.0  # 11x multiplier

    current_stake = base_stake
    consecutive_losses = 0
    in_recovery = False
    balance = 1000.0  # Starting balance
    peak_balance = balance
    max_drawdown = 0.0

    won_count = 0
    lost_count = 0

    # Simulate sequential tick processing
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        tick = Tick(symbol=row['symbol'], price=row['price'], timestamp=row['timestamp'])
        
        signal = strategy.on_tick(tick)

        if signal is not None:
            # Check recovery confidence filter
            if in_recovery and signal.confidence < 0.80:
                i += 1
                continue

            rec_digit = signal.barrier
            conf = signal.confidence

            # Next tick determines contract settlement (duration_ticks = 1)
            next_row = df.iloc[i + 1]
            next_price = next_row['price']
            next_digit = int(f"{next_price:.{decimals}f}"[-1])

            won = (next_digit != rec_digit)
            stake = current_stake

            if won:
                profit = stake * payout_ratio
                balance += profit
                won_count += 1
                consecutive_losses = 0
                in_recovery = False
                current_stake = base_stake
            else:
                profit = -stake
                balance += profit
                lost_count += 1
                consecutive_losses += 1
                if consecutive_losses == 1:
                    in_recovery = True
                    current_stake = recovery_stake
                else:
                    in_recovery = False
                    current_stake = base_stake

            peak_balance = max(peak_balance, balance)
            dd = peak_balance - balance
            max_drawdown = max(max_drawdown, dd)

            trades.append({
                'tick_idx': i,
                'time': row['dt'],
                'excluded_digit': rec_digit,
                'next_digit': next_digit,
                'won': won,
                'confidence': conf,
                'stake': stake,
                'profit': profit,
                'balance': balance
            })

            # Tell strategy outcome
            strategy.record_trade_outcome(row['symbol'], rec_digit, won, conf)

            # Advance to next tick (settlement tick is processed as next tick)
        i += 1

    trade_df = pd.DataFrame(trades)
    total_trades = len(trade_df)
    if total_trades > 0:
        win_rate = (won_count / total_trades) * 100
        breakeven_wr = (1.0 / (1.0 + payout_ratio)) * 100
        total_pnl = balance - 1000.0

        # Consecutive loss streaks
        loss_streaks = []
        cur_loss = 0
        for w in trade_df['won']:
            if not w:
                cur_loss += 1
            else:
                if cur_loss > 0:
                    loss_streaks.append(cur_loss)
                cur_loss = 0
        if cur_loss > 0:
            loss_streaks.append(cur_loss)
        max_consec_losses = max(loss_streaks) if loss_streaks else 0

        print(f"Total Signals Fired:  {total_trades:,} ({total_trades/total_ticks*100:.1f}% of ticks)")
        print(f"Winning Trades:       {won_count:,} ({win_rate:.2f}%)")
        print(f"Losing Trades:        {lost_count:,} ({100-win_rate:.2f}%)")
        print(f"Required Breakeven WR: {breakeven_wr:.2f}%")
        print(f"Edge over Random:     {win_rate - 90.00:+.2f}% (Random guess = 90.00%)")
        print(f"Max Consecutive Loss: {max_consec_losses}")
        print(f"Simulated Net P&L:    ${total_pnl:+.2f} (from $1,000 initial)")
        print(f"Max Peak-to-Trough DD:${max_drawdown:.2f}")

        # Performance by excluded digit
        print("\nPerformance Breakdown by Excluded Digit:")
        print("Digit | Trades | Wins  | Losses | Win Rate | Net P&L ($)")
        print("------+--------+-------+--------+----------+------------")
        for d in range(10):
            sub = trade_df[trade_df['excluded_digit'] == d]
            if len(sub) > 0:
                d_w = sub['won'].sum()
                d_l = len(sub) - d_w
                d_wr = (d_w / len(sub)) * 100
                d_pnl = sub['profit'].sum()
                print(f"  {d}   |  {len(sub):5d} | {d_w:5d} |  {d_l:4d}  |  {d_wr:5.2f}%  |  {d_pnl:+8.2f}")

    print("\n============================================================")
    print("ANALYSIS COMPLETE.")
    print("============================================================\n")


if __name__ == '__main__':
    run_analysis()
