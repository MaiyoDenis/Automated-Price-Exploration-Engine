#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('datasets/vps_apex_backup.db')
df = pd.read_sql_query('SELECT price FROM ticks ORDER BY timestamp ASC', conn)
conn.close()

decimals = 3
prices = df['price'].values
digits = [int(f'{p:.{decimals}f}'[-1]) for p in prices]
total = len(digits)

TOP_BIGRAMS = {
    (4, 8): 7,
    (2, 2): 0,
    (7, 7): 5,
    (7, 4): 8,
    (9, 2): 8,
    (5, 2): 4,
    (3, 4): 5,
    (8, 8): 8,
}

def simulate(rule_name, ban_lost_digit=False, cooldown_ticks=0, recovery_drought=0):
    balance = 1000.0
    base_stake = 1.0
    rec_stake = 11.0
    payout = 0.097
    
    in_recovery = False
    consec_losses = 0
    max_consec = 0
    double_losses_count = 0
    last_loss_tick = -9999
    last_lost_digit = None
    
    total_trades = 0
    wins = 0
    losses = 0
    
    # Drought tracker
    drought = {d: 0 for d in range(10)}
    
    i = 0
    while i < total - 2:
        cur_d = digits[i+1]
        for d in range(10):
            if d == cur_d:
                drought[d] = 0
            else:
                drought[d] += 1
                
        pair = (digits[i], digits[i+1])
        if pair in TOP_BIGRAMS:
            tgt = TOP_BIGRAMS[pair]
            
            # Check recovery filters
            if in_recovery:
                if ban_lost_digit and tgt == last_lost_digit:
                    i += 1
                    continue
                if cooldown_ticks > 0 and (i - last_loss_tick) < cooldown_ticks:
                    i += 1
                    continue
                if recovery_drought > 0 and drought[tgt] < recovery_drought:
                    i += 1
                    continue
                    
            stake = rec_stake if in_recovery else base_stake
            outcome = digits[i+2]
            won = (outcome != tgt)
            total_trades += 1
            
            if won:
                wins += 1
                balance += stake * payout
                consec_losses = 0
                in_recovery = False
                last_lost_digit = None
            else:
                losses += 1
                balance -= stake
                consec_losses += 1
                last_loss_tick = i
                last_lost_digit = tgt
                if consec_losses == 1:
                    in_recovery = True
                elif consec_losses >= 2:
                    double_losses_count += 1
                    in_recovery = False # Halt recovery after 2 losses
                    
            max_consec = max(max_consec, consec_losses)
            i += 2
        else:
            i += 1
            
    pnl = balance - 1000.0
    print(f"=== {rule_name} ===")
    print(f"  Trades: {total_trades}, Wins: {wins}, Losses: {losses}, WR: {wins/total_trades*100:.2f}%")
    print(f"  Max Consecutive Losses: {max_consec}")
    print(f"  Double Losses (2 in a row): {double_losses_count}")
    print(f"  Net PnL with 11x Recovery: ${pnl:+.2f}\n")


if __name__ == '__main__':
    simulate('Standard 11x Recovery (No Filter)')
    simulate('Rule B: Ban Lost Digit on Recovery', ban_lost_digit=True)
    simulate('Rule C: Cooldown 30 ticks (60s) on Recovery', cooldown_ticks=30)
    simulate('Rule D: Drought >= 8 Confirmed Recovery', recovery_drought=8)
    simulate('Rule E: Compound (Ban Lost Digit + Cooldown 30t + Drought >= 8)', ban_lost_digit=True, cooldown_ticks=30, recovery_drought=8)
