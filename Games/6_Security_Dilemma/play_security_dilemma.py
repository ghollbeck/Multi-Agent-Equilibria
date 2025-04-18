#!/usr/bin/env python3
"""
play_security_dilemma.py

Implements the Security Dilemma Game: a modified Prisoner's Dilemma with misperception.
Logs results in CSV/JSON, terminal outputs to a text file, and generates payoff plots.
Each run creates a timestamped folder under 'results/'.
"""

import argparse
import os
import sys
import csv
import json
import logging
import random
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Standard PD payoffs: T>R>P>S
R = 3   # mutual cooperation (Low, Low)
T = 5   # temptation to defect  (High vs. Low)
S = 0   # sucker's payoff      (Low vs. High)
PUN = 1 # mutual defection     (High, High)

def misperceive(action, prob):
    """Flip Low<->High with probability prob."""
    if random.random() < prob:
        return 'High' if action == 'Low' else 'Low'
    return action

def compute_payoff(self_act, perceived_other):
    """Compute payoff based on self action and perceived opponent action."""
    if self_act == 'Low' and perceived_other == 'Low':
        return R
    if self_act == 'High' and perceived_other == 'High':
        return PUN
    if self_act == 'High' and perceived_other == 'Low':
        return T
    if self_act == 'Low' and perceived_other == 'High':
        return S
    return None

def random_strategy(history, role):
    """Computer: choose uniformly at random."""
    return random.choice(['Low','High'])

def tit_for_tat_strategy(history, role):
    """Computer: start with Low, then copy the opponent's last perceived action."""
    if not history:
        return 'Low'
    last = history[-1]
    return last['perceived_B_by_A'] if role == 'A' else last['perceived_A_by_B']

def setup_logger(log_path):
    """Logger that writes INFO+ to both stdout and to a file."""
    logger = logging.getLogger('SecurityDilemma')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def get_action(role, mode, round_num, history, strategy, logger):
    """Return (action, response_time) for human or computer."""
    if mode == 'computer':
        start = time.time()
        action = (random_strategy if strategy=='random' else tit_for_tat_strategy)(history, role)
        rt = time.time() - start
        logger.info(f'Round {round_num} - computer ({role}) action: {action}')
        return action, rt

    action = None
    while action is None:
        prompt = f'Round {round_num} - {role} choose [L]ow or [H]igh: '
        start = time.time()
        resp = input(prompt).strip().upper()
        rt = time.time() - start
        if resp in ('L','LOW'):
            action = 'Low'
        elif resp in ('H','HIGH'):
            action = 'High'
        else:
            logger.info('Invalid input; please enter L or H.')
    logger.info(f'Round {round_num} - human ({role}) action: {action} (rt={rt:.3f}s)')
    return action, rt

def main():
    p = argparse.ArgumentParser(description='Security Dilemma Game')
    p.add_argument('--participant-id',    required=True,           help='Your participant ID')
    p.add_argument('--opponent-mode',     choices=['human','computer'], default='computer')
    p.add_argument('--misinterpretation-prob', type=float, default=0.1)
    p.add_argument('--rounds',            type=int, default=10)
    p.add_argument('--strategy',          choices=['random','tit_for_tat'], default='tit_for_tat')
    args = p.parse_args()

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(os.getcwd(), 'results', f'{ts}_{args.participant_id}')
    os.makedirs(run_dir, exist_ok=True)

    logf = os.path.join(run_dir, 'terminal.txt')
    logger = setup_logger(logf)
    logger.info('Starting Security Dilemma Game')
    logger.info(f'Parameters: {args}')

    history = []
    for r in range(1, args.rounds+1):
        aA, tA = get_action('A','human',r,history,args.strategy,logger)
        aB, tB = get_action('B', args.opponent_mode, r, history, args.strategy, logger)

        pB = misperceive(aB, args.misinterpretation_prob)
        pA = misperceive(aA, args.misinterpretation_prob)

        payA = compute_payoff(aA, pB)
        payB = compute_payoff(aB, pA)

        logger.info(f'Round {r} results → true: A={aA}, B={aB}; perceived: A→B={pB}, B→A={pA}')
        logger.info(f'Round {r} payoffs → A={payA}, B={payB}')

        history.append({
            'round': r,
            'action_A': aA,
            'rt_A': round(tA,4),
            'action_B': aB,
            'rt_B': round(tB,4),
            'perceived_B_by_A': pB,
            'perceived_A_by_B': pA,
            'payoff_A': payA,
            'payoff_B': payB
        })

    csvf = os.path.join(run_dir, 'results.csv')
    with open(csvf, 'w', newline='') as outf:
        w = csv.DictWriter(outf, fieldnames=history[0].keys())
        w.writeheader()
        for rec in history:
            w.writerow(rec)
    logger.info(f'Saved CSV → {csvf}')

    jsonf = os.path.join(run_dir, 'results.json')
    with open(jsonf, 'w') as outf:
        json.dump(history, outf, indent=2)
    logger.info(f'Saved JSON → {jsonf}')

    rounds = [h['round'] for h in history]
    pA = [h['payoff_A'] for h in history]
    pB = [h['payoff_B'] for h in history]
    plt.figure()
    plt.plot(rounds, pA, marker='o', label='A')
    plt.plot(rounds, pB, marker='s', label='B')
    plt.xlabel('Round'); plt.ylabel('Payoff')
    plt.title('Security Dilemma Payoffs over Rounds')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plotf = os.path.join(run_dir, 'payoffs.png')
    plt.savefig(plotf)
    logger.info(f'Saved payoff plot → {plotf}')

    logger.info(f'Run complete. All outputs in {run_dir}')

if __name__ == '__main__':
    main()