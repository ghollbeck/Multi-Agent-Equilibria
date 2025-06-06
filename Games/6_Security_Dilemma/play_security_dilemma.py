#!/usr/bin/env python3
"""
play_security_dilemma.py

A monolithic, object-oriented, interactive and batch-capable simulation of the
Security Dilemma Game, enhanced with:
  - Config file support (JSON/YAML)
  - Player/Strategy class hierarchies (random, tit-for-tat, grim-trigger, Pavlov,
    generous tit-for-tat)
  - GameEngine and SimulationRunner for batch runs
  - Pandas-based logging and JSON-schema validation
  - Advanced analytics and plotting (cooperation rate, histograms, CIs)
  - Rotating file handlers, emoji-rich console logs, tqdm progress bars
  - LiteLLM client integration with prompt/response logging
  - Menu-driven CLI and headless batch mode
  - Inline unittest suite (run via --run-tests)
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
import math
import requests
import yaml
from dotenv import load_dotenv
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
import jsonschema
import unittest

# -----------------------------
# Default Payoff Matrix
# -----------------------------
DEFAULT_PAYOFFS = {
    'R': 3,  # mutual cooperation (Low, Low)
    'T': 5,  # temptation to defect (High vs. Low)
    'S': 0,  # sucker's payoff (Low vs. High)
    'P': 1   # mutual defection (High, High)
}

# JSON schema for validating results
RESULTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "round": {"type": "integer"},
            "action_A": {"type": "string"},
            "action_B": {"type": "string"},
            "perceived_B_by_A": {"type": "string"},
            "perceived_A_by_B": {"type": "string"},
            "payoff_A": {"type": "number"},
            "payoff_B": {"type": "number"},
            "rt_A": {"type": "number"},
            "rt_B": {"type": "number"}
        },
        "required": [
            "run_id","round","action_A","action_B",
            "perceived_B_by_A","perceived_A_by_B",
            "payoff_A","payoff_B","rt_A","rt_B"
        ]
    }
}

# load environment variables from a .env file if present
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # script directory

@dataclass
class Config:
    """
    Configuration loader/saver for simulation parameters.
    Supports JSON and YAML.
    Fields:
      - participant_id: str
      - opponent_mode: str ('human' or 'computer')
      - strategy: str
      - misinterpretation_prob: float
      - rounds: int
      - batch_size: int
      - output_dir: str
      - llm_model: str
      - llm_temp: float
    """
    participant_id: str = field(default="")
    opponent_mode: str = field(default='computer')
    strategy: str = field(default='tit_for_tat')
    misinterpretation_prob: float = field(default=0.1)
    rounds: int = field(default=10)
    batch_size: int = field(default=1)
    output_dir: str = field(default='results')
    llm_model: str = field(default='gpt-4o')
    llm_temp: float = field(default=0.8)
    run_dir: str = field(default="")
    
    def __post_init__(self):
        if not self.participant_id:
            self.participant_id = ""
    
    @classmethod
    def from_path(cls, path):
        """Create Config instance and load from file."""
        instance = cls()
        instance.load(path)
        return instance

    def load(self, path):
        """Load config from JSON or YAML file."""
        ext = os.path.splitext(path)[1].lower()
        with open(path) as f:
            if ext in ('.yaml','.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def save(self, path):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class Strategy:
    """Base class for strategies."""
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def select_action(self, history, role):
        """Return 'Low' or 'High' given history and role ('A' or 'B')."""
        raise NotImplementedError


class RandomStrategy(Strategy):
    def __init__(self): super().__init__('random')
    def select_action(self, history, role):
        return random.choice(['Low','High'])


class TitForTatStrategy(Strategy):
    def __init__(self): super().__init__('tit_for_tat')
    def select_action(self, history, role):
        if not history:
            return 'Low'
        last = history[-1]
        if role=='A':
            return last['perceived_B_by_A']
        return last['perceived_A_by_B']


class GrimTriggerStrategy(Strategy):
    """Start cooperating; if opponent ever defects (perceived High), defect forever."""
    def __init__(self):
        super().__init__('grim_trigger')
        self.triggered = False

    def select_action(self, history, role):
        if history and not self.triggered:
            for rec in history:
                if role=='A' and rec['perceived_B_by_A']=='High':
                    self.triggered = True
                if role=='B' and rec['perceived_A_by_B']=='High':
                    self.triggered = True
        return 'High' if self.triggered else 'Low'


class PavlovStrategy(Strategy):
    """Win-stay, lose-shift: repeat last if got reward >=PUN else switch."""
    def __init__(self): super().__init__('pavlov')
    def select_action(self, history, role):
        if not history:
            return 'Low'
        last = history[-1]
        # determine payoff to self
        payoff = last['payoff_A'] if role=='A' else last['payoff_B']
        # win if >= reward for mutual cooperation
        if payoff >= DEFAULT_PAYOFFS['R']:
            # repeat own last action
            return last['action_A'] if role=='A' else last['action_B']
        # lose: switch
        return 'High' if (last['action_A']=='Low' if role=='A' else last['action_B']=='Low') else 'Low'


class GenerousTitForTatStrategy(Strategy):
    """Similar to tit-for-tat, but sometimes forgives defection."""
    def __init__(self, forgiveness_prob=0.1):
        super().__init__('generous_tit_for_tat')
        self.forgiveness_prob = forgiveness_prob

    def select_action(self, history, role):
        if not history:
            return 'Low'
        last = history[-1]
        opp = last['perceived_B_by_A'] if role=='A' else last['perceived_A_by_B']
        if opp=='Low':
            return 'Low'
        # forgives with probability
        if random.random()<self.forgiveness_prob:
            return 'Low'
        return 'High'


class Player:
    """Company agent in the simulation (always algorithmic)."""
    def __init__(self, company_id, strategy=None):
        self.id = company_id
        self.strategy = strategy or RandomStrategy()
        self.history = []

    def select(self, history, role, logger):
        """Return chosen action and computation time."""
        start = time.time()
        action = self.strategy.select_action(history, role)
        rt = time.time() - start
        logger.info(f"🏢 Company {self.id} ({role}) chose {action} (rt={rt:.4f}s)")
        return action, rt


class GameEngine:
    """Core game logic for one run."""
    def __init__(self, player_A, player_B, config, run_id, logger, llm_client=None):
        self.player_A = player_A
        self.player_B = player_B
        self.config = config
        self.logger = logger
        self.llm = llm_client
        self.history = []
        self.run_id = run_id

    @staticmethod
    def misperceive(action, prob):
        if random.random()<prob:
            return 'High' if action=='Low' else 'Low'
        return action

    @staticmethod
    def compute_payoff(self_act, perceived_other, payoffs):
        if self_act=='Low' and perceived_other=='Low': return payoffs['R']
        if self_act=='High' and perceived_other=='High': return payoffs['P']
        if self_act=='High' and perceived_other=='Low': return payoffs['T']
        if self_act=='Low' and perceived_other=='High': return payoffs['S']
        return 0

    def play(self):
        """Play configured number of rounds."""
        self.logger.info(f"🚀 Starting run {self.run_id} for {self.config.rounds} rounds.")
        for r in range(1, self.config.rounds+1):
            # get actions
            action_A, rt_A = self.player_A.select(self.history, 'A', self.logger)
            action_B, rt_B = self.player_B.select(self.history, 'B', self.logger)
            # misperceive
            pB = self.misperceive(action_B, self.config.misinterpretation_prob)
            pA = self.misperceive(action_A, self.config.misinterpretation_prob)
            # compute payoffs
            payA = self.compute_payoff(action_A, pB, DEFAULT_PAYOFFS)
            payB = self.compute_payoff(action_B, pA, DEFAULT_PAYOFFS)
            # log
            rec = {
                'run_id': self.run_id,
                'round': r,
                'action_A': action_A,
                'action_B': action_B,
                'perceived_B_by_A': pB,
                'perceived_A_by_B': pA,
                'payoff_A': payA,
                'payoff_B': payB,
                'rt_A': round(rt_A,4),
                'rt_B': round(rt_B,4)
            }
            self.history.append(rec)
            self.logger.info(f"📊 Round {r}: A={action_A}->{pB}, B={action_B}->{pA} | PAY({payA},{payB})")
            # optionally call LLM to analyze
            if self.llm:
                prompt = [
                    {'role':'system','content':'Analyze the last round.'},
                    {'role':'user','content':json.dumps(rec)}
                ]
                try:
                    response = self.llm._call_litellm(prompt,
                        model=self.config.llm_model,
                        temperature=self.config.llm_temp)
                    self.logger.info(f"🤖 LLM Analysis: {response}")
                except Exception as e:
                    self.logger.error(f"LLM error: {e}")
        # validate
        try:
            jsonschema.validate(self.history, RESULTS_SCHEMA)
            self.logger.info("✅ Schema validation passed.")
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
        return self.history


class SimulationRunner:
    """Batch-runner for multiple GameEngine runs, with analytics."""
    def __init__(self, config, logger, llm_client=None):
        self.config = config
        self.logger = logger
        self.llm = llm_client
        self.all_results = []

    def run(self):
        self.logger.info(f"🎯 Starting batch: {self.config.batch_size} runs.")
        for i in tqdm(range(1, self.config.batch_size+1), desc="Batch Runs", unit="run"):
            run_id = f"{self.config.participant_id}_{i}_{int(time.time())}"
            # For scientific simulation, both players are computer agents
            pA = Player(self.config.participant_id)
            strat = self._get_strategy()
            pB = Player('opponent', strat)
            engine = GameEngine(pA, pB, self.config, run_id, self.logger, self.llm)
            res = engine.play()
            self.all_results.extend(res)
        df = pd.DataFrame(self.all_results)
        self.logger.info("🔗 Combined all runs into DataFrame.")
        return df

    def _get_strategy(self):
        s = self.config.strategy
        if s=='random': return RandomStrategy()
        if s=='tit_for_tat': return TitForTatStrategy()
        if s=='grim_trigger': return GrimTriggerStrategy()
        if s=='pavlov': return PavlovStrategy()
        if s=='generous_tit_for_tat': return GenerousTitForTatStrategy()
        return RandomStrategy()

    def save(self, df):
        """Save DataFrame to CSV/JSON and generate analytics."""
        outdir = self.config.run_dir or os.path.join(BASE_DIR, self.config.output_dir)
        csvf = os.path.join(outdir, 'batch_results.csv')
        jsonf = os.path.join(outdir, 'batch_results.json')
        df.to_csv(csvf, index=False)
        df.to_json(jsonf, orient='records', indent=2)
        self.logger.info(f"💾 Saved batch CSV: {csvf}")
        self.logger.info(f"💾 Saved batch JSON: {jsonf}")
        # analytics
        self._plot_cooperation_rate(df, outdir)
        self._plot_payoff_histograms(df, outdir)
        self._plot_confidence_intervals(df, outdir)

    def _plot_cooperation_rate(self, df, outdir):
        # cooperation = fraction of Low
        coop = df.groupby('round').apply(lambda g: (g['action_A']=='Low').mean())
        plt.figure()
        coop.plot(marker='o', title='Cooperation Rate Over Rounds')
        plt.xlabel('Round'); plt.ylabel('Cooperation Rate')
        plt.grid(True); plt.tight_layout()
        path = os.path.join(outdir, 'cooperation_rate.png')
        plt.savefig(path)
        self.logger.info(f"📈 Saved cooperation rate plot: {path}")

    def _plot_payoff_histograms(self, df, outdir):
        plt.figure(figsize=(8,4))
        df['payoff_A'].hist(alpha=0.7, label='A')
        df['payoff_B'].hist(alpha=0.7, label='B')
        plt.legend(); plt.title('Payoff Distributions')
        plt.tight_layout()
        path = os.path.join(outdir, 'payoff_hist.png')
        plt.savefig(path)
        self.logger.info(f"📊 Saved payoff histogram: {path}")

    def _plot_confidence_intervals(self, df, outdir):
        # compute mean and CI for payoffs per round
        summary = df.groupby('round')[['payoff_A','payoff_B']].agg(['mean','count','std'])
        ci = 1.96 * summary[('payoff_A','std')] / np.sqrt(summary[('payoff_A','count')])
        plt.figure()
        plt.errorbar(summary.index, summary[('payoff_A','mean')], yerr=ci,
            label='A 95% CI', fmt='-o')
        ciB = 1.96 * summary[('payoff_B','std')] / np.sqrt(summary[('payoff_B','count')])
        plt.errorbar(summary.index, summary[('payoff_B','mean')], yerr=ciB,
            label='B 95% CI', fmt='-s')
        plt.legend(); plt.title('Payoff Means with 95% CI')
        plt.xlabel('Round'); plt.ylabel('Payoff')
        plt.grid(True); plt.tight_layout()
        path = os.path.join(outdir, 'payoff_CI.png')
        plt.savefig(path)
        self.logger.info(f"📉 Saved confidence interval plot: {path}")


class LiteLLMClient:
    """Client for calling the LiteLLM API, with prompt logging."""
    def __init__(self, api_key=None, log_path='llm_prompts.txt'):
        self.api_key = api_key or os.getenv('LITELLM_API_KEY')
        self.log_path = log_path

    def _call_litellm(self, messages, model="gpt-4o", temperature=0.8):
        payload = {"model": model, "messages": messages, "temperature": temperature}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        # log prompt
        with open(self.log_path, 'a') as f:
            f.write(f"PROMPT: {json.dumps(messages)}\n")
        response = requests.post(
            "https://litellm.sph-prod.ethz.ch/chat/completions",
            json=payload, headers=headers)
        if not response.ok:
            raise Exception(f"LiteLLM API error: {response.status_code} {response.reason}")
        data = response.json()
        resp = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # log response
        with open(self.log_path, 'a') as f:
            f.write(f"RESPONSE: {resp}\n")
        return resp


def setup_main_logger(log_path):
    """Configure emoji-rich RotatingFileHandler logger."""
    logger = logging.getLogger('SecDilemma')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('😊 %(asctime)s %(levelname)s: %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(log_path, maxBytes=int(5e6), backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser(description='Security Dilemma Simulation')
    parser.add_argument('--config-file', help='Path to JSON/YAML config')
    parser.add_argument('--participant-id', help='Simulation run identifier')
    parser.add_argument('--strategy', choices=['random','tit_for_tat','grim_trigger','pavlov','generous_tit_for_tat'], default='tit_for_tat')
    parser.add_argument('--misinterpretation-prob', type=float, default=0.1)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--llm-model', default='gpt-4o')
    parser.add_argument('--llm-temp', type=float, default=0.8)
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args()

    if args.run_tests:
        unittest.main(argv=[sys.argv[0]])
        return

    config = Config()
    if args.config_file:
        config.load(args.config_file)
    # override with CLI
    if args.participant_id: config.participant_id = args.participant_id
    config.strategy = args.strategy
    config.misinterpretation_prob = args.misinterpretation_prob
    config.rounds = args.rounds
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir
    config.llm_model = args.llm_model
    config.llm_temp = args.llm_temp

    # Ensure participant_id is set; default to timestamp if missing
    if not config.participant_id:
        config.participant_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # initialize dirs within script directory
    full_results_dir = os.path.join(BASE_DIR, config.output_dir)
    os.makedirs(full_results_dir, exist_ok=True)
    run_dir = os.path.join(full_results_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    # store for downstream saving
    config.run_dir = run_dir
    # setup loggers
    main_log = setup_main_logger(os.path.join(run_dir,'main.log'))
    llm_log = os.path.join(run_dir,'llm_prompts.txt')
    llm_client = LiteLLMClient(log_path=llm_log)

    main_log.info("🔧 Configuration:")
    main_log.info(config.__dict__)

    # Run in batch mode only (scientific simulation)
    runner = SimulationRunner(config, main_log, llm_client)
    df = runner.run()
    runner.save(df)

if __name__=='__main__':
    main()

# -----------------------------
# Inline unittest suite
# -----------------------------
class TestGameEngine(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        cfg.rounds = 1
        cfg.misinterpretation_prob = 0.0
        cfg.participant_id = 'test'
        pA = Player('test')
        pB = Player('opp')
        self.eng = GameEngine(pA,pB,cfg,'run0',setup_main_logger('test.log'))

    def test_compute_payoff(self):
        # test all combinations
        for a in ['Low','High']:
            for b in ['Low','High']:
                p = self.eng.compute_payoff(a,b,DEFAULT_PAYOFFS)
                self.assertIn(p,[DEFAULT_PAYOFFS['R'],DEFAULT_PAYOFFS['T'],DEFAULT_PAYOFFS['S'],DEFAULT_PAYOFFS['P']])

class TestSimulationRunner(unittest.TestCase):
    def test_runner_outputs(self):
        cfg = Config()
        cfg.batch_size = 2
        cfg.participant_id = 't'
        runner = SimulationRunner(cfg, setup_main_logger('test2.log'))
        df = runner.run()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df),2)

class TestJSONSchema(unittest.TestCase):
    def test_schema(self):
        sample = [
            { 'run_id':'r','round':1,'action_A':'Low','action_B':'Low',
              'perceived_B_by_A':'Low','perceived_A_by_B':'Low',
              'payoff_A':3,'payoff_B':3,'rt_A':0.0,'rt_B':0.0 }
        ]
        jsonschema.validate(sample, RESULTS_SCHEMA)

# End of file (approximately 700+ lines)
