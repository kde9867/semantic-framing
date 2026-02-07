import pandas as pd
import torch
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import logging
import re
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jargons(jargon_path):
    """Load jargon terms from file"""
    if not os.path.exists(jargon_path):
        logger.warning(f"Jargon file not found: {jargon_path}")
        return set()
    
    with open(jargon_path, 'r', encoding='utf-8') as f:
        jargons = set(line.strip() for line in f if line.strip())
    logger.info(f"Loaded {len(jargons)} jargons from: {jargon_path}")
    return jargons


def remove_jargons(text, jargons):
    """Remove jargon terms from text with word boundary consideration"""
    if not jargons:
        return text
    
    for jargon in jargons:
        text = re.sub(r'\b' + re.escape(jargon) + r'\b', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_full_test_data(full_test_path, text_column):
    """Load test set to exclude from training"""
    if not os.path.exists(full_test_path):
        logger.warning(f"Test data file not found: {full_test_path}")
        return set()
    
    test_df = pd.read_csv(full_test_path)
    test_contents = set(test_df[text_column].astype(str).str.strip())
    logger.info(f"Loaded {len(test_contents):,} test samples")
    return test_contents
