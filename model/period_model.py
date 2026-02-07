import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_full_test_data(full_test_path, text_column):
    """Load test set to exclude from training"""
    if not os.path.exists(full_test_path):
        logger.warning(f"Test data file not found: {full_test_path}")
        return set()
    
    test_df = pd.read_csv(full_test_path)
    test_contents = set(test_df[text_column].astype(str).str.strip())
    logger.info(f"Loaded {len(test_contents):,} test samples")
    return test_contents

def prepare_single_dataset(data_path, jargon_path, text_column, full_test_contents, val_ratio=0.1, seed=42):
    """Load data, remove jargons, exclude test set, and split into train/val"""
    logger.info("="*60)
    logger.info(f"Preparing data: {data_path}")
    logger.info("="*60)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded: {len(df):,} samples")
    
    jargons = load_jargons(jargon_path)
    if jargons:
        df[text_column] = df[text_column].apply(lambda x: remove_jargons(str(x), jargons) if pd.notna(x) else x)
    
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != '']
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_column], keep='first')
    logger.info(f"Removed {before_dedup - len(df):,} duplicates → {len(df):,} unique samples")
    
    if full_test_contents:
        before_count = len(df)
        df = df[~df[text_column].astype(str).str.strip().isin(full_test_contents)]
        logger.info(f"Excluded {before_count - len(df):,} test samples → {len(df):,} samples")
    
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    val_size = int(len(df) * val_ratio)
    
    train_data = df.iloc[indices[:-val_size]].reset_index(drop=True)
    val_data = df.iloc[indices[-val_size:]].reset_index(drop=True)
    
    logger.info(f"Split: Train {len(train_data):,} | Val {len(val_data):,}\n")
    return train_data, val_data

def add_new_keywords_only(tokenizer, keywords):
    """Add only keywords that are not already in vocabulary"""
    original_vocab_size = len(tokenizer)
    
    new_keywords = []
    existing_keywords = []
    
    for kw in keywords:
        tokens = tokenizer.encode(kw, add_special_tokens=False)
        if len(tokens) == 1:
            existing_keywords.append(kw)
        else:
            new_keywords.append(kw)
    
    logger.info(f"Keywords: {len(keywords)} total | {len(existing_keywords)} existing | {len(new_keywords)} new")
    
    if new_keywords:
        tokenizer.add_tokens(new_keywords, special_tokens=False)
        logger.info(f"Vocab size: {original_vocab_size:,} → {len(tokenizer):,}")
    
    all_keyword_ids = set()
    for kw in keywords:
        tokens = tokenizer.encode(kw, add_special_tokens=False)
        all_keyword_ids.update(tokens)
    
    return tokenizer, all_keyword_ids

class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, keyword_token_ids, max_length=512, stride=256):
        self.tokenizer = tokenizer
        self.keyword_token_ids = keyword_token_ids
        self.max_length = max_length
        self.stride = stride
        self.chunks = []
        
        for text in tqdm(texts, desc="Creating chunks"):
            tokens = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
            
            if len(tokens) > max_length - 2:
                for start in range(0, len(tokens), stride):
                    end = min(start + max_length - 2, len(tokens))
                    chunk = [tokenizer.cls_token_id] + tokens[start:end] + [tokenizer.sep_token_id]
                    self.chunks.append(chunk)
                    if end >= len(tokens):
                        break
            else:
                chunk = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
                self.chunks.append(chunk)
        
        logger.info(f"Created {len(self.chunks):,} chunks\n")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        token_ids = self.chunks[idx]
        
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(self.chunks[idx]) + [0] * padding_length
        else:
            token_ids = token_ids[:self.max_length]
            attention_mask = [1] * self.max_length
        
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        labels, masked_input_ids = self.mask_tokens(input_ids)
        
        return {'input_ids': masked_input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        
        maskable_positions = [i for i, tid in enumerate(input_ids) 
                             if tid.item() not in special_tokens]
        
        if not maskable_positions:
            return labels, masked_input_ids
        
        # Mask all keyword positions
        keyword_positions = [i for i in maskable_positions 
                            if input_ids[i].item() in self.keyword_token_ids]
        
        for pos in keyword_positions:
            masked_input_ids[pos] = self.tokenizer.mask_token_id
        
        # Calculate remaining masks needed (15% total)
        total_mask_target = max(1, int(len(maskable_positions) * 0.15))
        remaining_masks = total_mask_target - len(keyword_positions)
        
        # Mask additional random tokens
        if remaining_masks > 0:
            non_keyword_positions = [i for i in maskable_positions 
                                    if i not in keyword_positions]
            
            if non_keyword_positions:
                num_to_mask = min(remaining_masks, len(non_keyword_positions))
                random_mask_positions = np.random.choice(non_keyword_positions, num_to_mask, replace=False)
                
                for pos in random_mask_positions:
                    prob = np.random.random()
                    if prob < 0.8:
                        masked_input_ids[pos] = self.tokenizer.mask_token_id
                    elif prob < 0.9:
                        masked_input_ids[pos] = np.random.randint(0, self.tokenizer.vocab_size)
        else:
            random_mask_positions = []
        
        all_masked = set(keyword_positions) | set(random_mask_positions)
        for i in range(len(labels)):
            if i not in all_masked:
                labels[i] = -100
        
        return labels, masked_input_ids

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc=f'Epoch {epoch+1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            total_loss += loss.item()
    
    return total_loss / len(loader)

def train_single_model(data_path, jargon_path, model_name, keywords, args, device, range_name, full_test_contents):
    """Train a single model on one dataset"""
    logger.info(f"\nTraining {range_name} model")
    logger.info("="*80)
    
    train_data, val_data = prepare_single_dataset(
        data_path, jargon_path, args.text_column, full_test_contents, args.val_ratio, args.seed
    )
    
    train_texts = train_data[args.text_column].tolist()
    val_texts = val_data[args.text_column].tolist()
    
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    keyword_list = [kw.strip() for kw in keywords] if isinstance(keywords, list) else [kw.strip() for kw in keywords.split(',')]
    tokenizer, keyword_token_ids = add_new_keywords_only(tokenizer, keyword_list)
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    train_dataset = MLMDataset(train_texts, tokenizer, keyword_token_ids, args.max_length, args.stride)
    val_dataset = MLMDataset(val_texts, tokenizer, keyword_token_ids, args.max_length, args.stride)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
                                               num_training_steps=total_steps)
    
    save_dir = os.path.join(args.save_dir, f"{range_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'best_model')
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
        
        epoch_path = os.path.join(save_dir, f'epoch_{epoch+1}')
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)
    
    final_path = os.path.join(save_dir, 'final_model')
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"{range_name} training completed. Best Val Loss: {best_val_loss:.4f}")
    return save_dir

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    full_test_contents = load_full_test_data(args.test_data_path, args.text_column)
    
    datasets = [
        {'name': 'range1', 'data_path': args.data_path1, 'jargon_path': args.jargon_path1},
        {'name': 'range2', 'data_path': args.data_path2, 'jargon_path': args.jargon_path2},
        {'name': 'range3', 'data_path': args.data_path3, 'jargon_path': args.jargon_path3}
    ]
    
    results = []
    for dataset_info in datasets:
        save_dir = train_single_model(
            data_path=dataset_info['data_path'],
            jargon_path=dataset_info['jargon_path'],
            model_name=args.model_name,
            keywords=args.keywords,
            args=args,
            device=device,
            range_name=dataset_info['name'],
            full_test_contents=full_test_contents
        )
        results.append({'range': dataset_info['name'], 'save_dir': save_dir})
    
    logger.info("\nAll training completed")
    for result in results:
        logger.info(f"{result['range']}: {result['save_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    default_keywords = [
    '출산', '육아', '아이', '아기', '임신', '엄마', '결혼', '애', '남편', '신생아', 
    '병원', '저출산', '아빠', '지원', '산모', '정책', '개월', '분만', '아가', '자녀', 
    '임산부', '돌', '둘째', '부모', '인구', '조리원', '부부', '가족', '수유', '첫째', 
    '정부', '애기', '기저귀', '출산율', '교육', '분유', '가정', '검사', '딸', '여성', 
    '산후', '휴직', '육아휴직', '젖병', '태동', '출생', '아들', '혼인', '초음파', '부모님', 
    '입원', '진통', '태명', '자연분만', '제왕절개', '통증', '수술', '생후', '신혼부부', '물티슈', 
    '이유식', '비혼', '아내', '통계', '제도', '고령화', '양육', '연애', '출생아', '애들', 
    '남성', '내진', '통계청', '복지', '가사', '아동', '출산률', '태동검사', '산부인과', '돌봄', 
    '유도분만', '독박육아', '주택', '저출생', '합계출산율', '가구', '주거', '출생신고', '맞벌이', '혼인신고', 
    '집안일', '쌍둥이', '촉진제', '이혼', '모유', '어린이집', '모유수유', '의사', '산후조리', '유치원', 
    '태아', '전문의', '젖', '산후조리원', '의학', '아이들', '태교', '진료', '소아과'
]
    
    UNIQUE_DIR = '.path/semantic-framing/data/period_data'
    
    parser.add_argument('--test_data_path', type=str, 
                        default='.path/semantic-framing/data/all_ranges_test.csv')
    parser.add_argument('--data_path1', type=str, default=os.path.join(UNIQUE_DIR, 'pre_COVID.csv'))
    parser.add_argument('--data_path2', type=str, default=os.path.join(UNIQUE_DIR, 'during_COVID.csv'))
    parser.add_argument('--data_path3', type=str, default=os.path.join(UNIQUE_DIR, 'post_COVID.csv'))
    parser.add_argument('--jargon_path1', type=str, default='.path/semantic-framing/data/range_1_jargons.txt')
    parser.add_argument('--jargon_path2', type=str, default='.path/semantic-framing/data/range_2_jargons.txt')
    parser.add_argument('--jargon_path3', type=str, default='.path/semantic-framing/data/range_3_jargons.txt')
    parser.add_argument('--text_column', type=str, default='content')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--keywords', type=list, default=default_keywords)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=216)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
