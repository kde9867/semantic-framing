import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from model.preprocessing import load_jargons, remove_jargons, load_full_test_data
import os
import argparse
from datetime import datetime
import logging
import re
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_quarter_label(date):
    """Convert date to quarter label (e.g., 2020Q1)"""
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}Q{quarter}"

def split_data_by_quarters(df, date_column, start_date, end_date):
    """Split data by quarters"""
    if date_column not in df.columns:
        logger.error(f"Date column '{date_column}' not found")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise KeyError(f"Date column '{date_column}' does not exist")
    
    df[date_column] = pd.to_datetime(df[date_column])
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df_filtered = df[(df[date_column] >= start) & (df[date_column] <= end)].copy()
    
    logger.info(f"Date filtering: {start_date} ~ {end_date}")
    logger.info(f"Before: {len(df):,} → After: {len(df_filtered):,}")
    
    df_filtered['quarter'] = df_filtered[date_column].apply(get_quarter_label)
    
    quarter_data = {}
    for quarter, group in df_filtered.groupby('quarter'):
        quarter_data[quarter] = group.drop(columns=['quarter']).reset_index(drop=True)
    
    return quarter_data

def prepare_quarterly_datasets(data_path, jargon_path, text_column, date_column,
                               start_date, end_date, full_test_contents, 
                               val_ratio=0.1, seed=42, save_dir=None):
    """Split data by quarters and prepare train/val sets"""
    logger.info("="*80)
    logger.info(f"Preparing quarterly data: {data_path}")
    logger.info(f"Period: {start_date} ~ {end_date}")
    logger.info("="*80)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded: {len(df):,} samples")
    
    if date_column not in df.columns:
        logger.error(f"Date column '{date_column}' not found")
        date_like_columns = [col for col in df.columns if any(keyword in col.lower() 
                            for keyword in ['date', 'time', 'created'])]
        if date_like_columns:
            logger.info(f"Suggested date columns: {date_like_columns}")
        raise KeyError(f"Date column '{date_column}' does not exist")
    
    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' does not exist")
    
    jargons = load_jargons(jargon_path)
    if jargons:
        df[text_column] = df[text_column].apply(lambda x: remove_jargons(str(x), jargons) if pd.notna(x) else x)
    
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != '']
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_column], keep='first')
    logger.info(f"Removed {before_dedup - len(df):,} duplicates → {len(df):,} unique")
    
    if full_test_contents:
        before_count = len(df)
        df = df[~df[text_column].astype(str).str.strip().isin(full_test_contents)]
        logger.info(f"Excluded {before_count - len(df):,} test samples → {len(df):,}")
    
    quarter_data = split_data_by_quarters(df, date_column, start_date, end_date)
    
    quarters_sorted = sorted(quarter_data.keys())
    logger.info(f"\nTotal {len(quarters_sorted)} quarters:")
    for quarter in quarters_sorted:
        logger.info(f"  {quarter}: {len(quarter_data[quarter]):,} samples")
    
    quarterly_splits = {}
    np.random.seed(seed)
    
    for quarter in quarters_sorted:
        quarter_df = quarter_data[quarter]
        
        indices = np.random.permutation(len(quarter_df))
        val_size = int(len(quarter_df) * val_ratio)
        
        train_data = quarter_df.iloc[indices[:-val_size]].reset_index(drop=True)
        val_data = quarter_df.iloc[indices[-val_size:]].reset_index(drop=True)
        
        quarterly_splits[quarter] = {'train': train_data, 'val': val_data}
        logger.info(f"{quarter}: Train {len(train_data):,} | Val {len(val_data):,}")
        
        if save_dir:
            quarter_dir = os.path.join(save_dir, 'quarterly_data', quarter)
            os.makedirs(quarter_dir, exist_ok=True)
            train_data.to_csv(os.path.join(quarter_dir, 'train.csv'), index=False)
            val_data.to_csv(os.path.join(quarter_dir, 'val.csv'), index=False)
    
    return quarterly_splits

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
    
    return tokenizer

def get_keyword_token_ids(tokenizer, keywords):
    """Extract token IDs from keyword list"""
    keyword_ids = set()
    for kw in keywords:
        tokens = tokenizer.encode(kw, add_special_tokens=False)
        keyword_ids.update(tokens)
    return keyword_ids

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
        
        logger.info(f"Created {len(self.chunks):,} chunks")
    
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
        
        keyword_positions = [i for i in maskable_positions 
                            if input_ids[i].item() in self.keyword_token_ids]
        
        for pos in keyword_positions:
            masked_input_ids[pos] = self.tokenizer.mask_token_id
        
        total_mask_target = max(1, int(len(maskable_positions) * 0.15))
        remaining_masks = total_mask_target - len(keyword_positions)
        
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

def train_quarterly_models(range_name, data_path, jargon_path, date_column,
                          start_date, end_date, model_name, all_keywords, 
                          masking_targets, excluded_keywords, args, device, 
                          full_test_contents):
    """Train models for each quarter"""
    logger.info(f"\nTraining {range_name} quarterly models")
    logger.info(f"Period: {start_date} ~ {end_date}")
    logger.info("="*80)
    
    quarterly_splits = prepare_quarterly_datasets(
        data_path, jargon_path, args.text_column, date_column,
        start_date, end_date, full_test_contents, 
        args.val_ratio, args.seed, args.save_dir
    )
    
    logger.info(f"\nKeywords: {len(all_keywords)} total | {len(excluded_keywords)} excluded | {len(masking_targets)} target")
    
    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_tokenizer = add_new_keywords_only(base_tokenizer, all_keywords)
    
    target_keyword_ids = get_keyword_token_ids(base_tokenizer, masking_targets)
    logger.info(f"Target keyword token IDs: {len(target_keyword_ids)}")
    
    quarterly_results = []
    quarters_sorted = sorted(quarterly_splits.keys())
    
    for quarter in quarters_sorted:
        logger.info(f"\nTraining {range_name} - {quarter}")
        logger.info("="*80)
        
        train_data = quarterly_splits[quarter]['train']
        val_data = quarterly_splits[quarter]['val']
        
        train_texts = train_data[args.text_column].tolist()
        val_texts = val_data[args.text_column].tolist()
        
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer = add_new_keywords_only(tokenizer, all_keywords)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        
        train_dataset = MLMDataset(train_texts, tokenizer, target_keyword_ids, 
                                  args.max_length, args.stride)
        val_dataset = MLMDataset(val_texts, tokenizer, target_keyword_ids, 
                                args.max_length, args.stride)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=args.num_workers, 
                                 pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.num_workers, 
                               pin_memory=True)
        
        total_steps = len(train_loader) * args.epochs
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=args.warmup_steps, 
                                                   num_training_steps=total_steps)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.save_dir, f"{range_name}_{quarter}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        keyword_config = {
            'range': range_name,
            'quarter': quarter,
            'all_keywords': all_keywords,
            'excluded_keywords': list(excluded_keywords),
            'masking_targets': masking_targets,
            'train_size': len(train_data),
            'val_size': len(val_data)
        }
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(keyword_config, f, ensure_ascii=False, indent=2)
        
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            val_loss = validate(model, val_loader, device)
            
            logger.info(f"{quarter} Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
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
        
        logger.info(f"{quarter} completed. Best Val Loss: {best_val_loss:.4f}")
        
        quarterly_results.append({
            'quarter': quarter,
            'save_dir': save_dir,
            'best_val_loss': best_val_loss
        })
    
    return quarterly_results

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    full_test_contents = load_full_test_data(args.test_data_path, args.text_column)
    
    all_keywords = [
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
        '태아', '전문의', '젖', '산후조리원', '의학', '아이들', '태교', '진료', '소아과', '소아청소년과', 
        '유방', '속싸개', '난임', '우유', '육아브이로그', '태교음악', '유방외과', '육아클래스', '국제모유수유전문가', '유방이야기', 
        '쌍둥이오빠'
    ]
    
    excluded_keywords = {
        '소아청소년과', '유방', '속싸개', '난임', '우유', '육아브이로그', '태교음악', 
        '유방외과', '육아클래스', '국제모유수유전문가', '유방이야기', '쌍둥이오빠'
    }
    
    masking_targets = [kw for kw in all_keywords if kw not in excluded_keywords]
    
    logger.info(f"Keywords: {len(all_keywords)} total | {len(excluded_keywords)} excluded | {len(masking_targets)} target")
    
    ranges = [
        {
            'name': 'range1',
            'data_path': args.data_path1,
            'jargon_path': args.jargon_path1,
            'date_column': args.date_column,
            'start_date': '2016-11-01',
            'end_date': '2019-11-01'
        },
        {
            'name': 'range2',
            'data_path': args.data_path2,
            'jargon_path': args.jargon_path2,
            'date_column': args.date_column,
            'start_date': '2020-01-20',
            'end_date': '2023-01-17'
        },
        {
            'name': 'range3',
            'data_path': args.data_path3,
            'jargon_path': args.jargon_path3,
            'date_column': args.date_column,
            'start_date': '2023-05-05',
            'end_date': '2025-04-05'
        }
    ]
    
    all_results = {}
    for range_info in ranges:
        quarterly_results = train_quarterly_models(
            range_name=range_info['name'],
            data_path=range_info['data_path'],
            jargon_path=range_info['jargon_path'],
            date_column=range_info['date_column'],
            start_date=range_info['start_date'],
            end_date=range_info['end_date'],
            model_name=args.model_name,
            all_keywords=all_keywords,
            masking_targets=masking_targets,
            excluded_keywords=excluded_keywords,
            args=args,
            device=device,
            full_test_contents=full_test_contents
        )
        all_results[range_info['name']] = quarterly_results
    
    logger.info("\nAll training completed")
    for range_name, quarters in all_results.items():
        logger.info(f"\n{range_name}:")
        for q in quarters:
            logger.info(f"  {q['quarter']}: {q['save_dir']} (Best: {q['best_val_loss']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    UNIQUE_DIR = '.path/semantic-framing/data/period_data'
    
    parser.add_argument('--test_data_path', type=str, 
                        default='.path/semantic-framing/data/all_ranges_testcsv')
    parser.add_argument('--data_path1', type=str, default=os.path.join(UNIQUE_DIR, 'pre_COVID.csv'))
    parser.add_argument('--data_path2', type=str, default=os.path.join(UNIQUE_DIR, 'during_COVID.csv'))
    parser.add_argument('--data_path3', type=str, default=os.path.join(UNIQUE_DIR, 'post_COVID.csv'))
    parser.add_argument('--jargon_path1', type=str, default='.path/semantic-framing/data/range_1_jargons.txt')
    parser.add_argument('--jargon_path2', type=str, default='.path/semantic-framing/datarange_2_jargons.txt')
    parser.add_argument('--jargon_path3', type=str, default='.path/semantic-framing/data/range_3_jargons.txt')
    parser.add_argument('--text_column', type=str, default='content')
    parser.add_argument('--date_column', type=str, default='date')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='klue/bert-base')
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