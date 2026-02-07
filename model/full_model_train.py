import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from model.preprocessing import load_jargons, remove_jargons
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import logging
import re
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_process_data(train_path, val_path, test_path, jargon_paths, text_column):
    """Load pre-split data and remove jargon terms"""
    logger.info("Loading data and removing jargons")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # Load and merge all jargons
    all_jargons = set()
    for jargon_path in jargon_paths:
        jargons = load_jargons(jargon_path)
        all_jargons.update(jargons)
    
    logger.info(f"Total jargons: {len(all_jargons)}")
    
    # Apply jargon removal
    if all_jargons:
        train_df[text_column] = train_df[text_column].apply(
            lambda x: remove_jargons(str(x), all_jargons) if pd.notna(x) else x
        )
        val_df[text_column] = val_df[text_column].apply(
            lambda x: remove_jargons(str(x), all_jargons) if pd.notna(x) else x
        )
        test_df[text_column] = test_df[text_column].apply(
            lambda x: remove_jargons(str(x), all_jargons) if pd.notna(x) else x
        )
    
    # Remove null and empty strings
    train_df = train_df.dropna(subset=[text_column])
    train_df = train_df[train_df[text_column].str.strip() != '']
    val_df = val_df.dropna(subset=[text_column])
    val_df = val_df[val_df[text_column].str.strip() != '']
    test_df = test_df.dropna(subset=[text_column])
    test_df = test_df[test_df[text_column].str.strip() != '']
    
    logger.info(f"After filtering - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    return train_df, val_df, test_df


def add_new_keywords_only(tokenizer, keywords):
    """Add only keywords that are not in vocabulary"""
    original_vocab_size = len(tokenizer)
    
    # Identify new keywords not in vocabulary
    new_keywords = []
    existing_keywords = []
    
    for kw in keywords:
        tokens = tokenizer.encode(kw, add_special_tokens=False)
        
        if len(tokens) == 1:
            # Already exists as single token
            existing_keywords.append(kw)
        else:
            # Split into subwords = not in vocab
            new_keywords.append(kw)
    
    logger.info(f"Keywords - Total: {len(keywords)}, Existing: {len(existing_keywords)}, New: {len(new_keywords)}")
    
    # Add only new keywords
    if new_keywords:
        tokenizer.add_tokens(new_keywords, special_tokens=False)
        new_vocab_size = len(tokenizer)
        logger.info(f"Vocab size: {original_vocab_size:,} → {new_vocab_size:,}")
    
    return tokenizer


def get_keyword_token_ids(tokenizer, keywords):
    """Extract token IDs from keyword list"""
    keyword_ids = set()
    for kw in keywords:
        tokens = tokenizer.encode(kw, add_special_tokens=False)
        keyword_ids.update(tokens)
    return keyword_ids


class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling with sliding window chunking"""
    
    def __init__(self, texts, tokenizer, keyword_token_ids, max_length=512, stride=256):
        self.tokenizer = tokenizer
        self.keyword_token_ids = keyword_token_ids
        self.max_length = max_length
        self.stride = stride
        self.chunks = []
        
        # Create chunks with sliding window
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
        
        logger.info(f"Total chunks: {len(self.chunks):,}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        token_ids = self.chunks[idx]
        
        # Padding
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(self.chunks[idx]) + [0] * padding_length
        else:
            token_ids = token_ids[:self.max_length]
            attention_mask = [1] * self.max_length
        
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Apply masking
        labels, masked_input_ids = self.mask_tokens(input_ids)
        
        return {'input_ids': masked_input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def mask_tokens(self, input_ids):
        """
        15% total masking strategy:
        1. Mask 100% of target keywords
        2. Random mask remaining tokens to reach 15% total
        """
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        
        # Get maskable positions (exclude special tokens)
        maskable_positions = [i for i, tid in enumerate(input_ids) 
                             if tid.item() not in special_tokens]
        
        # 1. Find keyword positions (100% masking)
        keyword_positions = [i for i in maskable_positions
                           if input_ids[i].item() in self.keyword_token_ids]
        
        # Mask all keywords
        for pos in keyword_positions:
            masked_input_ids[pos] = self.tokenizer.mask_token_id
        
        # 2. Calculate target: 15% of total maskable tokens
        total_maskable = len(maskable_positions)
        target_mask_count = max(1, int(total_maskable * 0.15))
        
        # Already masked keywords
        keyword_masked_count = len(keyword_positions)
        
        # Calculate remaining masks needed
        remaining_mask_count = max(0, target_mask_count - keyword_masked_count)
        
        # 3. Random masking of non-keyword tokens
        non_keyword_positions = [i for i in maskable_positions
                                if i not in keyword_positions]
        
        random_mask_positions = []
        if remaining_mask_count > 0 and non_keyword_positions:
            actual_mask_count = min(remaining_mask_count, len(non_keyword_positions))
            random_mask_positions = np.random.choice(
                non_keyword_positions, 
                actual_mask_count, 
                replace=False
            ).tolist()
            
            for pos in random_mask_positions:
                prob = np.random.random()
                if prob < 0.8:  # 80%: [MASK]
                    masked_input_ids[pos] = self.tokenizer.mask_token_id
                elif prob < 0.9:  # 10%: random token
                    masked_input_ids[pos] = np.random.randint(0, self.tokenizer.vocab_size)
                # 10%: keep original
        
        # Set non-masked tokens to -100 (exclude from loss)
        all_masked_positions = set(keyword_positions) | set(random_mask_positions)
        for i in range(len(labels)):
            if i not in all_masked_positions:
                labels[i] = -100
        
        return labels, masked_input_ids


def compute_masking_statistics(dataset, num_samples=1000):
    """Compute masking statistics by sampling from dataset"""
    total_maskable = 0
    total_keyword_masked = 0
    total_random_masked = 0
    
    sample_size = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    special_tokens = {dataset.tokenizer.cls_token_id, dataset.tokenizer.sep_token_id, dataset.tokenizer.pad_token_id}
    
    for idx in tqdm(indices, desc="Computing masking stats"):
        token_ids = dataset.chunks[idx]
        
        # Padding
        padding_length = dataset.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [dataset.tokenizer.pad_token_id] * padding_length
        else:
            token_ids = token_ids[:dataset.max_length]
        
        # Count maskable tokens
        maskable_positions = [i for i, tid in enumerate(token_ids) 
                             if tid not in special_tokens]
        total_maskable += len(maskable_positions)
        
        # Count keyword tokens
        keyword_positions = [i for i in maskable_positions
                           if token_ids[i] in dataset.keyword_token_ids]
        total_keyword_masked += len(keyword_positions)
        
        # Calculate random masking count
        target_mask_count = max(1, int(len(maskable_positions) * 0.15))
        remaining_mask_count = max(0, target_mask_count - len(keyword_positions))
        
        non_keyword_positions = [i for i in maskable_positions
                                if i not in keyword_positions]
        if remaining_mask_count > 0 and non_keyword_positions:
            actual_mask_count = min(remaining_mask_count, len(non_keyword_positions))
            total_random_masked += actual_mask_count
    
    return {
        'total_maskable': total_maskable,
        'keyword_masked': total_keyword_masked,
        'random_masked': total_random_masked,
        'total_masked': total_keyword_masked + total_random_masked,
        'keyword_ratio': total_keyword_masked / total_maskable * 100 if total_maskable > 0 else 0,
        'random_ratio': total_random_masked / total_maskable * 100 if total_maskable > 0 else 0,
        'total_mask_ratio': (total_keyword_masked + total_random_masked) / total_maskable * 100 if total_maskable > 0 else 0
    }


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
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
    """Validate model"""
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


def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data and remove jargons
    jargon_paths = [args.jargon_path1, args.jargon_path2, args.jargon_path3]
    train_df, val_df, test_df = load_and_process_data(
        args.train_data_path, 
        args.val_data_path, 
        args.test_data_path,
        jargon_paths,
        args.text_column
    )
    
    train_texts = train_df[args.text_column].tolist()
    val_texts = val_df[args.text_column].tolist()
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    
   
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
        '유방', '속싸개', '난임', '우유', '육아브이로그', '태교음악', '유방외과', '육아클래스', '국제모유수유전문가', '유방이야기', '쌍둥이오빠'
    ]
    
    # Keywords to exclude from masking
    excluded_keywords = {
        '소아청소년과', '유방', '속싸개', '난임', '우유', '육아브이로그', '태교음악', 
        '유방외과', '육아클래스', '국제모유수유전문가', '유방이야기', '쌍둥이오빠'
    }
    
    # Target keywords for masking (all_keywords - excluded_keywords)
    masking_targets = [kw for kw in all_keywords if kw not in excluded_keywords]
    
    logger.info(f"Keyword configuration - Total: {len(all_keywords)}, "
                f"Excluded: {len(excluded_keywords)}, Masking targets: {len(masking_targets)}")
    
    tokenizer = add_new_keywords_only(tokenizer, all_keywords)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    target_keyword_ids = get_keyword_token_ids(tokenizer, masking_targets)
    logger.info(f"Masking target keywords: {len(masking_targets)}, Token IDs: {len(target_keyword_ids)}")
    
    train_dataset = MLMDataset(train_texts, tokenizer, target_keyword_ids, args.max_length, args.stride)
    val_dataset = MLMDataset(val_texts, tokenizer, target_keyword_ids, args.max_length, args.stride)
    
    # Compute masking statistics
    logger.info("Computing masking statistics (sample-based)")
    
    train_stats = compute_masking_statistics(train_dataset, num_samples=min(1000, len(train_dataset)))
    logger.info(f"Train - Maskable: {train_stats['total_maskable']:,}, "
                f"Keywords: {train_stats['keyword_masked']:,} ({train_stats['keyword_ratio']:.2f}%), "
                f"Random: {train_stats['random_masked']:,} ({train_stats['random_ratio']:.2f}%), "
                f"Total: {train_stats['total_masked']:,} ({train_stats['total_mask_ratio']:.2f}%)")
    
    val_stats = compute_masking_statistics(val_dataset, num_samples=min(1000, len(val_dataset)))
    logger.info(f"Val - Maskable: {val_stats['total_maskable']:,}, "
                f"Keywords: {val_stats['keyword_masked']:,} ({val_stats['keyword_ratio']:.2f}%), "
                f"Random: {val_stats['random_masked']:,} ({val_stats['random_ratio']:.2f}%), "
                f"Total: {val_stats['total_masked']:,} ({val_stats['total_mask_ratio']:.2f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    
    # Setup optimizer and scheduler
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
                                               num_training_steps=total_steps)
    
    # Create save directory
    save_dir = args.save_dir or f"klue-bert-mlm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save keyword configuration for reproducibility
    keyword_config = {
        'all_keywords': all_keywords,
        'excluded_keywords': list(excluded_keywords),
        'masking_targets': masking_targets
    }
    with open(os.path.join(save_dir, 'keyword_config.json'), 'w', encoding='utf-8') as f:
        json.dump(keyword_config, f, ensure_ascii=False, indent=2)
    logger.info(f"Keyword config saved to: {os.path.join(save_dir, 'keyword_config.json')}")
    
    logger.info(f"Training start | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.learning_rate}")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'best_model')
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info(f"Best model saved (val_loss: {val_loss:.4f})")
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model')
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"Training complete! Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--train_data_path', type=str, 
                       default='.path/semantic-framing/data/all_ranges_train.csv')
    parser.add_argument('--val_data_path', type=str, 
                       default='.path/semantic-framing/data/all_ranges_val.csv')
    parser.add_argument('--test_data_path', type=str, 
                       default='.path/semantic-framing/data/all_ranges_test.csv')
    parser.add_argument('--jargon_path1', type=str, 
                       default='.path/semantic-framing/data/range_1_jargons.txt')
    parser.add_argument('--jargon_path2', type=str, 
                       default='.path/semantic-framing/data/range_2_jargons.txt')
    parser.add_argument('--jargon_path3', type=str, 
                       default='.path/semantic-framing/data/range_3_jargons.txt')
    parser.add_argument('--text_column', type=str, default='content')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=256)
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    
    # Other settings
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
