# Dataset Scheduler for cross-domain training
# Implements dynamic dataset switching with memory retention

import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from memory_retention import MemoryRetention

class DatasetScheduler:
    """
    Manages cross-domain dataset switching while preserving model memory states.
    Enables smooth transitions between datasets during training.
    """
    def __init__(self, tokenizer, batch_size, schedule_path=None, seq_len=1024, 
                 memory_snapshots_dir=None, memory_adaptation_rate=0.8):
        """
        Initialize the dataset scheduler.
        
        Args:
            tokenizer: The tokenizer to use for encoding text.
            batch_size: Batch size for data loaders
            schedule_path: Path to JSON file containing dataset schedule
            seq_len: Sequence length for tokenization
            memory_snapshots_dir: Directory to store memory snapshots
            memory_adaptation_rate: Rate at which to adapt memory (0-1)
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.current_dataset = None
        self.current_loader = None
        self.current_iter = None
        
        # Load schedule if provided, otherwise use default
        self.schedule = self._load_schedule(schedule_path)
        self.current_schedule_idx = 0
        
        # Initialize memory retention system
        self.memory_retention = MemoryRetention(
            snapshot_dir=memory_snapshots_dir or "memory_snapshots",
            adaptation_rate=memory_adaptation_rate
        )
        
        # Initialize with first dataset in schedule
        self._switch_dataset(self.schedule[0])
        
    def _load_schedule(self, path):
        """Load dataset schedule from JSON file or use default"""
        if path and Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Default schedule if none provided
            return [
                {"step": 0, "dataset": "wikitext", "config": "wikitext-103-raw-v1", "split": "train[:5%]"},
                {"step": 50_000, "dataset": "roneneldan/TinyStories", "split": "train[:5%]"},
                {"step": 100_000, "dataset": "wikitext", "config": "wikitext-103-raw-v1", "split": "train[5%:10%]"},
            ]
    
    def _switch_dataset(self, dataset_info):
        """Switch to a new dataset based on info dict"""
        dataset_name = dataset_info["dataset"]
        dataset_config = dataset_info.get("config", None)
        dataset_split = dataset_info.get("split", "train")
        
        print(f"🔄 Switching to dataset: {dataset_name} ({dataset_split})")
        
        # Load and process the new dataset
        if dataset_name == "wikitext":
            self.current_loader = self._get_wikitext_loader(dataset_config, dataset_split)
        elif dataset_name == "roneneldan/TinyStories":
            self.current_loader = self._get_tinystories_loader(dataset_split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.current_dataset = dataset_name
        self.current_iter = iter(self.current_loader)
        
        return self.current_loader
    
    def _tokenize_and_chunk(self, examples):
        """Tokenize and chunk text examples"""
        from itertools import chain
        
        ids = self.tokenizer(examples["text"], add_special_tokens=False)["input_ids"]
        flat = list(chain.from_iterable(ids))
        
        # Chunk into sequences of seq_len
        result = {"input_ids": [flat[i:i+self.seq_len] for i in range(0, len(flat), self.seq_len)]}
        
        return result
    
    def _get_wikitext_loader(self, config, split):
        """Create a data loader for WikiText dataset"""
        ds = load_dataset("wikitext", config, split=split)
        ds = ds.map(self._tokenize_and_chunk, batched=True, remove_columns=["text"])
        ds = ds.filter(lambda e: len(e["input_ids"]) == self.seq_len)
        ds = ds.with_format("torch")
        
        return DataLoader(ds, batch_size=self.batch_size, shuffle=split.startswith("train"))
    
    def _get_tinystories_loader(self, split):
        """Create a data loader for TinyStories dataset"""
        ds = load_dataset("roneneldan/TinyStories", split=split)
        ds = ds.map(self._tokenize_and_chunk, batched=True, remove_columns=["text"])
        ds = ds.filter(lambda e: len(e["input_ids"]) == self.seq_len)
        ds = ds.with_format("torch")
        
        return DataLoader(ds, batch_size=self.batch_size, shuffle=split.startswith("train"))
    
    def check_schedule(self, step, model=None):
        """
        Check if we need to switch datasets based on current step.
        
        Args:
            step: Current training step
            model: Model with memory states to snapshot and adapt
            
        Returns:
            True if dataset was switched, False otherwise.
        """
        if self.current_schedule_idx + 1 >= len(self.schedule):
            return False  # No more datasets in schedule
            
        next_schedule = self.schedule[self.current_schedule_idx + 1]
        if step >= next_schedule["step"]:
            # Before switching, take memory snapshot of current dataset if model provided
            if model is not None and hasattr(model, 'G') and hasattr(model, 'C'):
                current_dataset_name = self.schedule[self.current_schedule_idx]["dataset"]
                self.memory_retention.take_snapshot(model, current_dataset_name, step)
                print(f"📸 Memory snapshot taken for {current_dataset_name} at step {step}")
            
            # Proceed with dataset switch
            self.current_schedule_idx += 1
            next_dataset = next_schedule
            self._switch_dataset(next_dataset)
            
            # After switching, adapt memory for new dataset if model provided
            if model is not None and hasattr(model, 'G') and hasattr(model, 'C'):
                new_dataset_name = next_dataset["dataset"]
                adaptation_stats = self.memory_retention.adapt_memory_for_transition(
                    model, new_dataset_name, step
                )
                
                if adaptation_stats["adapted"]:
                    print(f"🧠 Memory adapted for transition to {new_dataset_name} at step {step}")
                    print(f"   G change: {adaptation_stats['G_change_norm']:.4f}, "
                          f"C change: {adaptation_stats['C_change_norm']:.4f}")
            
            return True
            
        return False
        
    def visualize_memory_transitions(self, save_path=None):
        """Visualize memory transitions across datasets"""
        return self.memory_retention.visualize_memory_transitions(save_path)
    
    def get_next_batch(self):
        """Get next batch, reinitializing iterator if necessary"""
        try:
            return next(self.current_iter)
        except StopIteration:
            self.current_iter = iter(self.current_loader)
            return next(self.current_iter)
    
    def snapshot_memory(self, model, step):
        """
        Create a snapshot of model memory states using the MemoryRetention system
        
        Args:
            model: The model with memory states
            step: Current training step
            
        Returns:
            Path to saved snapshot
        """
        # Use memory retention system to take snapshot
        dataset_name = self.schedule[self.current_schedule_idx]["dataset"]
        snapshot_path = self.memory_retention.take_snapshot(model, dataset_name, step)
        
        if snapshot_path:
            print(f"📸 Memory snapshot saved for {dataset_name} at step {step}: {snapshot_path}")
            return snapshot_path
        
        return None
