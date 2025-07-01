"""
Caching and validation system for preprocessing targets.
"""
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class PreprocessingTargetCache:
    """
    Cache and validate preprocessing targets to avoid regenerating them repeatedly.
    """
    
    def __init__(self, cache_dir: str, source_name: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_name = source_name
        self.cache_file = self.cache_dir / f"{source_name}_targets.json"
    
    def get_cache_key(self, stage: str, year_range: tuple = None, **kwargs) -> str:
        """Generate cache key for target list."""
        key_data = {
            'stage': stage,
            'year_range': year_range,
            'source': self.source_name,
            **kwargs
        }
        return hashlib.md5(str(sorted(key_data.items())).encode()).hexdigest()
    
    def get_cached_targets(self, cache_key: str) -> List[Dict[str, Any]]:
        """Retrieve cached targets if they exist and are valid."""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            if cache_key in cache_data:
                entry = cache_data[cache_key]
                # Check if cache is still valid (e.g., less than 1 hour old)
                cache_time = datetime.fromisoformat(entry['timestamp'])
                if (datetime.now() - cache_time).seconds < 3600:
                    return entry['targets']
        except Exception:
            pass
        
        return None
    
    def cache_targets(self, cache_key: str, targets: List[Dict[str, Any]]):
        """Cache the generated targets."""
        try:
            # Load existing cache
            cache_data = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
            
            # Add new entry
            cache_data[cache_key] = {
                'targets': targets,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save cache
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            # Cache failures shouldn't break the workflow
            print(f"Warning: Failed to cache targets: {e}")
    
    def validate_targets(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate targets against current file system state."""
        valid_targets = []
        
        for target in targets:
            # Check if output already exists and is complete
            output_path = target['output_path']
            if os.path.exists(output_path):
                # Could add more sophisticated validation here
                target['status'] = 'completed'
            else:
                # Check if dependencies exist
                all_deps_exist = True
                for dep in target.get('dependencies', []):
                    if not os.path.exists(dep):
                        all_deps_exist = False
                        break
                
                if all_deps_exist:
                    target['status'] = 'ready'
                else:
                    target['status'] = 'waiting'
            
            valid_targets.append(target)
        
        return valid_targets
