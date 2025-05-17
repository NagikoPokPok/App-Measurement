import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, history_file: Union[str, Path] = "history.json"):
        """Initialize history manager with file path"""
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> List[Dict]:
        """Load project history from JSON file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return []

    def load_history_filtered(self, 
                            method: Optional[str] = None, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict]:
        """
        Load history with filters
        Args:
            method: Filter by estimation method
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            limit: Limit number of entries
        """
        try:
            history = self.load_history()
            
            # Convert to DataFrame for easier filtering
            df = pd.DataFrame(history)
            
            if not df.empty:
                # Apply filters
                if method:
                    df = df[df['method'] == method]
                
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                
                if end_date:
                    df = df[df['timestamp'] <= end_date]
                
                # Sort by timestamp descending
                df = df.sort_values('timestamp', ascending=False)
                
                # Apply limit
                if limit:
                    df = df.head(limit)
                
                return df.to_dict('records')
            return []
            
        except Exception as e:
            logger.error(f"Error loading filtered history: {e}")
            return []

    def save_history_entry(self, 
                          method: str, 
                          input_data: Dict, 
                          output_data: Dict,
                          tags: Optional[List[str]] = None) -> bool:
        """
        Save a new project entry to history
        Returns: True if successful, False otherwise
        """
        try:
            history = self.load_history()
            
            current_ids = [entry.get('id', 0) for entry in history]
            new_id = max(current_ids, default=0) + 1
            
            new_entry = {
                "id": new_id,  # Đảm bảo mỗi entry có một ID duy nhất
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "input": input_data,
                "output": output_data,
                "tags": tags or [],
                "last_modified": datetime.now().isoformat()
            }
            
            history.append(new_entry)
            
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Successfully saved history entry {new_entry['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving history entry: {e}")
            return False

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a history entry by ID"""
        try:
            history = self.load_history()
            history = [h for h in history if h.get('id') != entry_id]
            
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"Successfully deleted history entry {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting history entry: {e}")
            return False

    def clear_history(self) -> bool:
        """Clear all history"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)
            logger.info("Successfully cleared history")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False

    def export_history(self, format: str = 'json') -> Optional[Union[str, bytes]]:
        """
        Export history in different formats
        Args:
            format: 'json', 'csv', or 'excel'
        """
        try:
            history = self.load_history()
            df = pd.DataFrame(history)
            
            if format == 'json':
                return json.dumps(history, indent=2)
            elif format == 'csv':
                return df.to_csv(index=False)
            elif format == 'excel':
                return df.to_excel(index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            return None