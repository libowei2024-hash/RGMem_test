# file: l1_memory/memory/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryBase(ABC):


    @abstractmethod
    def add(
            self,
            data: Dict[str, Any],
            *,
            user_id: str,
            role_id: Optional[str] = None,
            # run_id is removed as it's less relevant for long-term KG memory
    ) -> Dict[str, Any]:

        pass

    @abstractmethod
    def search(
            self,
            query: str,
            *,
            user_id: str,
            role_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        pass

    @abstractmethod
    def get_all(
            self,
            *,
            user_id: str,
            role_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        pass

    @abstractmethod
    def delete_all(
            self,
            *,
            user_id: str,
            role_id: Optional[str] = None,
    ) -> Dict[str, str]:

        pass

    @abstractmethod
    def reset(self):

        pass