from datetime import datetime
from typing import Literal
Level = Literal["debug","info","warning","error"]
_LEVELS = {"debug":10,"info":20,"warning":30,"error":40}
class Logger:
    def __init__(self, level: str="info"):
        self.level = level if level in _LEVELS else "info"
    def _ok(self, lvl: Level) -> bool:
        return _LEVELS[lvl] >= _LEVELS.get(self.level, 20)
    def _log(self, lvl: Level, msg: str) -> None:
        if not self._ok(lvl): return
        print(f"[{datetime.utcnow().isoformat()}] {lvl.upper()}: {msg}", flush=True)
    def debug(self,msg:str)->None: self._log("debug",msg)
    def info(self,msg:str)->None: self._log("info",msg)
    def warning(self,msg:str)->None: self._log("warning",msg)
    def error(self,msg:str)->None: self._log("error",msg)
