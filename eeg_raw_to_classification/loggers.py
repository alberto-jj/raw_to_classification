"""Module dealing with logging related functionality and settings"""

import os
import logging
import sys
import json
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

def _excepthook(*args):
    """Catch Exceptions to logger.
    
    Notes
    -----
    See https://code.activestate.com/recipes/577074-logging-asserts/
    """
    logging.getLogger().error('Uncaught exception:', exc_info=args)

sys.excepthook = _excepthook # See _excepthook documentation


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['ENDC'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info'):
                log_data[key] = value
        
        return json.dumps(log_data)


class StructuredLogger:
    """Enhanced logger with structured logging and progress tracking"""
    
    def __init__(self, name: str, log_dir: Path, debug: bool = False):
        self.logger = self._setup_logger(name, log_dir, debug)
        self.metrics = {}
        
    def _setup_logger(self, name: str, log_dir: Path, debug: bool):
        """Setup logger with multiple handlers"""
        logger = logging.getLogger(name)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Set level
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Console handler with color coding
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for all logs
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Structured JSON handler for metrics
        json_handler = logging.FileHandler(log_dir / f"{name}_metrics.jsonl")
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(logging.INFO)
        logger.addHandler(json_handler)
        
        # Error-only handler
        error_handler = logging.FileHandler(log_dir / f"{name}_errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    @contextmanager
    def timed_operation(self, operation_name: str, **metadata):
        """Context manager for timing operations"""
        start_time = time.time()
        self.logger.info(f"Starting {operation_name}", extra={
            'operation': operation_name,
            'event': 'start',
            **metadata
        })
        
        try:
            yield
            duration = time.time() - start_time
            self.logger.info(f"Completed {operation_name} in {duration:.2f}s", extra={
                'operation': operation_name,
                'event': 'success',
                'duration': duration,
                **metadata
            })
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Failed {operation_name} after {duration:.2f}s: {e}", extra={
                'operation': operation_name,
                'event': 'error',
                'duration': duration,
                'error': str(e),
                **metadata
            })
            raise
    
    def log_progress(self, current: int, total: int, item_name: str = "items"):
        """Log progress with percentage"""
        percentage = (current / total) * 100
        self.logger.info(f"Progress: {current}/{total} {item_name} ({percentage:.1f}%)", extra={
            'event': 'progress',
            'current': current,
            'total': total,
            'percentage': percentage
        })
    
    def log_metrics(self, metrics: Dict[str, Any], context: str = ""):
        """Log structured metrics"""
        self.logger.info(f"Metrics{' - ' + context if context else ''}", extra={
            'event': 'metrics',
            'context': context,
            **metrics
        })


def setup_pipeline_logging(pipeline_name: str, log_dir: Path, debug: bool = False) -> StructuredLogger:
    """Setup logging for a pipeline"""
    return StructuredLogger(pipeline_name, log_dir, debug)


def setup_logging(log_file=None, debug=False):
    """Setup the logging (legacy function for backward compatibility)
    Parameters
    ----------
    log_file: str
        Name of the logfile
    debug: bool
        Set log level to DEBUG if debug==True
    Returns
    -------
    logging.logger:
        The logger.
    
    Notes
    -----
    This function is a copy of the one found in bidscoin.
    https://github.com/Donders-Institute/bidscoin/blob/748ea2ba537b06d8eee54ac7217b909bdf91a812/bidscoin/bidscoin.py#L41-L83
    """
    currentDT = datetime.now()
    currentDT.strftime("%Y-%m-%d %H:%M:%S")

    noDate = True
    # Get the root logger
    logger = logging.getLogger()

    # Set the format and logging level
    if debug:
        fmt = '%(asctime)s - %(name)s - %(levelname)s | %(message)s'
        logger.setLevel(logging.DEBUG)
    else:
        fmt = '%(asctime)s - %(levelname)s | %(message)s'
        logger.setLevel(logging.INFO)
    datefmt   = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Set & add the streamhandler and add some color to those boring terminal logs! :-)
    #coloredlogs.install(level=logger.level, fmt=fmt, datefmt=datefmt)

    if not log_file:
        return logger

    # Set & add the log filehandler
    logdir,log_name = os.path.split(log_file)
    os.makedirs(logdir,exist_ok=True) # Create the log dir if it does not exist
    log_name = os.path.join(logdir,currentDT.strftime("%Y-%m-%d__%H_%M_%S") + '__' + log_name)
    if noDate:
        log_name=log_file
    loghandler = logging.FileHandler(log_name)
    loghandler.setLevel(logging.DEBUG)
    loghandler.setFormatter(formatter)
    loghandler.set_name('loghandler')
    logger.addHandler(loghandler)

    # Set & add the error / warnings handler
    error_file = log_name +'.errors'            # Derive the name of the error logfile from the normal log_file
    errorhandler = logging.FileHandler(error_file, mode='w')
    errorhandler.setLevel(logging.WARNING)
    errorhandler.setFormatter(formatter)
    errorhandler.set_name('errorhandler')
    logger.addHandler(errorhandler)
    return logger