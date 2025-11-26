import logging
import os
import coloredlogs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['COLOREDLOGS_LOG_FORMAT']='[%(asctime)s] %(message)s'
logger = logging.getLogger(__name__)
# logger = logging.getLogger("numba"); logger.setLevel(logging.ERROR)
# logging.disable(logging.WARNING)
coloredlogs.install(level='INFO')
coloredlogs.install(level='INFO', logger=logger)