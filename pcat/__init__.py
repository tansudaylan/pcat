import warnings

import tdpy

# Legacy label strings use LaTeX escapes in normal literals. Keep runtime output
# clean by suppressing this specific warning category/message during import.
warnings.filterwarnings(
	'ignore',
	message=r'invalid escape sequence',
	category=SyntaxWarning,
)

if not hasattr(tdpy, 'retr_labltotlsing') and hasattr(tdpy, 'retr_labltotl'):
    tdpy.retr_labltotlsing = tdpy.retr_labltotl

from .main import *
