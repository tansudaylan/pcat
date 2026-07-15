import warnings

# Legacy label strings use LaTeX escapes in normal literals. Keep runtime output
# clean by suppressing this specific warning category/message during import.
warnings.filterwarnings(
	'ignore',
	message=r'invalid escape sequence',
	category=SyntaxWarning,
)

from .main import *
