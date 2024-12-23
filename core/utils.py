from typing import Callable, Optional, Any
from functools import wraps


def set_status(status_during: str, status_after: Optional[str] = None, status_error: Optional[str] = None):
    """
    Sets the status of the experiment to the specified value during and after the decorated function execution.
    """
    def decorator(method: Callable):
        @wraps(method)
        def wrapper(self, *args, **kwargs) -> Any:
            # Record status before method execution
            prior_status = None if not hasattr(self, 'status') else self.status
            # Set the status to the status_during
            self.status = status_during
            
            try:
                # Call the original method
                result = method(self, *args, **kwargs)
            except Exception as e:
                # Special status for error
                if status_error:
                    self.status = status_error
                    raise e
                # If an exception occurs and no status_eror provided, set the status to status_after (if provided) or retain the previous status
                self.status = status_after if status_after is not None else prior_status
                # Propagate the exception
                raise e
            else:
                # If no exception occurs, set the status to status_after (if provided) or retain the previous status
                self.status = status_after if status_after is not None else prior_status
            
            return result
        
        return wrapper
    return decorator