# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
import multiprocessing
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError,
)
from typing import Any, Callable, Literal

__all__ = ["DataNotReadyError", "TaskQueueFulledError", "OrderedTaskExecutor"]


logger = logging.getLogger(__file__)


class DataNotReadyError(Exception):
    """Exception raised when data is not yet prepared to be retrieved."""

    pass


class TaskQueueFulledError(Exception):
    """Exception when the task queue has reached its maximum capacity."""

    pass


def _get_multiprocess_context(context: str):
    """Get a multiprocessing context.

    Args:
        context (str): The context to use for multiprocessing
            ('fork', 'spawn', or 'forkserver').

    Returns:
        multiprocessing.context.BaseContext: The requested multiprocessing
        context.
    """
    return multiprocessing.get_context(context)


# Global instance storage (used only in multi-processing mode)
_global_fn = None


def _initialize_fn(fn: Callable):
    """Initialize the function instance in subprocesses.

    This method is used when running in a multi-processing environment.
    Each subprocess will call this function during initialization to ensure
    that `fn` is instantiated only once per subprocess.

    Args:
        fn (Callable): The class or function to be instantiated in
            the subprocess.
    """
    global _global_fn
    _global_fn = fn


def _worker_fn(*args, **kwargs):
    """Worker function executed in ProcessPoolExecutor subprocesses.

    This function ensures that the globally initialized `_fn_instance` is used
    for task execution instead of creating a new instance on every
    function call.

    Returns:
        The result of the function execution.
    """
    global _global_fn
    if _global_fn is None:
        raise RuntimeError("Function instance not initialized in subprocess.")
    return _global_fn(*args, **kwargs)


class OrderedTaskExecutor:
    """Executor for managing and processing tasks in an ordered fashion.

    This class supports both synchronous and asynchronous task
    execution with optional multiprocessing.

    In multi-processing mode, each subprocess will initialize an instance of fn during startup,
    ensuring that the instance is not recreated for every function call.

    Attributes:
        send_idx (int): The index of the next task to be sent for execution.
        rcvd_idx (int): The index of the next result to be retrieved.
        buf_size (int): The current size of the task buffer.
    """  # noqa

    def __init__(
        self,
        fn: Callable,
        num_workers: int = 0,
        executor_type: Literal["thread", "process"] = "process",
        mp_context: str = "spawn",
        max_queue_size: int = 0,
        queue_full_action: Literal[
            "raise", "drop_last", "drop_first"
        ] = "raise",
    ):
        """Initialize the OrderedTaskExecutor.

        Args:
            fn (Callable): The function to execute tasks.
            num_workers (int): The number of worker processes for
                multiprocessing. Defaults to 0.
            executor_type (Literal["thread", "process"]): The concurrent
                executor type, ignored when num_workers = 0.
                Defaults to "process"
            mp_context (str): The multiprocessing context. Defaults to "spawn".
            max_queue_size (int): The maximum size of the task queue.
                Defaults to 0 (unlimited).
            queue_full_action (Literal["raise", "drop_last", "drop_first"]):
                Action to take when the task queue is full (i.e.,
                reaches max_queue_size). Defaults to "raise".
                Options are:
                - "raise": Raise a TaskQueueFulledError.
                - "drop_last": Discard the newest task (the one being added).
                - "drop_first": Discard the oldest task in the queue.
        """
        self._max_queue_size = max_queue_size
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._data_buffer = dict()
        self._queue_full_action = queue_full_action
        self._cancel_queues = []
        self._num_workers = num_workers
        self._drop_cnt = 0
        self._offset = 0
        if num_workers > 0:
            if max_queue_size > 0 and max_queue_size < 2 * num_workers:
                raise ValueError(
                    "The minimum value of `max_queue_size` is 2 * num_workers"
                )

            if executor_type == "thread":
                self._executor = ThreadPoolExecutor(
                    max_workers=num_workers,
                    initializer=_initialize_fn,
                    initargs=(fn,),
                )
                self._fn = _worker_fn
            else:
                self._executor = ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=_get_multiprocess_context(mp_context),
                    initializer=_initialize_fn,
                    initargs=(fn,),
                )
                self._fn = _worker_fn
        else:
            self._fn = fn

    def put(self, *args, **kwargs):
        """Submit a task for execution.

        Args:
            *args: Positional arguments to pass to the task function.
            **kwargs: Keyword arguments to pass to the task function.

        Raises:
            TaskQueueFulledError: If the task queue has reached its maximum
            size.
        """
        if (
            self._max_queue_size > 0
            and len(self._data_buffer) == self._max_queue_size
        ):
            if self._queue_full_action == "raise":
                raise TaskQueueFulledError
            elif self._queue_full_action == "drop_last":
                self._drop_cnt += 1
                logger.warning(
                    "Task queue is full (size={}), dropping latest message (already drop {} messages). "  # noqa: E501
                    "Possible causes: Task processing is slower than input rate, max_queue_size is too small, "  # noqa: E501
                    "or worker processes may still be initializing. "
                    "Suggestions: Increase num_workers, optimize task function, increase max_queue_size, "  # noqa: E501
                    "reduce input message frequency, or ensure process pool is fully initialized before processing.".format(  # noqa: E501
                        self._max_queue_size, self._drop_cnt
                    ),
                )
                return
            elif self._queue_full_action == "drop_first":
                oldest_idx = min(self._data_buffer.keys())
                oldest_task = self._data_buffer.pop(oldest_idx, None)
                if isinstance(oldest_task, Future):
                    oldest_task.cancel()  # try to cancel
                    self._cancel_queues.append(oldest_task)
                self._drop_cnt += 1
                self._offset += 1
                logger.warning(
                    "Task queue is full (size={}), dropping oldest message (already drop {} messages). "  # noqa: E501
                    "Possible causes: Task processing is slower than input rate, max_queue_size is too small, "  # noqa: E501
                    "or worker processes may still be initializing. "
                    "Suggestions: Increase num_workers, optimize task function, increase max_queue_size, "  # noqa: E501
                    "reduce input message frequency, or ensure process pool is fully initialized before processing.".format(  # noqa: E501
                        self._max_queue_size, self._drop_cnt
                    ),
                )
            else:
                raise ValueError(
                    "Invalid queue full action = {}".format(
                        self._queue_full_action
                    )
                )

        if hasattr(self, "_executor"):
            self._data_buffer[self._sent_idx] = self._executor.submit(
                self._fn, *args, **kwargs
            )
        else:
            self._data_buffer[self._sent_idx] = self._fn(*args, **kwargs)

        self._sent_idx += 1

    def get(self, block: bool = False, timeout: float | None = None) -> Any:
        """Retrieve the result of the next completed task.

        This method retrieves the result of the next task in the queue.
        If the task result is not immediately available, the behavior
        depends on the `block` parameter:
        - If `block` is False, it raises a `DataNotReadyError` if the result
        is not yet ready.
        - If `block` is True, it waits for the result to become available,
        up to the specified `timeout`.

        Args:
            block (bool): Whether to block until the task is completed.
                Defaults to False.
            timeout (float | None): Maximum time to wait for the task result
                in seconds. If `block` is False, this parameter is ignored.
                If None, it waits indefinitely. Defaults to None.

        Returns:
            Any: The result of the completed task.

        Raises:
            DataNotReadyError:
                - If the task result is not yet available and `block` is False.
                - If the task does not complete within the specified `timeout`
                when `block` is True.
            KeyError: If the internal task buffer does not contain the expected
                result. This is an unexpected state and might indicate an
                internal bug.
            TimeoutError: Raised internally by `Future.result()` when waiting
                for task completion times out.
        """
        fetch_idx = self._rcvd_idx + self._offset

        if fetch_idx not in self._data_buffer:
            raise DataNotReadyError

        ret = self._data_buffer[fetch_idx]

        if isinstance(ret, Future):
            if not block:
                if not ret.done():
                    raise DataNotReadyError
                else:
                    ret = ret.result()
            else:
                try:
                    ret = ret.result(timeout)
                except TimeoutError:
                    raise DataNotReadyError

        _ = self._data_buffer.pop(fetch_idx)

        self._rcvd_idx += 1

        return ret

    @property
    def send_idx(self):
        """Get the index of the next task to be sent for execution.

        Returns:
            int: The index of the next task to be sent.
        """
        return self._sent_idx

    @property
    def rcvd_idx(self):
        """Get the index of the next result to be retrieved.

        Returns:
            int: The index of the next result to be retrieved.
        """
        return self._rcvd_idx

    @property
    def buf_size(self):
        """Get the current size of the task buffer.

        Returns:
            int: The number of tasks currently in the buffer.
        """
        return len(self._data_buffer)

    def __del__(self):
        """Clean up and finalize all remaining tasks."""
        if hasattr(self, "_rcvd_idx"):
            while self._rcvd_idx < self._sent_idx:
                if self._rcvd_idx in self._data_buffer:
                    ret = self._data_buffer.pop(self._rcvd_idx)
                    try:
                        if isinstance(ret, Future):
                            _ = ret.result()
                    except:  # noqa
                        pass
                self._rcvd_idx += 1
        if hasattr(self, "_cancel_queues"):
            for f in self._cancel_queues:
                try:
                    if isinstance(f, Future):
                        _ = f.result()
                except:  # noqa
                    pass
        if hasattr(self, "_executor"):
            self._executor.shutdown()
