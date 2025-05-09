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

import time

import pytest

from robo_orchard_core.utils.task_executor import (
    DataNotReadyError,
    OrderedTaskExecutor,
    TaskQueueFulledError,
)


def dummy_task(x):
    return x * 2


def delayed_task(x):
    time.sleep(1)
    return x * 2


def test_executor_single_thread():
    """Test single-threaded task execution."""

    executor = OrderedTaskExecutor(fn=dummy_task)
    executor.put(3)
    result = executor.get()
    assert result == 6, "Single-threaded task result mismatch."


@pytest.mark.parametrize("executor_type", ["thread", "process"])
def test_executor_multi_thread(executor_type):
    """Test multi-threaded task execution with ProcessPoolExecutor."""

    executor = OrderedTaskExecutor(
        fn=dummy_task, num_workers=2, executor_type=executor_type
    )
    executor.put(3)
    executor.put(4)
    result1 = executor.get(block=True)
    result2 = executor.get(block=True)
    assert result1 == 6, "Multi-threaded task result 1 mismatch."
    assert result2 == 8, "Multi-threaded task result 2 mismatch."


def test_queue_size_limit():
    """Test TaskQueueFulledError when task queue exceeds max size."""

    executor = OrderedTaskExecutor(fn=dummy_task, max_queue_size=1)
    executor.put(1)
    with pytest.raises(TaskQueueFulledError):
        executor.put(2)


def test_data_not_ready_error():
    """Test DataNotReadyError when attempting to get a task not ready."""

    executor = OrderedTaskExecutor(fn=dummy_task)
    with pytest.raises(DataNotReadyError):
        executor.get()  # No task submitted yet


def test_async_execution_with_delay():
    """Test async task execution with a delay."""

    executor = OrderedTaskExecutor(fn=delayed_task, num_workers=2)
    executor.put(2)

    with pytest.raises(DataNotReadyError):
        executor.get()  # Task should not be ready immediately

    time.sleep(1.1)  # Wait for task to complete
    result = executor.get(block=True)
    assert result == 4, "Async task result mismatch."


def test_send_and_receive_indices():
    """Test the send_idx and rcvd_idx properties."""

    executor = OrderedTaskExecutor(fn=dummy_task)
    assert executor.send_idx == 0, "Initial send_idx should be 0."
    assert executor.rcvd_idx == 0, "Initial rcvd_idx should be 0."

    executor.put(1)
    assert executor.send_idx == 1, (
        "send_idx should increment after putting a task."
    )
    executor.get()
    assert executor.rcvd_idx == 1, (
        "rcvd_idx should increment after getting a result."
    )


def test_buffer_size():
    """Test buf_size property."""

    executor = OrderedTaskExecutor(fn=dummy_task)
    assert executor.buf_size == 0, "Initial buffer size should be 0."

    executor.put(1)
    assert executor.buf_size == 1, (
        "Buffer size should increment after putting a task."
    )

    executor.get()
    assert executor.buf_size == 0, (
        "Buffer size should decrement after getting a result."
    )


def test_cleanup_on_delete():
    """Test cleanup behavior when executor is deleted."""

    executor = OrderedTaskExecutor(fn=dummy_task, num_workers=2)
    executor.put(1)
    executor.put(2)


def test_drop_first():
    executor = OrderedTaskExecutor(
        fn=dummy_task,
        num_workers=2,
        max_queue_size=5,
        queue_full_action="drop_first",
    )
    data = list(range(10))
    for data_i in data:
        executor.put(data_i)
    for data_i in data[5:]:
        assert executor.get(block=True) == dummy_task(data_i)


def test_drop_last():
    executor = OrderedTaskExecutor(
        fn=dummy_task,
        num_workers=2,
        max_queue_size=5,
        queue_full_action="drop_last",
    )
    data = list(range(10))
    for data_i in data:
        executor.put(data_i)
    for data_i in data[:5]:
        assert executor.get(block=True) == dummy_task(data_i)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
