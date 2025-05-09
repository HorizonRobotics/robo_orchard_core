# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

from __future__ import annotations
import traceback
import warnings
from abc import ABCMeta, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from IPython.display import display

try:
    from ipycanvas import Canvas, hold_canvas
    from ipyevents import Event
    from ipywidgets import (
        Image as ImageWidget,
        Output,
    )
    from PIL import Image as PILImage
except Exception:
    warnings.warn(f"Error importing {__file__}:\n" + traceback.format_exc())


def is_left_click(event: DomEvent) -> bool:
    return bool(event["buttons"] & 1)


def is_right_click(event: DomEvent) -> bool:
    return bool(event["buttons"] & 2)


def is_middle_click(event: DomEvent) -> bool:
    return bool(event["buttons"] & 4)


def draw_image(canvas: Canvas, image: np.ndarray, format="PNG", quality=100):
    if not (isinstance(image, np.ndarray) and image.dtype == np.uint8):
        raise ValueError("image must be a numpy array of dtype uint8.")
    if format.lower() == "png" and image.shape[-1] in [1, 3]:
        quality = int((100 - quality) * 9 / 100.0)

    if format in ["jpeg", "jpg"] and image.shape[-1] == 4:
        image = image[..., :3]

    img_height, img_width = image.shape[:2]

    min_sclale = min(
        float(canvas.width) / img_width, float(canvas.height) / img_height
    )

    resize_width = int(img_width * min_sclale)
    resize_height = int(img_height * min_sclale)
    # Todo: Better way to convert image to bytes?
    # image = cv2.resize(image, (resize_width, resize_height))

    f = BytesIO()
    PILImage.fromarray(image).save(f, format, quality=quality)

    image_widget = ImageWidget(value=f.getvalue())
    with hold_canvas(canvas):
        canvas.draw_image(image_widget, 0, 0, resize_width, resize_height)


class DomEvent(TypedDict):
    target: dict
    altKey: bool
    metaKey: bool
    shiftKey: bool
    ctrlKey: bool
    type: str
    event: str
    buttons: int
    # 0: no button, 1: left button, 2: right button, 4: middle button.
    # Bitwise, can be combined.

    clientX: int
    clientY: int
    offsetX: int
    offsetY: int
    pageX: int
    pageY: int
    screenX: int
    screenY: int
    x: int
    y: int
    relativeX: int
    relativeY: int
    deltaY: int


class BaseIpyViz(metaclass=ABCMeta):
    def __init__(
        self,
        height: int,
        width: int,
        watched_events: list[str],
        canvas: Canvas | None = None,
        max_fps: int = 20,
    ):
        self.height = height
        self.width = width

        if canvas is None:
            canvas: Canvas = Canvas(width=width, height=height)
        else:
            if canvas.width != width or canvas.height != height:
                warnings.warn(
                    f"canvas.width={canvas.width} != width={width} or "
                    f"canvas.height={canvas.height} != height={height}. "
                    "height and width will be set to canvas.height and canvas.width."  # noqa: E501
                )
                self.height = canvas.height
                self.width = canvas.width

        self.canvas: Canvas = canvas
        self.output = Output()

        wait = 1000 // max_fps if max_fps > 0 else 0

        self.event = Event(
            source=self.canvas,
            watched_events=watched_events,
            prevent_default_action=True,
            wait=wait,
        )

        self.event.on_dom_event(self._on_event)

    @abstractmethod
    def _on_event(self, event: DomEvent):
        raise NotImplementedError

    def display(self):
        """Display the canvas and output.

        This function should only be called in Jupyter notebook.
        """
        display(self.canvas, self.output)
