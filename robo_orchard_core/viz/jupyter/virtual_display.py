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
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import mss
import numpy as np

if TYPE_CHECKING:
    from IPython.display import display

try:
    import pyautogui
    from ipycanvas import Canvas
    from pyvirtualdisplay.smartdisplay import SmartDisplay
except Exception:
    warnings.warn(f"Error importing {__file__}:\n" + traceback.format_exc())

from robo_orchard_core.viz.jupyter.base_viz import (
    BaseIpyViz,
    DomEvent,
    draw_image,
    is_left_click,
    is_middle_click,
    is_right_click,
)


class IpyVirtualDisplay(BaseIpyViz):
    def __init__(
        self,
        height: int,
        width: int,
        virtual_display: SmartDisplay,
        additional_watched_events: list[str] | None = None,
        canvas: Canvas | None = None,
        max_fps: int = 20,
        format="jpeg",
        quality=80,
    ):
        _watched_events = set(
            [
                "wheel",
                "mousedown",
                "mouseup",
                "mousemove",
                "mouseleave",
                "mouseenter",
                "contextmenu",
            ]
        )
        if additional_watched_events:
            _watched_events.update(additional_watched_events)

        BaseIpyViz.__init__(
            self,
            height=height,
            width=width,
            watched_events=list(_watched_events),
            canvas=canvas,
            max_fps=max_fps,
        )
        self._virtual_display = virtual_display
        self._mss = mss.mss()
        self._mss.__enter__()
        self._position = (-1.0, -1.0)
        self._format = format
        self._quality = quality
        self._render_executor = ThreadPoolExecutor(max_workers=1)
        self._last_render_task = None
        self._left_click = False
        self._right_click = False
        self._middle_click = False

    def display(self):
        display(
            self.canvas,
            self.output,
        )
        # # render the first image.
        self._render()

    def _on_event(self, event: DomEvent):
        with self.output:
            # handle ctrlKey
            if event["ctrlKey"]:
                pyautogui.keyDown("ctrl", _pause=False)
            else:
                pyautogui.keyUp("ctrl", _pause=False)

            # handle wheel and mouse events
            if event["type"] == "wheel":
                pyautogui.scroll(event["deltaY"], _pause=False)

            elif event["type"] in ["mousedown", "mouseup", "mousemove"]:
                self._position = (
                    float(event["relativeX"]),
                    float(event["relativeY"]),
                )
                screen_wh = pyautogui.resolution()

                min_sclale = min(
                    float(self.canvas.width) / screen_wh[0],
                    float(self.canvas.height) / screen_wh[1],
                )
                resize_width = int(screen_wh[0] * min_sclale)
                resize_height = int(screen_wh[1] * min_sclale)

                target_x = int(
                    (event["relativeX"] / resize_width) * screen_wh[0]
                )
                target_y = int(
                    (event["relativeY"] / resize_height) * screen_wh[1]
                )

                if event["type"] == "mousedown":
                    if is_left_click(event):
                        pyautogui.mouseDown(button="left", _pause=True)
                        self._left_click = True
                    elif is_right_click(event):
                        pyautogui.mouseDown(button="right", _pause=True)
                        self._right_click = True
                    elif is_middle_click(event):
                        pyautogui.mouseDown(button="middle", _pause=True)
                        self._middle_click = True
                elif event["type"] == "mouseup":
                    # self._render()
                    if not is_left_click(event) and self._left_click:
                        self._left_click = False
                        pyautogui.mouseUp(button="left", _pause=True)
                    if not is_right_click(event) and self._right_click:
                        self._right_click = False
                        pyautogui.mouseUp(button="right", _pause=True)
                    if not is_middle_click(event) and self._middle_click:
                        self._middle_click = False
                        pyautogui.mouseUp(button="middle", _pause=True)
                elif event["type"] == "mousemove":
                    # +x: right, +y: down in the screen
                    pyautogui.moveTo(target_x, target_y, _pause=False)
            else:
                print(f"Unhandled event: {event}")
            self._render()

    def _render(self):
        def render_impl(
            img: np.ndarray,
            canvas: Canvas,
        ):
            draw_image(
                canvas,
                img[..., 0:3],
                format=self._format,
                quality=self._quality,
            )

        display_id = self._virtual_display.display
        screen_wh = self._virtual_display._size
        img = self._mss.grab(
            {
                "top": 0,
                "left": 0,
                "width": screen_wh[0],
                "height": screen_wh[1],
                "mon": display_id,
            }
        )
        np_image = np.array(img)[..., 0:3]
        # change the color space from BGR to RGB
        np_image = np_image[..., ::-1]
        self._last_render_task = self._render_executor.submit(
            render_impl,
            img=np_image,
            canvas=self.canvas,
        )
