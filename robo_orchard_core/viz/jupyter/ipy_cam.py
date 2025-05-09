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
import math
import traceback
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from robo_orchard_core.datatypes.geometry import BatchPose6D
from robo_orchard_core.utils.math.coord_convention import CoordAxis
from robo_orchard_core.viz.cam import CameraMixin, CameraMoveMixin

if TYPE_CHECKING:
    from IPython.display import display

try:
    import ipywidgets as widgets
    from ipycanvas import Canvas
    from ipywidgets import FloatSlider
except Exception:
    warnings.warn(f"Error importing {__file__}:\n" + traceback.format_exc())

from robo_orchard_core.viz.jupyter.base_viz import (
    BaseIpyViz,
    DomEvent,
    is_left_click,
    is_right_click,
)


class IpyFPVCameraViz(BaseIpyViz, CameraMoveMixin, CameraMixin):
    """A first person view camera visualization in Jupyter notebook.

    Camera control:
        - Left click and drag to rotate the camera.
        - Right click and drag to move the camera.
        - Scroll to move forward/backward.
        - Ctrl + scroll to change move scale.

    Args:
        height (int): The height of the canvas.
        width (int): The width of the canvas.
        local_coord_axis (CoordAxis): The local coordinate axis of the camera.
        additional_watched_events (list[str]|None): The list of additional
            watched events. Default is None.
        canvas (Canvas | None): The canvas to render the camera view. If None,
            a new canvas will be created. Default is None.
        max_fps (int): The maximum frame per second. Default is 20.
        initial_move_scale (float): The initial move scale. Default is 2.5.
        forward_sensitivity (float): The sensitivity of moving
            forward/backward. Default is 0.01.
        translation_sensitivity (float): The sensitivity of translation.
            Default is 1.0.
        rotation_sensitivity (float): The sensitivity of rotation.
            Default is 0.1.

    """

    def __init__(
        self,
        height: int,
        width: int,
        local_coord_axis: CoordAxis,
        additional_watched_events: list[str] | None = None,
        canvas: Canvas | None = None,
        max_fps: int = 20,
        initial_move_scale: float = 2.5,
        forward_sensitivity: float = 0.01,
        translation_sensitivity: float = 1.0,
        rotation_sensitivity: float = 0.1,
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
        CameraMoveMixin.__init__(self, local_coord_axis=local_coord_axis)
        CameraMixin.__init__(self)

        if initial_move_scale <= 0:
            raise ValueError("initial_move_scale must be positive.")

        self.move_scale = initial_move_scale
        self.forward_sensitivity = forward_sensitivity
        self.translation_sensitivity = translation_sensitivity
        self.rotation_sensitivity = rotation_sensitivity
        self._position = (-1.0, -1.0)

        self._init_layout()

    def _init_layout(self):
        description_str = (
            "Camera control: \n"
            "  - Left click and drag to rotate the camera.\n"
            "  - Right click and drag to move the camera.\n"
            "  - Scroll to move forward/backward. \n"
            "  - Ctrl + scroll to change move scale."
        )
        # display newlines in the description.
        self._description = widgets.HTML(
            value=description_str.replace("\n", "<br>"),
            layout={"width": "300px"},
        )
        self._log_move_scale_slider = FloatSlider(
            value=math.log(self.move_scale),
            min=-4,
            max=3,
            description="",
            disabled=True,
        )

        def _on_log_move_scale_slider_change(change: dict[str, Any]):
            self.move_scale = math.exp(change["new"])

        self._log_move_scale_slider.observe(
            _on_log_move_scale_slider_change,
            names="value",
        )

        self._camera_position_label = widgets.Label(value="")
        self._camera_quat_label = widgets.Label(value="")
        self._mouse_position_label = widgets.Label(value="")
        layout = {"width": "120px"}
        self._camera_info_box = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.Label(
                            value="log(move_scale): ", layout=layout
                        ),
                        self._log_move_scale_slider,
                    ]
                ),
                widgets.HBox(
                    [
                        widgets.Label(
                            value="cam position xyz: ", layout=layout
                        ),
                        self._camera_position_label,
                    ]
                ),
                widgets.HBox(
                    [
                        widgets.Label(
                            value="cam rotation wxyz: ", layout=layout
                        ),
                        self._camera_quat_label,
                    ]
                ),
                widgets.HBox(
                    [
                        widgets.Label(
                            value="mouse position xy: ", layout=layout
                        ),
                        self._mouse_position_label,
                    ]
                ),
            ]
        )

    def display(self):
        """Display the canvas and output.

        This function should only be called in Jupyter notebook.
        """
        display(
            widgets.HBox([self._description, self._camera_info_box]),
            self.canvas,
            self.output,
        )
        # render the first image.
        self._render()
        self._on_pose_change()

    def _on_pose_change(self):
        pose = self.get_pose_view_world()
        self._camera_position_label.value = str(pose.xyz[0])
        self._camera_quat_label.value = str(pose.quat[0])

    def _on_increase_move_scale(self):
        self._log_move_scale_slider.value = math.log(self.move_scale)

    def increase_move_scale(self, amount: float):
        """Increase the move scale by the given amount."""
        new_log_v = math.log(self.move_scale) + amount
        self.move_scale = math.exp(new_log_v)
        self._on_increase_move_scale()

    def _on_event(self, event: DomEvent):
        with self.output:
            if event["type"] == "wheel":
                if event["ctrlKey"]:
                    self.increase_move_scale(
                        event["deltaY"] * self.forward_sensitivity * -1
                    )
                else:
                    self.move_forward(
                        event["deltaY"] * self.forward_sensitivity,
                        move_scale=self.move_scale,
                    )
                    self._on_pose_change()
                    self._position = (
                        float(event["relativeX"]),
                        float(event["relativeY"]),
                    )
                    self._mouse_position_label.value = str(self._position)
                self._render()
            elif event["type"] in ["mousedown", "mouseup"]:
                self._position = (
                    float(event["relativeX"]),
                    float(event["relativeY"]),
                )
                self._mouse_position_label.value = str(self._position)
            elif event["type"] == "mousemove":
                # +x: right, +y: down
                dx = (
                    event["relativeX"] - self._position[0]
                ) / self.canvas.width
                dy = (
                    event["relativeY"] - self._position[1]
                ) / self.canvas.height
                self._position = (
                    float(event["relativeX"]),
                    float(event["relativeY"]),
                )
                self._mouse_position_label.value = str(self._position)
                if is_left_click(event):
                    self.first_person_view_rot(
                        pitch_down=dy * self.rotation_sensitivity,
                        yaw_right=dx * self.rotation_sensitivity,
                    )
                elif is_right_click(event):
                    # right click to drag image.
                    # so the movement is opposite to the mouse movement.
                    self.move_translation(
                        amount_down=dy * self.translation_sensitivity * -1,
                        amount_right=dx * self.translation_sensitivity * -1,
                        move_scale=self.move_scale,
                    )
                self._on_pose_change()
                self._render()

    @abstractmethod
    def _render(self):
        pass

    @abstractmethod
    def get_pose_view_world(self) -> BatchPose6D:
        pass

    @abstractmethod
    def _apply_to_view_local(
        self, translation: torch.Tensor | None, quat: torch.Tensor | None
    ):
        pass
