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
from typing import Generic, TypeVar

from robo_orchard_core.datatypes.dataclass import DataClass
from robo_orchard_core.datatypes.geometry import BatchFrameTransform

EDGE_TYPE = TypeVar("EDGE_TYPE")
NODE_TYPE = TypeVar("NODE_TYPE")


class EdgeGraph(Generic[EDGE_TYPE, NODE_TYPE], DataClass):
    """A generic edge graph data structure."""

    graph: dict[str, dict[str, EDGE_TYPE]]
    nodes: dict[str, NODE_TYPE]

    def __init__(self):
        self.graph = {}
        self.nodes = {}
        self._in_degree = {node_id: 0 for node_id in self.nodes}

    def _add_node(self, node_id: str, node: NODE_TYPE):
        """Add a node to the graph."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists.")
        self.nodes[node_id] = node
        self.graph[node_id] = {}
        self._in_degree[node_id] = 0

    def _add_edge(self, from_node: str, to_node: str, edge: EDGE_TYPE):
        """Add an edge between two nodes."""
        if from_node not in self.nodes:
            raise ValueError(f"From node {from_node} does not exist.")
        if to_node not in self.nodes:
            raise ValueError(f"To node {to_node} does not exist.")
        if to_node in self.graph[from_node]:
            raise ValueError(
                f"Edge from {from_node} to {to_node} already exists."
            )
        self.graph[from_node][to_node] = edge
        self._in_degree[to_node] += 1

    def connected_subgraph_number(self) -> int:
        """Count the number of all subgraphs in the graph."""

        zero_in_degree_nodes = [
            node_id
            for node_id, degree in self._in_degree.items()
            if degree == 0
        ]
        return max(1, len(zero_in_degree_nodes))

    def get_path_by_bfs(
        self, src_node_id: str, dst_node_id: str
    ) -> list[EDGE_TYPE] | None:
        """Get the path from src_node_id to dst_node_id.

        This method uses breadth-first search (BFS) to find the shortest path
        between two nodes in the graph. If no path exists, it returns None.

        Args:
            src_node_id (str): The ID of the source node.
            dst_node_id (str): The ID of the destination node.

        Returns:
            list[EDGE_TYPE] | None: A list of edges representing the path from
            src_node_id to dst_node_id.

        """
        if (
            src_node_id not in self.nodes
            or dst_node_id not in self.nodes
            or src_node_id == dst_node_id
        ):
            return None

        # apply breadth-first search (BFS) to find the path
        queue = [src_node_id]
        visited = {src_node_id}
        # Record who first visited the node.
        # This is used to reconstruct the shortest path.
        parent_map: dict[str, str | None] = {src_node_id: None}
        while queue:
            current_node = queue.pop(0)
            if current_node == dst_node_id:
                break
            for neighbor in self.graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current_node
                    queue.append(neighbor)
                else:
                    continue

        if dst_node_id not in visited:
            return None

        # reconstruct the path
        path = []
        current_node = dst_node_id
        while current_node is not None:
            parent = parent_map[current_node]
            if parent is not None:
                edge = self.graph[parent].get(current_node)
                if edge is not None:
                    path.append(edge)
            current_node = parent
        path.reverse()
        return path


class BatchFrameTransformGraph(EdgeGraph[BatchFrameTransform, str]):
    """A graph structure for batch frame transforms.

    This graph structure is specifically designed to handle batch frame
    transforms, where each edge represents a transformation between two frames.
    The nodes are identified by their frame IDs.
    """

    def __init__(self, tf_list: list[BatchFrameTransform] | None):
        super().__init__()

        if tf_list is not None:
            self.add_tf(tf_list)

    def add_tf(self, tf_list: list[BatchFrameTransform]):
        for tf in tf_list:
            if tf.parent_frame_id is None:
                raise ValueError(
                    "BatchFrameTransform must have a parent frame ID."
                )
            if tf.child_frame_id is None:
                raise ValueError(
                    "BatchFrameTransform must have a child frame ID."
                )
            if tf.parent_frame_id not in self.nodes:
                self._add_node(tf.parent_frame_id, tf.parent_frame_id)
            if tf.child_frame_id not in self.nodes:
                self._add_node(tf.child_frame_id, tf.child_frame_id)
            self._add_edge(
                from_node=tf.parent_frame_id,
                to_node=tf.child_frame_id,
                edge=tf,
            )

    def get_tf(
        self, parent_frame_id: str, child_frame_id: str
    ) -> BatchFrameTransform | None:
        """Get the transformation between two frames.

        Args:
            parent_frame_id (str): The ID of the parent frame.
            child_frame_id (str): The ID of the child frame.

        Returns:
            BatchFrameTransform | None: The transformation between
                the two frames, or None if no such transformation exists.
        """
        if (
            parent_frame_id not in self.nodes
            or child_frame_id not in self.nodes
        ):
            return None

        path = self.get_path_by_bfs(
            src_node_id=parent_frame_id, dst_node_id=child_frame_id
        )
        if path is None:
            return None
        assert len(path) > 0, "Path should not be empty."

        if len(path) == 1:
            return path[0]
        else:
            return path[0].compose(*path[1:])
