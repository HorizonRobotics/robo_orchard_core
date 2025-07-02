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

    edges: dict[str, dict[str, EDGE_TYPE]]
    """Graph is represented as a set of edges."""
    nodes: dict[str, NODE_TYPE]
    """The nodes are represented as string dict."""

    def __init__(self):
        self.edges = {}
        self.nodes = {}
        self._in_degree = {node_id: 0 for node_id in self.nodes}

    def _add_node(self, node_id: str, node: NODE_TYPE):
        """Add a node to the graph."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists.")
        self.nodes[node_id] = node
        self.edges[node_id] = {}
        self._in_degree[node_id] = 0

    def _add_edge(self, from_node: str, to_node: str, edge: EDGE_TYPE):
        """Add an edge between two nodes."""
        if from_node not in self.nodes:
            raise ValueError(f"From node {from_node} does not exist.")
        if to_node not in self.nodes:
            raise ValueError(f"To node {to_node} does not exist.")
        if to_node in self.edges[from_node]:
            raise ValueError(
                f"Edge from {from_node} to {to_node} already exists."
            )
        self.edges[from_node][to_node] = edge
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
            for neighbor in self.edges[current_node]:
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
                edge = self.edges[parent].get(current_node)
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

    Args:
        tf_list (list[BatchFrameTransform] | None): A list of
            BatchFrameTransform objects to initialize the graph with.
            If None, an empty graph is created.
        bidirectional (bool): Whether to add mirrored edges in the opposite
            direction. Defaults to True.
        static_tf (list[bool] | None): A list of booleans indicating whether
            each BatchFrameTransform is static. If None, all transforms are
            considered non-static. Defaults to None.

    """

    edges: dict[str, dict[str, BatchFrameTransform]]
    """Graph is represented as a set of tf.

    Mirrored(inversed) edges are also stored in the graph.
    """

    def __init__(
        self,
        tf_list: list[BatchFrameTransform] | None,
        bidirectional: bool = True,
        static_tf: list[bool] | None = None,
    ):
        super().__init__()
        self._mirrored_edges: dict[str, dict[str, BatchFrameTransform]] = {}
        self._static_edges: dict[str, dict[str, bool]] = {}

        if tf_list is not None:
            self.add_tf(
                tf_list, bidirectional=bidirectional, static_tf=static_tf
            )

    def is_mirrored_tf(
        self, parent_frame_id: str, child_frame_id: str
    ) -> bool:
        """Check if the edge is a mirrored edge."""
        return (
            parent_frame_id in self._mirrored_edges
            and child_frame_id in self._mirrored_edges[parent_frame_id]
        )

    def is_static_tf(self, parent_frame_id: str, child_frame_id: str) -> bool:
        """Check if the edge is a static edge."""
        return parent_frame_id in self._static_edges and self._static_edges[
            parent_frame_id
        ].get(child_frame_id, False)

    def _add_node(self, node_id: str, node: str):
        """Add a node to the graph.

        Overwrites the base class method to ensure that mirrored edges
        and static edges are initialized correctly.
        """
        ret = super()._add_node(node_id, node)
        if node_id in self._mirrored_edges:
            raise ValueError(
                f"Node {node_id} already exists in mirrored edges."
            )
        self._mirrored_edges[node_id] = {}
        self._static_edges[node_id] = {}
        return ret

    def _add_edge(
        self,
        from_node: str,
        to_node: str,
        edge: BatchFrameTransform,
        bidirectional: bool = True,
        is_static: bool = False,
    ):
        """Add an edge between two nodes.

        Overwrites the base class method to handle mirrored edges
        and static edges.
        """

        super()._add_edge(from_node, to_node, edge)
        if is_static:
            self._static_edges[from_node][to_node] = True
        if bidirectional:
            # Add the mirrored edge in the opposite direction
            mirrored_tf = edge.inverse()
            mirrored_from_node = mirrored_tf.parent_frame_id
            mirrored_to_node = mirrored_tf.child_frame_id
            self._mirrored_edges[mirrored_from_node][mirrored_to_node] = (
                mirrored_tf
            )
            super()._add_edge(
                from_node=mirrored_from_node,
                to_node=mirrored_to_node,
                edge=mirrored_tf,
            )
            if is_static:
                self._static_edges[mirrored_from_node][mirrored_to_node] = True

    def update_tf(self, tf: BatchFrameTransform):
        """Update a BatchFrameTransform in the graph.

        You can only update a non-static BatchFrameTransform. If the
        BatchFrameTransform is static, it will raise a ValueError.

        If the mirrored(inversed) edge exists, it will also be updated
        accordingly.

        Args:
            tf (BatchFrameTransform): The BatchFrameTransform to update.
        """
        old = self.edges.get(tf.parent_frame_id, {}).get(
            tf.child_frame_id, None
        )
        if old is None:
            raise ValueError(
                f"BatchFrameTransform from {tf.parent_frame_id} to "
                f"{tf.child_frame_id} does not exist."
            )
        # check if the new transform is static
        is_static = self._static_edges.get(tf.parent_frame_id, {}).get(
            tf.child_frame_id, False
        )
        if is_static:
            raise ValueError(
                f"Cannot update static BatchFrameTransform from "
                f"{tf.parent_frame_id} to {tf.child_frame_id}."
            )
        # update the edge
        self.edges[tf.parent_frame_id][tf.child_frame_id] = tf
        # update the mirrored edge if it exists
        if self.is_mirrored_tf(
            tf.child_frame_id,
            tf.parent_frame_id,
        ):
            mirrored_tf = tf.inverse()
            mirrored_from_node = mirrored_tf.parent_frame_id
            mirrored_to_node = mirrored_tf.child_frame_id
            self._mirrored_edges[mirrored_from_node][mirrored_to_node] = (
                mirrored_tf
            )
            self.edges[mirrored_from_node][mirrored_to_node] = mirrored_tf

    def add_tf(
        self,
        tf_list: list[BatchFrameTransform],
        bidirectional: bool = True,
        static_tf: list[bool] | None = None,
    ):
        """Add a list of BatchFrameTransform to the graph.

        Args:
            tf_list (list[BatchFrameTransform]): A list of BatchFrameTransform
                objects to add to the graph.
            bidirectional (bool): Whether to add mirrored edges in the opposite
                direction. Defaults to True.
            static_tf (list[bool] | None): A list of booleans indicating
                whether each BatchFrameTransform is static. If None, all
                transforms are considered non-static. Defaults to None.
        """

        if static_tf is not None and len(static_tf) != len(tf_list):
            raise ValueError(
                "static_tf and tf_list must have the same length."
            )
        static_tf = static_tf or [False] * len(tf_list)
        for tf, is_static in zip(tf_list, static_tf, strict=True):
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
                bidirectional=bidirectional,
                is_static=is_static,
            )

    def get_tf(
        self, parent_frame_id: str, child_frame_id: str, compose: bool = True
    ) -> BatchFrameTransform | list[BatchFrameTransform] | None:
        """Get the transformation between two frames.

        Args:
            parent_frame_id (str): The ID of the parent frame.
            child_frame_id (str): The ID of the child frame.

        Returns:
            BatchFrameTransform | list[BatchFrameTransform] | None: The
            transformation between the two frames. If compose is True, it
            returns a single BatchFrameTransform object. If compose is False,
            it returns a list of BatchFrameTransform objects representing the
            path from the parent frame to the child frame. If no path exists,
            it returns None.
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

        if compose:
            if len(path) == 1:
                return path[0]
            else:
                return path[0].compose(*path[1:])
        else:
            return path

    def export_edges(
        self,
        include_mirrored: bool = False,
    ) -> list[BatchFrameTransform]:
        """Export the edges of the graph.

        Args:
            include_mirrored (bool): Whether to include mirrored edges in the
                exported edges. Defaults to False.
        """
        edges = []
        for from_node, to_edges in self.edges.items():
            for to_node, edge in to_edges.items():
                if not include_mirrored and self.is_mirrored_tf(
                    from_node, to_node
                ):
                    # Skip mirrored edges if not included
                    continue
                edges.append(edge)
        return edges
