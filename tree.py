from image_edit_domain.image_edit_dsl import *
from typing import Dict


class Tree:
    def __init__(self, _id: int):
        self.id: int = _id
        self.nodes: Dict[int, Expression] = {}
        self.to_children: Dict[int, List[int]] = {}
        self.to_parent: Dict[int, int] = {}
        self.depth = 1
        self.size = 1
        self.var_nodes = []

    def duplicate(self, _id: int) -> "Tree":
        ret = Tree(_id)
        # ret.nodes = copy.copy(self.nodes)
        ret.nodes = {}
        for key, val in self.nodes.items():
            if isinstance(val, Hole) or isinstance(val, Node):
                ret.nodes[key] = val.duplicate()
            else:
                ret.nodes[key] = val
        ret.to_children = self.to_children.copy()
        ret.to_parent = self.to_parent.copy()
        ret.var_nodes = self.var_nodes.copy()
        ret.depth = self.depth
        ret.size = self.size
        return ret

    def __lt__(self, other):
        if self.size == other.size and self.depth == other.depth:
            return self.id < other.id
        if self.size == other.size:
            return self.depth < other.depth
        return self.size < other.size
    

class Hole:
    def __init__(self, depth, node_type, output_over=None, output_under=None):
        self.depth = depth
        self.node_type = node_type
        self.output_over = output_over
        self.output_under = output_under
        self.val = None

    def __str__(self):
        return type(self).__name__

    def duplicate(self):
        return Hole(
            self.depth, 
            self.node_type, 
            self.output_over, 
            self.output_under,
        )

    def __lt__(self, other):
        if not isinstance(other, Hole):
            return False
        return str(self) < str(other)