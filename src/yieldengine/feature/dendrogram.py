from yieldengine.feature.linkage import LinkageNode, LinkageTree


class DendrogramDrawer:
    # note: draws for left->right oriented version (i.e. root node on the right)
    def __init__(self, linkage: LinkageTree):
        self._linkage = linkage

    def draw(self, width: int, height: int) -> None:
        # initialize figure/canvas using supplied bounds
        # run _draw_dendrogram
        pass

    def _draw_dendrogram(self, node: LinkageNode, y: int, width: int) -> int:
        # returns height
        pass

    def _draw_link_leg(self, node: LinkageNode, y: int, width: int) -> int:
        # returns height
        pass

    def _draw_link_leg_connector(self, y1: int, y2: int, x: int, color: str) -> None:
        pass

    def _draw_label(self, text: str, y: int) -> int:
        # returns height
        pass

    @staticmethod
    def color(node: LinkageNode) -> str:
        # map node.importance to hex-string
        pass
