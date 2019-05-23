from yieldengine.feature.linkage import LinkageTree


class DendrogramDrawer:
    def __init__(self, linkage: LinkageTree):
        self._linkage = linkage

    def draw(self, width: int, height: int) -> None:
        pass
