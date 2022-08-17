import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, LineCollection
from numpy.linalg import norm


def plot_multiplex_network(
    ax,
    networks,
    pos,
    gc_colors,
    iso_colors,
    edge_colors,
    plane_colors,
    labels,
    node_size=5,
    edge_width=0.5,
    width=8,
    height=6,
    num_thru_nodes=10,
    label_pos=0.8,
):
    if ax.name != "3d":
        raise Exception("The axis is not 3d!")

    # node_radius = np.sqrt(node_size)

    for gi, G in enumerate(networks):
        # node colors
        xs = list()
        ys = list()
        zs = list()
        cs = list()
        gc = max(nx.connected_components(G), key=len)

        for node, coord in pos.items():
            xs.append(coord[0])
            ys.append(coord[1])
            zs.append(gi)
            if node in gc:
                cs.append(gc_colors[gi])
            else:
                cs.append(iso_colors[gi])

        # if you want to have between-layer connections
        if gi > 0:
            thru_nodes = np.random.choice(
                list(G.nodes()), num_thru_nodes, replace=False
            )
            lines3d_between = [
                (list(pos[i]) + [gi - 1], list(pos[i]) + [gi]) for i in thru_nodes
            ]
            between_lines = Line3DCollection(
                lines3d_between,
                zorder=gi,
                color=".5",
                alpha=0.4,
                linestyle="--",
                linewidth=1,
            )
            ax.add_collection3d(between_lines)

        # add within-layer edges
        lines3d = [(list(pos[i]) + [gi], list(pos[j]) + [gi]) for i, j in G.edges()]
        line_collection = Line3DCollection(
            lines3d, zorder=gi, color=edge_colors[gi], alpha=0.8, linewidth=edge_width
        )
        ax.add_collection3d(line_collection)

        # now add GC nodes
        ax.scatter(
            xs,
            ys,
            zs,
            c=cs,
            s=node_size,
            edgecolors=".2",
            linewidth=0.2,
            marker="o",
            alpha=1,
            zorder=gi + 1,
        )

        # add a plane to designate the layer
        xdiff = max(xs) - min(xs)
        ydiff = max(ys) - min(ys)
        ymin = min(ys) - ydiff * 0.1
        ymax = max(ys) + ydiff * 0.1
        xmin = min(xs) - xdiff * 0.1 * (width / height)
        xmax = max(xs) + xdiff * 0.1 * (width / height)
        xx, yy = np.meshgrid([xmin, xmax], [ymin, ymax])
        zz = np.zeros(xx.shape) + gi
        ax.plot_surface(xx, yy, zz, color=plane_colors[gi], alpha=0.1, zorder=gi)

        # add label
        x_pos = xmin + label_pos * xdiff
        ax.text(
            0.0,
            x_pos,
            gi * 0.95 + 0.5,
            labels[gi],
            color="black",
            fontsize="large",
            zorder=1e5,
            ha="center",
            va="center",
        )

    # set them all at the same x,y,zlims
    ax.set_ylim(min(ys) - ydiff * 0.1, max(ys) + ydiff * 0.1)
    ax.set_xlim(min(xs) - xdiff * 0.1, max(xs) + xdiff * 0.1)
    ax.set_zlim(-0.1, len(networks) - 1 + 0.1)

    # select viewing angle
    angle = 30
    height_angle = 20
    ax.view_init(height_angle, angle)

    # how much do you want to zoom into the fig
    ax.dist = 8.5

    ax.set_axis_off()
