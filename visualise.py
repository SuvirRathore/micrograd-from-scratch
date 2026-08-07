from graphviz import Digraph


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        dot.node(name=str(id(n)), label="{ %s | data %.2f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        # added node, str(n) gives name, the label is for precision, shape is rectangle
        # for shape rectangle use 'box' but then change label to f"data {n.data: .4f}"
        # dot.node(name=str(id(n)), label=f"data {n.data: .4f}", shape='box')
        if n._op:
            # add node and edge for operation (say node + to node d as d = a*b + c)
            dot.node(name=str(id(n)) + n._op, label=n._op, shape='circle', style='filled', fillcolor='lightblue')
            dot.edge(str(id(n)) + n._op, str(id(n)))
    # add edges as expected but from given node to the operation its applied to (like a*b to +)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot