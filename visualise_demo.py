from engine import Value
from visualise import draw_dot
from IPython.display import display
import random

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = a*b + c
d.label = 'd'
d.backward()
draw_dot(d).render('graph', view=True)

from nn import Neuron
random.seed(0)
n = Neuron(3)
x = [Value(1.0, label='x1'), Value(2.0, label='x2'), Value(0.5, label='x3')]
out = n(x)
out.label = 'out'
out.backward()
draw_dot(out).render('graph', view=True)