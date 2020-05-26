import pytest

from openmodelica_microgrid_gym.env import PlotTmpl

v = [['a', 'b'], ['c', 'd']]
tmpl = PlotTmpl(v, color=[None, ['C2', 'C1']], style=[None, '--'])


@pytest.mark.parametrize('i,o', [[[k for k in PlotTmpl(v)],
                                  [('a', dict(c='C1')), ('b', dict(c='C2')), ('c', dict(c='C1')), ('d', dict(c='C2'))]],
                                 [[k for k in PlotTmpl(v, c=[None, ['C2', 'C1']])],
                                  [('a', dict(c='C1')), ('b', dict(c='C2')), ('c', dict(c='C2')), ('d', dict(c='C1'))]],
                                 [[k for k in PlotTmpl(v, color=[None, ['C2', 'C1']])],
                                  [('a', dict(color='C1')), ('b', dict(color='C2')),
                                   ('c', dict(color='C2')), ('d', dict(color='C1'))]],
                                 [tmpl[2],
                                  ('c', dict(color='C2', style='--'))],
                                 [tmpl[1],
                                  ('b', dict(color='C2'))]])
def test_plot_tmpl(i, o):
    assert i == o
