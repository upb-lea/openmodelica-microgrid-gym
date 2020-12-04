import os.path as p

import matplotlib.pyplot as plt

from openmodelica_microgrid_gym.agents import SafeOptAgent


class PlotManager:
    def __init__(self, used_agent: SafeOptAgent,
                 save_results: bool = False, save_folder: str = 'test_folder', show_plots: bool = True):
        """
        Class for plot configuration, knows agent, so can include agent params to title (e.g. performance)
        If more plots should be stored in save_folder, extend the corresponding label function with save command
        :param used_agent: agent used for experiment
        :param save_results: if True, saves results to save_folder
        :param save_folder: folder name
        :param show_plots: if True, shows plots after each run
        """

        self.agent = used_agent
        self.save_results = save_results
        self.save_folder = save_folder
        self.show_plots = show_plots

    def xylables_v_abc(self, fig):
        self.update_axes(fig,
                         ylabel='$v_{\mathrm{abc}}\,/\,\mathrm{V}$',
                         filename=f'{self.agent.history.df.shape[0]}_J_{self.agent.performance}_v_abc0',
                         legend=dict(handle_slice=slice(None, None, 3), labels=('Measurement', 'Setpoint'), loc='best'))

    def xylables_v_dq0(self, fig):
        self.update_axes(fig,
                         ylabel='$v_{\mathrm{dq0}}\,/\,\mathrm{V}$',
                         filename=f'{self.agent.history.df.shape[0]}_J_{self.agent.performance}_v_dq0',
                         legend=dict(handle_slice=slice(None, None, 3), labels=('Measurement', 'Setpoint'), loc='best'))

    def xylables_i_abc(self, fig):
        self.update_axes(fig,
                         ylabel='$i_{\mathrm{abc}}\,/\,\mathrm{A}$',
                         filename=f'{self.agent.history.df.shape[0]}_J_{self.agent.performance}_i_abc',
                         legend=dict(handle_slice=slice(None, None, 3), labels=('Measurement', 'Setpoint'), loc='best'))

    def xylables_i_dq0(self, fig):
        self.update_axes(fig,
                         ylabel='$i_{\mathrm{dq0}}\,/\,\mathrm{A}$',
                         filename=f'{self.agent.history.df.shape[0]}_J_{self.agent.performance}_i_dq0',
                         legend=dict(handle_slice=slice(None, None, 3), labels=('Measurement', 'Setpoint'), loc='best'))

    def update_axes(self, fig, title=None, xlabel=r'$t\,/\,\mathrm{s}$', ylabel=None, legend=None, filename=None):
        """
        General function to handle most of the standard modifications
        :param fig: figure to change
        :param title: optional title
        :param xlabel: optional label, time in seconds is default
        :param ylabel: optional ylabel
        :param legend: optional legend, can have optional key "handle_slice" which is used to slice the line handles for legend and set labels accordingly.
        :param filename: optional filename
        """
        ax = fig.gca()
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.grid(which='both')
        if legend is not None:
            if 'handle_slice' in legend:
                _slice = legend['handle_slice']
                del legend['handle_slice']
                plt.legend(handles=ax.lines[_slice], **legend)
            else:
                plt.legend(**legend)

        if self.save_results and filename is not None:
            for filetype in ['pgf', 'pdf']:
                fig.savefig(p.join(self.save_folder, f'{filename}.{filetype}'))
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
