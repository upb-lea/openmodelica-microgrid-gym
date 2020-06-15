from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.env.stochastic_components import Load, Noise
import matplotlib.pyplot as plt


class PlotManager:

    def __init__(self, used_agent: SafeOptAgent, used_r_load: Load, used_l_filt: Load, used_noise: Noise,
                 save_results: bool = False, save_folder: str = 'test_folder', show_plots: bool = True):
        """
        Class for plot configuration, knows agent, so can include agent params to title (e.g. performance)
        If more plots should be stored in save_folder, extend the corresponding label function with save command
        :param used_agent: agent used for experiment
        :param used_r_load: used resistor to plot R-value to title
        :param used_l_filt: used inductance to plot L-value to title
        :param used_noise: used noise to plot value to title
        :param save_results: if True, saves results to save_folder
        :param save_folder: folder name
        :param show_plots: if True, shows plots after each run
        """

        self.agent = used_agent
        self.r_load = used_r_load
        self.l_filt = used_l_filt
        self.noise = used_noise
        self.save_results = save_results
        self.save_folder = save_folder
        self.show_plots = show_plots

    def set_title(self):
        plt.title('Simulation: J = {:.2f}; R = {} \n L = {}; \n noise = {}'.format(self.agent.performance,
                                                                                   ['%.4f' % elem for elem in
                                                                                    self.r_load.gains],
                                                                                   ['%.6f' % elem for elem in
                                                                                    self.l_filt.gains],
                                                                                   ['%.4f' % elem for elem in
                                                                                    self.noise.gains]))

    def xylables_v_abc(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        plt.legend(ax.lines[::3], ('Measurement', 'Setpoint'), loc='best')
        if self.save_results:
            fig.savefig(
                self.save_folder + '/{}_J_{}_v_abc.pdf'.format(self.agent.history.df.shape[0], self.agent.performance))
            fig.savefig(
                self.save_folder + '/{}_J_{}_v_abc.pgf'.format(self.agent.history.df.shape[0], self.agent.performance))
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_v_dq0(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        plt.legend(ax.lines[::3], ('Measurement', 'Setpoint'), loc='best')
        if self.save_results:
            fig.savefig(
                self.save_folder + '/{}_J_{}_v_dq0.pdf'.format(self.agent.history.df.shape[0], self.agent.performance))
            fig.savefig(
                self.save_folder + '/{}_J_{}_v_dq0.pgf'.format(self.agent.history.df.shape[0], self.agent.performance))
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_i_abc(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        # plt.xlim(0.02, 0.0205)
        # plt.ylim(-5, -3)
        # plt.legend(['Measurement', None , None, 'Setpoint', None, None], loc='best')
        plt.legend(ax.lines[::3], ('Measurement', 'Setpoint'), loc='best')
        # self.set_title()
        if self.save_results:
            fig.savefig(
                self.save_folder + '/{}_J_{}_i_abc.pdf'.format(self.agent.history.df.shape[0], self.agent.performance))
            fig.savefig(
                self.save_folder + '/{}_J_{}_i_abc.pgf'.format(self.agent.history.df.shape[0], self.agent.performance))
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_i_dq0(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        # plotter.set_title()
        if self.save_results:
            fig.savefig(
                self.save_folder + '/{}_J_{}_i_dq0.pdf'.format(self.agent.history.df.shape[0], self.agent.performance))
            fig.savefig(
                self.save_folder + '/{}_J_{}_i_dq0.pgf'.format(self.agent.history.df.shape[0], self.agent.performance))
        plt.ylim(0, 36)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_i_hat(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{o estimate,abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_mdq0(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$m_{\mathrm{dq0}}\,/\,\mathrm{}$')
        plt.title('Simulation')
        ax.grid(which='both')
        # plt.ylim(0,36)
        if self.save_results:
            fig.savefig(self.save_folder + '/Sim_m_dq0.pdf')
            fig.savefig(self.save_folder + '/Sim_m_dq0.pgf')
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_mabc(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$m_{\mathrm{abc}}\,/\,\mathrm{}$')
        plt.title('Simulation')
        ax.grid(which='both')
        # plt.ylim(0,36)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_R(self, fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{123}}\,/\,\mathrm{\Omega}$')
        ax.grid(which='both')
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
