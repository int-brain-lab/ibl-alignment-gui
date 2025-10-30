from ibl_alignment_gui.plugins.cluster_features import setup as setup_cluster_popup


class Plugins:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_window.plugin_options = self.main_window.view.menu_widgets.addMenu('Plugins')
        self.main_window.plugins = dict()

        setup_cluster_popup(self.main_window)

    # def add_plugin(self, plugin):
    #     self.plugins.append(plugin)
    #
    # def get_plugins(self):
    #     return self.plugins

