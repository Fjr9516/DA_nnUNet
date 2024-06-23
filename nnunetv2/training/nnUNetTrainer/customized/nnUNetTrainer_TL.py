import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_TL(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        assert torch.cuda.is_available(), "This only works on GPU"
        self.params_to_train = [
        '.seg_layers.',
        ]

    def unfreeze_layers(self, params_to_train):
        if not self.was_initialized:
            self.initialize()

        self.print_to_log_file(f"################### Train on params:{params_to_train} ###################",
                               also_print_to_console=True, add_timestamp=False)

        # unfreeze some layers
        for name, param in self.network.named_parameters():
            param.requires_grad = True if any([i in name for i in params_to_train]) else False
            # verify
            self.print_to_log_file(f"{name} \t {param.requires_grad}", also_print_to_console=True, add_timestamp=False)
        self.print_to_log_file("################### Done ###################",
                               also_print_to_console=True, add_timestamp=False)

    def run_training(self):
        # JR: okay, I only added a function that takes params_to_train as input
        # to unfreeze these layers, and all the other layers are frozen
        self.unfreeze_layers(self.params_to_train)
        super().run_training()

# TL: Load pre-trained model and fine tune segmentation layers from scratch
class nnUNetTrainer_TL_300epochs(nnUNetTrainer_TL):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300

class nnUNetTrainer_TL_500epochs(nnUNetTrainer_TL):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnUNetTrainer_TL_5epochs(nnUNetTrainer_TL):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        """used for debugging plans etc"""
        self.num_epochs = 5

# TL+FT: Load pre-trained TL model and fine tune with small lr for 300 eps
class nnUNetTrainer_300epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300

class nnUNetTrainer_TL_FT_1en5_300epochs(nnUNetTrainer_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-5

class nnUNetTrainer_TL_FT_1en4_300epochs(nnUNetTrainer_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4

class nnUNetTrainer_TL_FT_1en3_300epochs(nnUNetTrainer_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

# TL+FTen: Load pre-trained TL model and fine tune with small lr for 300 eps
class nnUNetTrainer_TL_FTen_1en5_300epochs(nnUNetTrainer_TL_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.params_to_train = [
            'encoder.',
            '.seg_layers.', # JR: should always unfreeze segmentation related layers
        ]
        self.initial_lr = 1e-5

class nnUNetTrainer_TL_FTen_1en4_300epochs(nnUNetTrainer_TL_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.params_to_train = [
            'encoder.',
            '.seg_layers.', # JR: should always unfreeze segmentation related layers
        ]
        self.initial_lr = 1e-4

class nnUNetTrainer_TL_FTen_1en3_300epochs(nnUNetTrainer_TL_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.params_to_train = [
            'encoder.',
            '.seg_layers.', # JR: should always unfreeze segmentation related layers
        ]
        self.initial_lr = 1e-3

# TL+FTde: Load pre-trained TL model and fine tune with small lr for 300 eps
class nnUNetTrainer_TL_FTde_1en5_300epochs(nnUNetTrainer_TL_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.params_to_train = [
            'decoder.',
        ]
        self.initial_lr = 1e-5

class nnUNetTrainer_TL_FTde_1en4_300epochs(nnUNetTrainer_TL_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.params_to_train = [
            'decoder.',
        ]
        self.initial_lr = 1e-4

class nnUNetTrainer_TL_FTde_1en3_300epochs(nnUNetTrainer_TL_300epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.params_to_train = [
            'decoder.',
        ]
        self.initial_lr = 1e-3

class nnUNetTrainer_500ep_noDS(nnUNetTrainer_TL_500epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

class nnUNetTrainer_500epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnUNetTrainer_500ep_noDS_correct(nnUNetTrainer_500epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False