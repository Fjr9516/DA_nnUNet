import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
import numpy as np
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from typing import Tuple
from time import time, sleep
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d_balanced import nnUNetDataLoader3D_Balanced
from torch import nn, autocast
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.collate_outputs import collate_outputs
from typing import List
from torch import distributed as dist

import multiprocessing
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from time import sleep
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
import warnings
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

# customized:
from .losses import DC_BCE_masked_loss
from .unet_da import PlainConvUNet_DA, PlainConvUNet_DAonDecoder
from .schedulers import DALRScheduler, GRLAlphaScheduler
from .logger import My_nnUNetLogger
from .predictors import My_nnUNetPredictor

# some examples given by nnUNet
class nnUNetTrainerCELoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = RobustCrossEntropyLoss(
            weight=None, ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCELoss_5epochs(nnUNetTrainerCELoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5

# ====== DANN ======
class nnUNetTrainerDA(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        ## SB: Define Domain-specific config
        self.target_domain = 'BraTS-PED'
        self.domain_mapping = {
            'BraTS-PED': 0,
            'BraTS-GLI': 1
        }
        self.target_domain_label = self.domain_mapping[self.target_domain]
        ## SB: Define Domain Classifier loss
        self.da_loss = RobustCrossEntropyLoss()
        self.da_weight = 1e-2

        # JR: define a grl_alpha_scheduler
        self.optimizer = self.lr_scheduler = self.grl_alpha_scheduler = None

        # JR: use cunstomized Logger
        self.logger = My_nnUNetLogger()

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D_Balanced(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       classes=list(self.domain_mapping.keys()),
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3D_Balanced(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        classes=list(self.domain_mapping.keys()),
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                self.enable_deep_supervision,
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            # JR: initialize also grl_alpha_scheduler
            self.optimizer, self.lr_scheduler, self.grl_alpha_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _build_loss(self):
        if self.label_manager.has_regions:
            ## SB: Custom Dice + CE Masked Loss, Target Domain batch components are masked out from loss computation
            ## SB: IMPORTANT! DA Loss is not defined here, since this loss is wrapped in the Deep Supervision
            loss = DC_BCE_masked_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=True,
                                   masked_label=self.target_domain_label,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            assert self.label_manager.has_regions, "only support regions by this trainer"

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # JR: update grl_alpha and record
        self.grl_alpha_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(
            f"Current GRL param alpha: {np.round(self.network.domain_classifier.get_alpha(), decimals=5)}")
        self.logger.log('grl_alphas', self.network.domain_classifier.get_alpha(), self.current_epoch)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes) # JR: 6

        dim = len(configuration_manager.conv_kernel_sizes[0]) # JR: 3
        conv_op = convert_dim_to_conv_op(dim) # JR: nn.Conv3d

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = 'PlainConvUNet_DAonDecoder' # JR: Need to specify here
        mapping = {
            'PlainConvUNet_DA': PlainConvUNet_DA,
            'PlainConvUNet_DAonDecoder': PlainConvUNet_DAonDecoder,
        }
        kwargs = {
            'PlainConvUNet_DA': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'PlainConvUNet_DAonDecoder': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
                'on_ith_decoder': 4, # can be 1, 2, 3, 4
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accommodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        # JR: define Domain Adaptation learning rate scheduler
        lr_scheduler = DALRScheduler(optimizer, self.initial_lr, self.num_epochs)
        grl_alpha_scheduler = GRLAlphaScheduler(self.network, self.num_epochs)
        return optimizer, lr_scheduler, grl_alpha_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        ## SB: Batch Domain from batch.properties
        batch_domain = torch.Tensor([self.domain_mapping[prop['domain']] for prop in batch['properties']]).long().to(
            self.device)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_S, output_D = self.network(data)
            ## SB: We fake the domain classifier prediction, to be later removed and replaced by actual network output
            # domain_pred = torch.randn(output[0].shape[0], len(self.domain_mapping)).to(self.device)

            ## SB: Adding batch_domain to mask target domain out
            if self.enable_deep_supervision:
                l_s = self.loss(output_S, target,
                            [batch_domain] * len(self._get_deep_supervision_scales()))  # JR: change to self._get...
            else:
                l_s = self.loss(output_S, target,
                                batch_domain)
            l_d = self.da_loss(output_D, batch_domain)
            ## SB: Computing DA Loss
            l = l_s + self.da_weight * l_d

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # JR: calculate domain classifier accuracy:
        D_acc_tr = (torch.argmax(output_D, 1) == batch_domain).float().mean()
        if l.detach().cpu().numpy() > 100:
            print({'loss': l.detach().cpu().numpy(),
                    'loss_s': l_s.detach().cpu().numpy(),
                    'loss_d': l_d.detach().cpu().numpy(),
                    'D_acc_tr': D_acc_tr.detach().cpu().numpy(),
                    })
            print(output_D, batch_domain)
        return {'loss': l.detach().cpu().numpy(),
                'loss_s': l_s.detach().cpu().numpy(),
                'loss_d': l_d.detach().cpu().numpy(),
                'D_acc_tr': D_acc_tr.detach().cpu().numpy(),
                }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        ## SB: Batch Domain from batch.properties
        batch_domain = torch.Tensor([self.domain_mapping[prop['domain']] for prop in batch['properties']]).long().to(
            self.device)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, output_D = self.network(data)
            del data
            ## SB: Adding batch_domain to mask target domain out
            # JR:
            if self.enable_deep_supervision:
                l_s = self.loss(output, target,
                            [batch_domain] * len(self._get_deep_supervision_scales()))  # JR: change to self._get...
            else:
                l_s = self.loss(output, target,
                                batch_domain)
            l_d = self.da_loss(output_D, batch_domain)
            ## SB: Computing DA Loss
            l = l_s + self.da_weight * l_d

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg
        
        masked_label = 1
        mask_size = list(target.shape)
        mask_size[1] = 1
        mask = torch.ones(mask_size, dtype=torch.bool).to("cuda")
        mask[batch_domain == masked_label, :] = 0

        tp_domain0, fp_domain0, fn_domain0, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard_domain0 = tp_domain0.detach().cpu().numpy()
        fp_hard_domain0 = fp_domain0.detach().cpu().numpy()
        fn_hard_domain0 = fn_domain0.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard_domain0 = tp_hard_domain0[1:]
            fp_hard_domain0 = fp_hard_domain0[1:]
            fn_hard_domain0 = fn_hard_domain0[1:]

        masked_label = 0
        mask_size = list(target.shape)
        mask_size[1] = 1
        mask = torch.ones(mask_size, dtype=torch.bool).to("cuda")
        mask[batch_domain == masked_label, :] = 0


        tp_domain1, fp_domain1, fn_domain1, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard_domain1 = tp_domain1.detach().cpu().numpy()
        fp_hard_domain1 = fp_domain1.detach().cpu().numpy()
        fn_hard_domain1 = fn_domain1.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard_domain1 = tp_hard_domain1[1:]
            fp_hard_domain1 = fp_hard_domain1[1:]
            fn_hard_domain1 = fn_hard_domain1[1:]

        # JR: calculate domain classifier accuracy:
        D_acc_val = (torch.argmax(output_D, 1) == batch_domain).float().mean()

        return {'loss': l.detach().cpu().numpy(),
                'tp_hard_domain0': tp_hard_domain0,
                'fp_hard_domain0': fp_hard_domain0,
                'fn_hard_domain0': fn_hard_domain0,
                'tp_hard_domain1': tp_hard_domain1,
                'fp_hard_domain1': fp_hard_domain1,
                'fn_hard_domain1': fn_hard_domain1,
                'D_acc_val': D_acc_val.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        # JR: log s and d losses
        loss_s_here = np.mean(outputs['loss_s'])
        loss_d_here = np.mean(outputs['loss_d'])
        D_acc_here = np.mean(outputs['D_acc_tr'])

        self.logger.log('train_losses_s', loss_s_here, self.current_epoch)
        self.logger.log('train_losses_d', loss_d_here, self.current_epoch)
        self.logger.log('train_D_acc', D_acc_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp_domain0 = np.sum(outputs_collated['tp_hard_domain0'], 0)
        fp_domain0 = np.sum(outputs_collated['fp_hard_domain0'], 0)
        fn_domain0 = np.sum(outputs_collated['fn_hard_domain0'], 0)

        tp_domain1 = np.sum(outputs_collated['tp_hard_domain1'], 0)
        fp_domain1 = np.sum(outputs_collated['fp_hard_domain1'], 0)
        fn_domain1 = np.sum(outputs_collated['fn_hard_domain1'], 0)

        if self.is_ddp:
            ...
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class_domain0 = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_domain0, fp_domain0, fn_domain0)]]
        mean_fg_dice_domain0 = np.nanmean(global_dc_per_class_domain0)
        self.logger.log('mean_fg_dice_domain0', mean_fg_dice_domain0, self.current_epoch)
        self.logger.log('dice_per_class_or_region_domain0', global_dc_per_class_domain0, self.current_epoch)
        global_dc_per_class_domain1 = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_domain1, fp_domain1, fn_domain1)]]
        mean_fg_dice_domain1 = np.nanmean(global_dc_per_class_domain1)
        self.logger.log('mean_fg_dice_domain1', mean_fg_dice_domain1, self.current_epoch)
        self.logger.log('dice_per_class_or_region_domain1', global_dc_per_class_domain1, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # JR
        D_acc_here = np.mean(outputs_collated['D_acc_val'])

        self.logger.log('val_D_acc', D_acc_here, self.current_epoch)


    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice [Domain 0]', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region_domain0'][-1]])
        self.print_to_log_file('Pseudo dice [Domain 1]', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region_domain1'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice_domain0'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice_domain0'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        # JR
        print("== Calling perform_actual_validation! ==")
        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = My_nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=True,# JR: verbose True
                                    verbose_preprocessing=False, allow_tqdm=False,
                                    )
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.keys()):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

class nnUNetTrainerDA_5epochs(nnUNetTrainerDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        """used for debugging plans etc"""
        self.num_epochs = 5

class nnUNetTrainerDA_1epochs(nnUNetTrainerDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        """used for debugging plans etc"""
        self.num_epochs = 1

# === some variants for tuning hyperparameters ===
# 0/ epoch: default 1000, BS = 4, so let us use 500
class nnUNetTrainerDA_500epochs(nnUNetTrainerDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnUNetTrainerDA_1000epochs(nnUNetTrainerDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000

# 1/ da_weight: to control strength of D in loss: default 1e-2
class nnUNetTrainerDA_500ep_DAweight1(nnUNetTrainerDA_500epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.da_weight = 1.0

class nnUNetTrainerDA_500ep_DAweight1en1(nnUNetTrainerDA_500epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.da_weight = 0.1

# if enable deep supervision, Segmenter is much powerful, try train with more weight on D?
class nnUNetTrainerDA_500ep_DAweight100(nnUNetTrainerDA_500epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.da_weight = 100.0

# or disenable deep supervision
class nnUNetTrainerDA_500ep_noDS(nnUNetTrainerDA_500epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

class nnUNetTrainerDA_1000ep_noDS(nnUNetTrainerDA_1000epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

# 2/ three params in GRLalphascheduler: default: p1=0.2,p2=0.7;alpha_max=3

# 4/ some params in Domain Classifier: output_channels = 100 # conv kernel channel/ num_convs = 2 # number of conv per block
class nnUNetTrainerDA_500ep_noDS_4Convs(nnUNetTrainerDA_500ep_noDS):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)  # JR: 6

        dim = len(configuration_manager.conv_kernel_sizes[0])  # JR: 3
        conv_op = convert_dim_to_conv_op(dim)  # JR: nn.Conv3d

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = 'PlainConvUNet_DAonDecoder'  # JR: Need to specify here
        mapping = {
            'PlainConvUNet_DA': PlainConvUNet_DA,
            'PlainConvUNet_DAonDecoder': PlainConvUNet_DAonDecoder,
        }
        kwargs = {
            'PlainConvUNet_DA': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'PlainConvUNet_DAonDecoder': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
                'on_ith_decoder': 4,  # can be 1, 2, 3, 4
                'num_convblock_domain_classifier': 4,
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accommodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model

class nnUNetTrainerDA_500ep_DS_4Convs(nnUNetTrainerDA_500ep_noDS_4Convs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True

class nnUNetTrainerDA_1000ep_noDS_4Convs(nnUNetTrainerDA_1000ep_noDS):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)  # JR: 6

        dim = len(configuration_manager.conv_kernel_sizes[0])  # JR: 3
        conv_op = convert_dim_to_conv_op(dim)  # JR: nn.Conv3d

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = 'PlainConvUNet_DAonDecoder'  # JR: Need to specify here
        mapping = {
            'PlainConvUNet_DA': PlainConvUNet_DA,
            'PlainConvUNet_DAonDecoder': PlainConvUNet_DAonDecoder,
        }
        kwargs = {
            'PlainConvUNet_DA': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'PlainConvUNet_DAonDecoder': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
                'on_ith_decoder': 4,  # can be 1, 2, 3, 4
                'num_convblock_domain_classifier': 4,
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accommodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model

# 5/ other architecture
class nnUNetTrainerDA_500ep_onEncoder(nnUNetTrainerDA_500epochs):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes) # JR: 6

        dim = len(configuration_manager.conv_kernel_sizes[0]) # JR: 3
        conv_op = convert_dim_to_conv_op(dim) # JR: nn.Conv3d

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = 'PlainConvUNet_DA' # JR: Need to specify here
        mapping = {
            'PlainConvUNet_DA': PlainConvUNet_DA,
            'PlainConvUNet_DAonDecoder': PlainConvUNet_DAonDecoder,
        }
        kwargs = {
            'PlainConvUNet_DA': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'PlainConvUNet_DAonDecoder': {
                'conv_bias': True,
                'alpha': 0.,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
                'on_ith_decoder': 4, # can be 1, 2, 3, 4
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accommodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model