import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss_fn_class, loss_fn_domain, metric_ftns, optimizer, config, device,
                 data_loader_source, valid_data_loader_source=None, data_loader_target=None,
                 valid_data_loader_target=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.loss_fn_class = loss_fn_class
        self.loss_fn_domain = loss_fn_domain
        self.data_loader_source = data_loader_source
        self.valid_data_loader_source = valid_data_loader_source
        self.data_loader_target = data_loader_target
        self.valid_data_loader_target = valid_data_loader_target
        self.model.to(self.device)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = min(len(self.data_loader_source),
                                 len(self.data_loader_target))
        else:
            # FIXME: implement source/target style training or remove this feature
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        # FIXME: handle validation round
        self.do_validation = None
        # self.valid_data_loader = valid_data_loader
        #self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.data_loader_source.batch_size))

        self.train_metrics = MetricTracker(
            'loss', 'class_loss', 'domain_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', 'class_loss', 'domain_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Setting model into train mode, required_grad
        self.model.train()
        # Reset all metric in metric dataframe
        self.train_metrics.reset()
        batch_idx = 0
        for source, target in zip(self.data_loader_source, self.data_loader_target):

            # source, target = source.to(self.device), target.to(self.device)

            # Calculate training progress and GRL λ
            p = float(batch_idx + (epoch-1) * self.len_epoch) / \
                (self.epochs * self.len_epoch)
            λ = 2. / (1. + np.exp(-10 * p)) - 1

            # === Train on source domain
            X_source, y_source = source
            X_source, y_source = X_source.to(
                self.device), y_source.to(self.device)

            # generate source domain labels: 0
            y_s_domain = torch.zeros(X_source.shape[0], dtype=torch.float32)
            y_s_domain = y_s_domain.to(self.device)

            class_pred_source, domain_pred_source = self.model(X_source, λ)
            # source classification loss
            loss_s_label = self.loss_fn_class(
                class_pred_source.squeeze(), y_source)

            # Compress from tensor size batch*1*1*1 => batch
            domain_pred_source = torch.squeeze(domain_pred_source)
            loss_s_domain = self.loss_fn_domain(
                domain_pred_source, y_s_domain)  # source domain loss (via GRL)

            # === Train on target domain
            X_target, _ = target
            # generate source domain labels: 0
            y_t_domain = torch.ones(X_target.shape[0], dtype=torch.float32)
            X_target = X_target.to(self.device)
            y_t_domain = y_t_domain.to(self.device)
            _, domain_pred_target = self.model(X_target, λ)

            domain_pred_target = torch.squeeze(domain_pred_target)
            loss_t_domain = self.loss_fn_domain(
                domain_pred_target, y_t_domain)  # source domain loss (via GRL)

            # === Optimizer ====

            self.optimizer.zero_grad()
            loss_s_domain = torch.log(loss_s_domain + 1e-9)
            loss = loss_t_domain + loss_s_domain + loss_s_label

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch-1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('class_loss', loss_s_label.item())
            self.train_metrics.update('domain_loss', loss_s_domain.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(class_pred_source, y_source))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.4f} Source class loss: {loss_s_label.item():3f} Source domain loss {loss_s_domain.item():3f}')
                self.writer.add_image('input', make_grid(
                    X_source.cpu(), nrow=4, normalize=True))

            batch_idx += 1
            if batch_idx == self.len_epoch:
                break
        # Average the accumulated result to log the result
        log = self.train_metrics.result()
        # update lambda value to metric tracker
        log["lambda"] = λ
        # Run validation after each epoch if validation dataloader is available.
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Set model to evaluation mode, required_grad = False
        # disables dropout and has batch norm use the entire population statistics
        self.model.eval()
        # Reset validation metrics in dataframe for a new validation round
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(
                    X_source.cpu(), nrow=4, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader_source, 'n_samples'):
            current = batch_idx * self.data_loader_source.batch_size
            total = self.data_loader_source.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
