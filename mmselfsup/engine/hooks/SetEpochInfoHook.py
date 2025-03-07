
from mmengine.hooks import Hook

from mmselfsup.registry import HOOKS
from mmengine.model import is_model_wrapper

@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch+1
        model = runner.model
        if is_model_wrapper(model):
    # # 当模型被包在 wrapper 里时获取这个模型
            model = model.module
        model.set_epoch(epoch)
