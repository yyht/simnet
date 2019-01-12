
from copy import copy

from model.dsmm.esim import ESIMDecAttBaseModel


class DecAtt(ESIMDecAttBaseModel):
    def __init__(self, *args, **kargs):
        super(DecAtt, self).__init__(*args, **kargs)

    def build_placeholder(self, config):
        config = copy(config)
        # model config
        config.update({
            "encode_method": "project",
            "attend_method": ["ave", "max", "min", "self-attention"],

            "project_type": "fc",
            "project_hidden_units": [256, 128, 64],
            "project_dropouts": [0.2, 0.2, 0.2],

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [256, 128],
            "fc_dropouts": [0.3, 0.3],
        })
        super(DecAtt, self).build_placeholder(config)