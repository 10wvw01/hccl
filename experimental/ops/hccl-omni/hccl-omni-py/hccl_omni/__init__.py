'''
HCCL-OMNI: A PyTorch-based communication operator framework for Ascend NPU.
'''


# Configuration class - defined first to avoid import cycles
class HcclOmniConfig:
    '''Configuration for HCCL-OMNI operator generation.'''

    def __init__(
        self,
        op_key: str | None = None,
        ins_file_path: str | None = None,
        ins_gen_mode: str | None = None,
    ):
        '''
        Initialize HCCL-OMNI configuration.

        Args:
            op_key: User-defined operator identifier
            ins_file_path: Path for instruction file storage
            ins_gen_mode: Instruction generation mode (AlgSynthesizer/DSL/Vanilla)
        '''
        self.op_key = op_key
        self.ins_file_path = ins_file_path
        self.ins_gen_mode = ins_gen_mode

    def to_dict(self) -> dict[str, object]:
        '''Convert configuration to dictionary.'''
        return {
            'op_key': self.op_key,
            'ins_file_path': self.ins_file_path,
            'ins_gen_mode': self.ins_gen_mode,
        }


# Now import other modules
from .jit import jit

from .op_param import OpName, ReduceOp

# Re-export the main decorator and interpreter modes
__all__ = ['jit', 'HcclOmniConfig', 'OpName', 'ReduceOp']
