'''
XML to binary converter for HCCL-OMNI framework.

Converts XML instruction tags to binary format according to specified bit fields.
All XML attributes use camelCase naming.
'''

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Type


# ============== Enums for field values ==============

class Opcode(Enum):
    RES_REQUEST = 0
    PRE_SYNC_INTER_THREADS = 1
    POST_SYNC_INTER_THREADS = 2
    LOCAL_COPY = 3
    LOCAL_REDUCE = 4
    SEND_RECV_WRITE = 5
    SEND_WRITE = 6
    RECV_WRITE = 7
    SEND_RECV_WRITE_REDUCE = 8
    SEND_WRITE_REDUCE = 9
    RECV_WRITE_REDUCE = 10
    SEND_RECV_READ = 11
    SEND_READ = 12
    RECV_READ = 13
    SEND_RECV_READ_REDUCE = 14
    SEND_READ_REDUCE = 15
    RECV_READ_REDUCE = 16
    SEND_RECV_WRITE_DPU = 17
    SEND_WRITE_DPU = 18
    RECV_WRITE_DPU = 19
    GROUP_BROADCAST = 20
    GROUP_REDUCE = 21
    WAIT_EVENT = 22

class LinkType(Enum):
    P2P = 0

class LinkProto(Enum):
    HCCS = 0
    PCIE = 1
    RDMA = 2
    UB = 3

class ReduceType(Enum):
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3

class DataType(Enum):
    FLOAT16 = 0
    FLOAT32 = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5

class BufferType(Enum):
    HCCL_BUFFER = 0
    INPUT = 1
    OUTPUT = 2


# ============== Bit field definitions ==============

@dataclass
class BitField:
    '''Definition of a single bit field in binary encoding.'''
    name: str                                   # XML attribute name in camelCase
    width: int                                  # Bit width
    offset: int = None                          # Bit offset from LSB (0), use None to indicate auto deduced
    enum_type: Type[Enum] | None = None         # Associated enum type for value mapping
    default: int = 0                            # Default value if attribute missing
    is_bool: bool = False                       # Whether this is a boolean flag (true/false)

    @property
    def mask(self) -> int:
        '''Bit mask for this field.'''
        return ((1 << self.width) - 1) << self.offset


# Resource request instruction header fields
RESREQ_INSTRUCTION_HEADER_FIELDS = [
    BitField('opCode', 5, enum_type=Opcode),
    BitField('slaveThreadNum', 5),
    BitField('notifyNumOnMainThread', 5),
    BitField('notifyNumPerThread', 5),
    BitField('netLayerNum', 2, default=1),
    BitField('chanCount', 8, default=1),
]

# Channel fields
CHANNEL_ENTRY_FIELDS = [
    BitField('netLayerId', 5, default=1),
    BitField('localRank', 10),
    BitField('remoteRank', 10),
    BitField('linkProto', 3, enum_type=LinkProto),
]

# PRE/PST sync instruction header fields
PRE_PST_SYNC_INSTRUCTION_HEADER_FIELDS = [
    BitField('opCode', 5, enum_type=Opcode),
    BitField('mainThreadIdx', 5),
    BitField('subThreadNum', 5),
]

# Subthread entry fields
SUBTHREAD_ENTRY_FIELDS = [
    BitField('subThreadId', 8)
]

# Control instruction header fields
CONTROL_INSTRUCTION_HEADER_FIELDS = [
    BitField('opCode', 5, enum_type=Opcode),
    BitField('netLayerId', 2),
    BitField('linkProto', 3, enum_type=LinkProto),
    BitField('sliceNum', 10, default=1),
    BitField('srcSliceNum', 4, default=1),
    BitField('dstSliceNum', 4, default=1),
    BitField('notifyFlag', 1, is_bool=True),
    BitField('notifyThread', 4),
    BitField('waitFlag', 1, is_bool=True),
    BitField('waitThread', 4),
    BitField('threadIdx', 5),
    BitField('reduceType', 2),
    BitField('inputDataType', 4, enum_type=DataType),
    BitField('outputDataType', 4, enum_type=DataType),
    BitField('instructionId', 10),
]

# Src slice entry fields
SRC_SLICE_ENTRY_FIELDS = [
    BitField('bufferType', 2, enum_type=BufferType),
    BitField('sliceIdx', 10),
    BitField('rankId', 10),
    BitField('recvRankId', 10),
    BitField('cnt', 10),
    BitField('gap', 10),
]

# Dst slice entry fields
DST_SLICE_ENTRY_FIELDS = [
    BitField('bufferType', 2, enum_type=BufferType),
    BitField('sliceIdx', 10),
    BitField('rankId', 10),
    BitField('recvRankId', 10),
    BitField('cnt', 10),
    BitField('gap', 10),
]

# Instruction config: (header_fields, child_configs)
# child_configs: list of (tag_name, fields, byte_size, count_attr)
RESREQ_CONFIG = (
    RESREQ_INSTRUCTION_HEADER_FIELDS,
    [('channel', CHANNEL_ENTRY_FIELDS, 4, 'chanCount')]
)
PRE_PST_SYNC_CONFIG = (
    PRE_PST_SYNC_INSTRUCTION_HEADER_FIELDS,
    [('subThread', SUBTHREAD_ENTRY_FIELDS, 1, 'subThreadNum')]
)
CONTROL_CONFIG = (
    CONTROL_INSTRUCTION_HEADER_FIELDS,
    [
        ('srcSlice', SRC_SLICE_ENTRY_FIELDS, 8, 'srcSliceNum'),
        ('dstSlice', DST_SLICE_ENTRY_FIELDS, 8, 'dstSliceNum'),
    ]
)

# Map opCode to instruction config
OPCODE_CONFIG_MAP = {
    Opcode.RES_REQUEST: RESREQ_CONFIG,
    Opcode.PRE_SYNC_INTER_THREADS: PRE_PST_SYNC_CONFIG,
    Opcode.POST_SYNC_INTER_THREADS: PRE_PST_SYNC_CONFIG,
    Opcode.WAIT_EVENT: CONTROL_CONFIG,
}

INSTRUCTION_HEADER_BYTE_SIZE = 8

# ============== Helper functions ==============

def _enum_match(enum_cls, value_str: str) -> int:
    if value_str is None:
        raise ValueError(f"Value cannot be None for enum {enum_cls.__name__}")

    # Normalize input: lowercase for comparison
    normalized = value_str.strip().upper()

    for enum_member in enum_cls:
        # Compare with enum name (converted to lowercase)
        if enum_member.name.upper() == normalized:
            return enum_member.value

    # Try to parse as snake case
    normalized = re.sub(r"(?<!^)(?=[A-Z])", '_', value_str.strip()).upper()

    for enum_member in enum_cls:
        # Compare with enum name (converted to lowercase)
        if enum_member.name.upper() == normalized:
            return enum_member.value

    # Try to parse as integer directly
    try:
        return int(value_str)
    except ValueError:
        raise ValueError(
            f"No matching enum value found for '{value_str}' in {enum_cls.__name__}. "
            f"Available values: {[e.name for e in enum_cls]}"
        )


def _parse_int(value_str: str | None, default: int = 0) -> int:
    if value_str is None:
        return default
    try:
        return int(value_str)
    except ValueError:
        return default


def _set_bits(value: int, width: int, offset: int, target: int) -> int:
    mask = ((1 << width) - 1) << offset
    cleared = target & ~mask
    return cleared | ((value << offset) & mask)


def _extract_field_value(element: ET.Element, field: BitField) -> int:
    attr_value = element.get(field.name)

    if attr_value is None:
        return field.default

    if field.is_bool:
        # Boolean flag: "true" -> 1, anything else -> 0
        return 1 if attr_value.lower() == 'true' else 0

    if field.enum_type is not None:
        # Enum field: map string to enum value
        try:
            return _enum_match(field.enum_type, attr_value)
        except ValueError:
            # If enum mapping fails, try to parse as integer
            try:
                return int(attr_value)
            except ValueError:
                return field.default

    # Integer field
    try:
        return int(attr_value)
    except ValueError:
        return field.default


def encode_fields(element: ET.Element, field_definitions: list[BitField], element_name: str = 'element') -> int:
    known_attrs = {field.name for field in field_definitions}
    for attr in element.attrib:
        if attr not in known_attrs:
            raise ValueError(f"Unknown attribute '{attr}' in <{element_name}>")

    result = 0
    current_offset = 0

    for field in field_definitions:
        value = _extract_field_value(element, field)
        if field.offset is None:
            field.offset = current_offset
        max_value = (1 << field.width) - 1
        if value > max_value or value < 0:
            raise ValueError(
                f"Value {value} for field '{field.name}' in <{element_name}> exceeds "
                f"maximum representable value {max_value} for width {field.width} bits"
            )
        result = _set_bits(value, field.width, field.offset, result)
        current_offset = field.offset + field.width

    return result


def encode_child_elements(
    parent: ET.Element,
    child_tag: str,
    field_definitions: list[BitField],
    expected_count: int | None = None,
    parent_name: str = 'parent',
    known_child_tags: set[str] | None = None
) -> list[int]:
    all_children = list(parent)
    for child in all_children:
        if child.tag != child_tag and known_child_tags is not None and child.tag not in known_child_tags:
            import warnings
            warnings.warn(f"Unknown child tag <{child.tag}> in <{parent_name}>")

    child_elements = parent.findall(child_tag)
    if expected_count is not None and len(child_elements) != expected_count:
        raise ValueError(
            f"Expected {expected_count} <{child_tag}> elements in <{parent_name}>, "
            f"but found {len(child_elements)}"
        )

    return [encode_fields(child, field_definitions, child_tag) for child in child_elements]


def _get_opcode(element: ET.Element) -> Opcode:
    op_code_str = element.get('opCode')
    if op_code_str is None:
        raise ValueError("Expected opCode attr in <instruction>, but nothing found!")

    return Opcode(_enum_match(Opcode, op_code_str))


def _get_instruction_config(op_code: Opcode) -> tuple:
    return OPCODE_CONFIG_MAP.get(op_code, CONTROL_CONFIG)


def xml_to_bin_converter(xml_content: bytes) -> bytes:
    instruction_tags = []
    try:
        root = ET.fromstring(xml_content)
        instruction_tags = root.findall('.//instruction')
        if not instruction_tags and root.tag == 'instruction':
            instruction_tags = [root]
    except ET.ParseError:
        import re
        xml_str = xml_content.decode('utf-8')
        pattern = r"<instruction([^>]*)>(.*?)</instruction>"
        for attrs, content in re.findall(pattern, xml_str, re.DOTALL):
            instr_xml = f"<root><instruction{attrs}>{content}</instruction></root>"
            try:
                root = ET.fromstring(instr_xml.encode())
                instr_elem = root.find('instruction')
                if instr_elem is not None:
                    instruction_tags.append(instr_elem)
            except ET.ParseError:
                continue
        for attrs in re.findall(r"<instruction([^>]*)/>", xml_str):
            try:
                root = ET.fromstring(f"<root><instruction{attrs}/></root>".encode())
                instr_elem = root.find('instruction')
                if instr_elem is not None:
                    instruction_tags.append(instr_elem)
            except ET.ParseError:
                continue

    if not instruction_tags:
        return b""

    binary_parts = []
    for instr_elem in instruction_tags:
        op_code = _get_opcode(instr_elem)
        header_fields, child_configs = _get_instruction_config(op_code)

        header_value = encode_fields(instr_elem, header_fields, 'instruction')
        binary_parts.append(header_value.to_bytes(INSTRUCTION_HEADER_BYTE_SIZE, byteorder='little'))

        known_tags = set()
        if isinstance(child_configs, list):
            for child_tag, child_fields, byte_size, count_attr in child_configs:
                known_tags.add(child_tag)
        else:
            child_tag, child_fields, byte_size, count_attr = child_configs
            known_tags.add(child_tag)
            child_configs = [(child_tag, child_fields, byte_size, count_attr)]

        for child_tag, child_fields, byte_size, count_attr in child_configs:
            count_field = next((f for f in header_fields if f.name == count_attr), None)
            default_count = count_field.default if count_field else 1
            count = _parse_int(instr_elem.get(count_attr), default=default_count)
            entries = encode_child_elements(
                instr_elem, child_tag, child_fields, count, 'instruction', known_tags
            )
            for entry in entries:
                binary_parts.append(entry.to_bytes(byte_size, byteorder='little'))

    return b"".join(binary_parts)
