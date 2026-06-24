'''
XML splitting utilities for HCCL-OMNI framework.
'''

import xml.etree.ElementTree as ET


def split_xml_by_rank(xml_content: bytes, rank: int | None = None,
                      world_size: int | None = None) -> bytes:
    '''
    Split XML content for the given rank.
    Extracts content from NPU tags with matching rankId attribute.

    Args:
        xml_content: XML content as bytes
        rank: Rank ID to extract. If None, attempts to get from torch.distributed
        world_size: Total number of ranks. If None, attempts to get from torch.distributed

    Note:
        When rank is not provided, this function requires torch.distributed to be initialized.
    '''
    # Lazy import torch.distributed only when rank is not provided
    # world_size parameter is kept for API compatibility but not used in current implementation
    if rank is None:
        try:
            import torch.distributed as dist
        except ImportError:
            raise ImportError(
                'rank is None and torch.distributed is not available. '
                'Please provide explicit rank value.'
            )
        rank = dist.get_rank()

    root = ET.fromstring(xml_content)
    npu_tags = root.findall('NPU')

    if not npu_tags:
        return b'<?xml version="1.0"?><root/>'

    matching_elements = []

    for npu_tag in npu_tags:
        rank_id_attr = npu_tag.get('rankId')

        if rank_id_attr is None:
            continue

        try:
            tag_rank_id = int(rank_id_attr)
        except ValueError:
            continue

        if tag_rank_id == rank:
            for child in npu_tag:
                matching_elements.append(child)

            if npu_tag.text and npu_tag.text.strip():
                text_element = ET.Element('text')
                text_element.text = npu_tag.text.strip()
                matching_elements.append(text_element)

    if not matching_elements:
        import logging
        logging.warning(f"No instruction data found for rank {rank}, returning empty XML")
        return b'<?xml version="1.0"?><root/>'

    new_root = ET.Element('root')
    for element in matching_elements:
        new_root.append(element)

    xml_str = ET.tostring(new_root, encoding='utf-8', method='xml')
    result = b'<?xml version="1.0"?>\n' + xml_str

    return result
