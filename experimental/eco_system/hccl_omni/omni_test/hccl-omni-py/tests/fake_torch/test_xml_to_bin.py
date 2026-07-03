"""
Test XML to BIN conversion flow for HCCL-OMNI framework.

Reads XML files from xml_example/, splits them by rank, converts to binary
format, and verifies the output.
"""

import os
from pathlib import Path

import pytest

from hccl_omni.converter import xml_to_bin_converter
from hccl_omni.splitter import split_xml_by_rank


PROJECT_ROOT = Path(__file__).parent.parent.parent
XML_EXAMPLE_DIR = PROJECT_ROOT / 'xml_example'
DEFAULT_XML_FILE = '4pfullmesh.xml'


class TestXmlToBin:

    def test_split_and_convert_single_rank(self, tmp_path):
        """Split XML for rank 0 and convert to binary."""
        xml_path = XML_EXAMPLE_DIR / DEFAULT_XML_FILE
        if not xml_path.exists():
            pytest.skip(f"XML example file not found: {xml_path}")

        xml_content = xml_path.read_bytes()
        rank_xml = split_xml_by_rank(xml_content, rank=0)
        binary = xml_to_bin_converter(rank_xml)

        assert isinstance(binary, bytes)
        assert len(binary) > 0

    def test_full_flow_all_ranks(self, tmp_path):
        """Split and convert XML for all 4 ranks, save bin files."""
        xml_path = XML_EXAMPLE_DIR / DEFAULT_XML_FILE
        if not xml_path.exists():
            pytest.skip(f"XML example file not found: {xml_path}")

        xml_content = xml_path.read_bytes()
        num_ranks = 4

        for rank in range(num_ranks):
            rank_xml = split_xml_by_rank(xml_content, rank=rank)
            binary = xml_to_bin_converter(rank_xml)

            assert isinstance(binary, bytes)
            assert len(binary) > 0

            bin_file = tmp_path / f'rank_{rank}.bin'
            bin_file.write_bytes(binary)
            assert bin_file.exists()
            assert bin_file.stat().st_size == len(binary)

    def test_both_xml_files(self, tmp_path):
        """Verify both XML example files can be split and converted."""
        for xml_name in ['4pfullmesh.xml', '4pfullmesh_ccu_sch.xml']:
            xml_path = XML_EXAMPLE_DIR / xml_name
            if not xml_path.exists():
                pytest.skip(f"XML example file not found: {xml_path}")

            xml_content = xml_path.read_bytes()
            rank_xml = split_xml_by_rank(xml_content, rank=0)
            binary = xml_to_bin_converter(rank_xml)

            assert isinstance(binary, bytes)
            assert len(binary) > 0
