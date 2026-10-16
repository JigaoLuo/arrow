# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import gc

import pytest

import pyarrow as pa

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import pyarrow.parquet.encryption as pe
except ImportError:
    pe = None
else:
    from pyarrow.tests.parquet.encryption import InMemoryKmsClient


# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not parquet'
pytestmark = pytest.mark.parquet

NUM_ROWS = 10_000


@pytest.fixture(scope='module')
def table():
    # 'int' has no nulls and only distinct values, 'str' has nulls and
    # repeated values
    return pa.table({
        'int': pa.array(range(NUM_ROWS), pa.int64()),
        'str': pa.array([None if i % 10 == 0 else f"value{i % 100:02d}"
                         for i in range(NUM_ROWS)]),
    })


def _write(table, **kwargs):
    # Use small pages so that every column chunk has several data pages
    kwargs.setdefault('data_page_size', 1024)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, **kwargs)
    return pq.ParquetFile(pa.BufferReader(sink.getvalue()))


def _data_pages(pages):
    return [page for page in pages if page.page_type != 'DICTIONARY_PAGE']


def test_iter_pages(table):
    pf = _write(table)
    for column in range(pf.metadata.num_columns):
        chunk = pf.metadata.row_group(0).column(column)
        pages = list(pf.iter_pages(row_group=0, column=column))
        assert all(isinstance(page, pq.ParquetPage) for page in pages)

        # The dictionary page comes first, followed by the data pages
        dictionary_page, data_pages = pages[0], pages[1:]
        assert dictionary_page.page_type == 'DICTIONARY_PAGE'
        assert dictionary_page.encoding == 'PLAIN'
        assert dictionary_page.num_values == len(
            table.column(column).drop_null().unique())
        assert dictionary_page.is_sorted is False
        assert dictionary_page.statistics is None
        assert dictionary_page.uncompressed_size is None

        assert len(data_pages) > 1
        assert all(page.page_type == 'DATA_PAGE' for page in data_pages)
        assert all(page.encoding == 'RLE_DICTIONARY' for page in data_pages)
        assert sum(page.num_values for page in data_pages) == chunk.num_values
        assert {page.encoding for page in pages} <= set(chunk.encodings)

        for page in pages:
            assert isinstance(page.buffer, pa.Buffer)
            assert len(page.buffer) > 0
        for page in data_pages:
            assert len(page.buffer) == page.uncompressed_size

    with pytest.raises(TypeError, match="Do not call"):
        pq.ParquetPage()


def test_iter_pages_without_dictionary(table):
    pf = _write(table, use_dictionary=False)
    chunk = pf.metadata.row_group(0).column(0)
    pages = list(pf.iter_pages(0, 0))
    assert len(pages) > 1
    assert all(page.page_type == 'DATA_PAGE' for page in pages)
    assert all(page.encoding == 'PLAIN' for page in pages)
    assert sum(page.num_values for page in pages) == chunk.num_values
    # Attributes of other page types are not available
    assert all(page.is_sorted is None for page in pages)


def test_iter_pages_data_page_v1(table):
    pf = _write(table, data_page_version='1.0')
    pages = _data_pages(pf.iter_pages(0, 1))
    assert len(pages) > 1
    for page in pages:
        assert page.page_type == 'DATA_PAGE'
        assert page.definition_level_encoding == 'RLE'
        assert page.repetition_level_encoding == 'RLE'
        # DATA_PAGE_V2 attributes are not available
        assert page.num_nulls is None
        assert page.num_rows is None
        assert page.definition_levels_byte_length is None
        assert page.repetition_levels_byte_length is None
        assert page.is_compressed is None


@pytest.mark.parametrize('compression', ['none', 'snappy'])
def test_iter_pages_data_page_v2(table, compression):
    pf = _write(table, data_page_version='2.0', compression=compression)
    chunk = pf.metadata.row_group(0).column(1)
    pages = _data_pages(pf.iter_pages(0, 1))
    assert len(pages) > 1
    for page in pages:
        assert page.page_type == 'DATA_PAGE_V2'
        assert page.is_compressed is (compression != 'none')
        # 'str' is nullable but not repeated
        assert page.definition_levels_byte_length > 0
        assert page.repetition_levels_byte_length == 0
        # DATA_PAGE attributes are not available
        assert page.definition_level_encoding is None
        assert page.repetition_level_encoding is None
    assert sum(page.num_values for page in pages) == chunk.num_values
    assert sum(page.num_rows for page in pages) == NUM_ROWS
    assert sum(page.num_nulls for page in pages) == table.column(1).null_count


@pytest.mark.parametrize('data_page_version', ['1.0', '2.0'])
@pytest.mark.parametrize('compression', ['snappy', 'gzip', 'zstd'])
def test_iter_pages_compressed(table, data_page_version, compression):
    if not pa.Codec.is_available(compression):
        pytest.skip(f"{compression} codec is not available")
    reference = _write(table, data_page_version=data_page_version,
                       compression='none')
    pf = _write(table, data_page_version=data_page_version,
                compression=compression)
    assert pf.metadata.row_group(0).column(0).compression == compression.upper()
    for column in range(pf.metadata.num_columns):
        pages = list(pf.iter_pages(0, column))
        expected_pages = list(reference.iter_pages(0, column))
        assert len(pages) == len(expected_pages) > 1
        for page, expected in zip(pages, expected_pages):
            assert page.page_type == expected.page_type
            assert page.encoding == expected.encoding
            assert page.num_values == expected.num_values
            assert page.uncompressed_size == expected.uncompressed_size
            # Page buffers are decompressed
            assert len(page.buffer) == len(expected.buffer)
            assert page.buffer.equals(expected.buffer)


@pytest.mark.parametrize('buffer_size', [0, 16, 4096])
def test_iter_pages_buffered_stream(table, buffer_size):
    # buffer_size > 0 reads the column chunk through a BufferedInputStream
    # instead of reading it at once
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, data_page_size=1024, compression='zstd')
    buf = sink.getvalue()
    expected_pages = list(
        pq.ParquetFile(pa.BufferReader(buf)).iter_pages(0, 1))
    pf = pq.ParquetFile(pa.BufferReader(buf), buffer_size=buffer_size)
    pages = list(pf.iter_pages(0, 1))
    assert len(pages) == len(expected_pages) > 1
    for page, expected in zip(pages, expected_pages):
        assert page.page_type == expected.page_type
        assert page.num_values == expected.num_values
        assert page.buffer.equals(expected.buffer)


def test_iter_pages_statistics(table):
    pf = _write(table, write_statistics=True)

    # 'int' is increasing, so page statistics partition the value range
    chunk_statistics = pf.metadata.row_group(0).column(0).statistics
    pages = _data_pages(pf.iter_pages(0, 0))
    first_value = 0
    for page in pages:
        statistics = page.statistics
        assert isinstance(statistics, pq.Statistics)
        assert statistics.has_min_max
        assert statistics.physical_type == 'INT64'
        assert statistics.min == first_value
        assert statistics.max == first_value + page.num_values - 1
        assert statistics.null_count == 0
        assert statistics.num_values == page.num_values
        first_value += page.num_values
    assert pages[0].statistics.min == chunk_statistics.min
    assert pages[-1].statistics.max == chunk_statistics.max

    # 'str' has nulls
    pages = _data_pages(pf.iter_pages(0, 1))
    for page in pages:
        statistics = page.statistics
        assert statistics.physical_type == 'BYTE_ARRAY'
        assert statistics.logical_type.type == 'STRING'
        assert isinstance(statistics.min, str)
        assert statistics.min <= statistics.max
        assert statistics.num_values == page.num_values - statistics.null_count
    assert sum(page.statistics.null_count for page in pages) == \
        table.column(1).null_count


def test_iter_pages_incorrect_statistics(parquet_test_datadir):
    # Written by parquet-mr 1.8.1, whose BYTE_ARRAY min/max values are not
    # reliable (PARQUET-251): like ColumnChunkMetaData.statistics, page
    # statistics are not exposed
    pf = pq.ParquetFile(parquet_test_datadir / 'datapage_v2.snappy.parquet')
    chunk = pf.metadata.row_group(0).column(0)
    assert chunk.physical_type == 'BYTE_ARRAY'
    assert chunk.statistics is None
    pages = _data_pages(pf.iter_pages(0, 0))
    assert len(pages) > 0
    assert all(page.statistics is None for page in pages)
    # The statistics of the other columns are fine
    chunk = pf.metadata.row_group(0).column(1)
    assert chunk.physical_type == 'INT32'
    assert chunk.statistics is not None
    pages = _data_pages(pf.iter_pages(0, 1))
    assert len(pages) > 0
    assert all(page.statistics is not None for page in pages)
    assert min(page.statistics.min for page in pages) == chunk.statistics.min
    assert max(page.statistics.max for page in pages) == chunk.statistics.max


def test_iter_pages_without_statistics(table):
    pf = _write(table, write_statistics=False)
    pages = list(pf.iter_pages(0, 0))
    assert len(pages) > 1
    assert all(page.statistics is None for page in pages)


def test_iter_pages_multiple_row_groups(table):
    pf = _write(table, row_group_size=3000)
    assert pf.metadata.num_row_groups == 4
    for row_group in range(pf.metadata.num_row_groups):
        for column in range(pf.metadata.num_columns):
            chunk = pf.metadata.row_group(row_group).column(column)
            pages = list(pf.iter_pages(row_group, column))
            assert pages[0].page_type == 'DICTIONARY_PAGE'
            assert sum(page.num_values
                       for page in _data_pages(pages)) == chunk.num_values


def test_iter_pages_nested_column():
    # 'column' is the index of a leaf column, not of a top-level field
    table = pa.table({
        'list': pa.array([None if i % 5 == 0 else [i, None, i + 1]
                          for i in range(NUM_ROWS)], pa.list_(pa.int32())),
        'struct': pa.array([{'a': i, 'b': i % 3 == 0} for i in range(NUM_ROWS)]),
    })
    pf = _write(table, data_page_version='2.0')
    assert len(pf.schema_arrow) == 2
    assert pf.metadata.num_columns == 3
    row_group = pf.metadata.row_group(0)
    assert [row_group.column(i).path_in_schema for i in range(3)] == [
        'list.list.element', 'struct.a', 'struct.b']

    pages = _data_pages(pf.iter_pages(0, 0))
    assert len(pages) > 1
    assert sum(page.num_values for page in pages) == row_group.column(0).num_values
    assert sum(page.num_rows for page in pages) == NUM_ROWS
    # 'list.list.element' is repeated and nullable
    assert all(page.repetition_levels_byte_length > 0 for page in pages)
    assert all(page.definition_levels_byte_length > 0 for page in pages)

    pages = _data_pages(pf.iter_pages(0, 2))
    assert sum(page.num_values for page in pages) == NUM_ROWS
    assert all(page.statistics.physical_type == 'BOOLEAN' for page in pages)

    with pytest.raises(IndexError, match="out of bounds"):
        pf.iter_pages(0, 3)


@pytest.mark.parametrize(('row_group', 'column'),
                         [(-1, 0), (1, 0), (0, -1), (0, 2)])
def test_iter_pages_invalid_index(table, row_group, column):
    pf = _write(table)
    assert pf.metadata.num_row_groups == 1
    assert pf.metadata.num_columns == 2
    with pytest.raises(IndexError, match="out of bounds"):
        pf.iter_pages(row_group, column)


def test_iter_pages_pages_are_self_contained(table):
    pf = _write(table, compression='snappy')
    pages = list(pf.iter_pages(0, 1))
    statistics = [page.statistics for page in _data_pages(pages)]
    # Pages collected earlier are not modified by reading the following
    # pages, nor by closing the file
    pf.close()
    del pf
    expected_pages = list(_write(table, compression='snappy').iter_pages(0, 1))
    assert len(pages) == len(expected_pages)
    for page, expected in zip(pages, expected_pages):
        assert page.buffer.equals(expected.buffer)
    assert len({page.buffer.to_pybytes() for page in pages}) == len(pages)
    # Statistics stay valid without the pages they were read from
    del pages
    gc.collect()
    for stats, expected in zip(statistics, _data_pages(expected_pages)):
        assert stats.min == expected.statistics.min
        assert stats.max == expected.statistics.max
        assert stats.null_count == expected.statistics.null_count


def test_iter_pages_repr(table):
    pf = _write(table, data_page_version='2.0')
    pages = list(pf.iter_pages(0, 1))
    assert "page_type: DICTIONARY_PAGE" in repr(pages[0])
    assert "is_sorted: False" in repr(pages[0])
    assert "page_type: DATA_PAGE_V2" in repr(pages[1])
    assert "num_nulls:" in repr(pages[1])
    assert "has_min_max: True" in repr(pages[1])


@pytest.mark.parquet_encryption
def test_iter_pages_encrypted(tempdir, table):
    footer_key = b"0123456789112345"
    column_key = b"1234567890123450"
    kms_connection_config = pe.KmsConnectionConfig(
        custom_kms_conf={
            "footer_key": footer_key.decode("UTF-8"),
            "column_key": column_key.decode("UTF-8"),
        })
    crypto_factory = pe.CryptoFactory(
        lambda kms_connection_config: InMemoryKmsClient(kms_connection_config))
    # Encrypt the footer and column 'int'; column 'str' stays plaintext
    encryption_config = pe.EncryptionConfiguration(
        footer_key="footer_key", column_keys={"column_key": ["int"]})
    encryption_properties = crypto_factory.file_encryption_properties(
        kms_connection_config, encryption_config)
    path = tempdir / 'encrypted.parquet'
    pq.write_table(table, path, data_page_size=1024,
                   encryption_properties=encryption_properties)

    decryption_properties = crypto_factory.file_decryption_properties(
        kms_connection_config, pe.DecryptionConfiguration())
    pf = pq.ParquetFile(path, decryption_properties=decryption_properties)
    reference = _write(table)
    for column in range(pf.metadata.num_columns):
        pages = list(pf.iter_pages(0, column))
        expected_pages = list(reference.iter_pages(0, column))
        assert len(pages) == len(expected_pages) > 1
        for page, expected in zip(pages, expected_pages):
            assert page.page_type == expected.page_type
            assert page.encoding == expected.encoding
            assert page.num_values == expected.num_values
            assert page.buffer.equals(expected.buffer)
