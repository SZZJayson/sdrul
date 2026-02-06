"""
Tests for data loading and processing modules.

Tests dataset classes, configuration, and compatibility validation.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    # Base classes
    BaseRULDataset,
    RULDatasetWrapper,
    ConcatRULDataset,
    # Config
    DatasetConfig,
    CMAPSS_CONFIG,
    CMAPSS_EXTENDED_CONFIG,
    NCMAPSS_CONFIG,
    get_dataset_config,
    validate_dataset_model_compatibility,
    # C-MAPSS
    CMAPSSDataset,
    ContinualCMAPSS,
    create_synthetic_cmapss,
    # N-CMAPSS
    NCMAPSSDataset,
    ContinualNCMAPSS,
    create_synthetic_ncmapss,
)


class TestDatasetConfig:
    """Test DatasetConfig class and presets."""

    def test_cmapss_config_values(self):
        """Test C-MAPSS config has correct values."""
        assert CMAPSS_CONFIG.name == 'cmapss'
        assert CMAPSS_CONFIG.sensor_dim == 14
        assert CMAPSS_CONFIG.num_conditions == 4
        assert CMAPSS_CONFIG.seq_len == 30
        assert CMAPSS_CONFIG.max_rul == 125

    def test_ncmapss_config_values(self):
        """Test N-CMAPSS config has correct values."""
        assert NCMAPSS_CONFIG.name == 'ncmapss'
        assert NCMAPSS_CONFIG.sensor_dim == 20
        assert NCMAPSS_CONFIG.num_conditions == 4
        assert NCMAPSS_CONFIG.seq_len == 50
        assert NCMAPSS_CONFIG.max_rul == 200

    def test_condition_mapping(self):
        """Test condition ID mapping."""
        assert CMAPSS_CONFIG.get_condition_id('FD001') == 0
        assert CMAPSS_CONFIG.get_condition_id('FD002') == 1
        assert CMAPSS_CONFIG.get_condition_id('FD003') == 2
        assert CMAPSS_CONFIG.get_condition_id('FD004') == 3

    def test_get_sub_datasets(self):
        """Test getting list of sub-datasets."""
        sub_datasets = CMAPSS_CONFIG.get_sub_datasets()
        assert 'FD001' in sub_datasets
        assert 'FD002' in sub_datasets
        assert len(sub_datasets) == 4

    def test_get_dataset_config(self):
        """Test get_dataset_config function."""
        config = get_dataset_config('cmapss')
        assert config is CMAPSS_CONFIG

        config = get_dataset_config('ncmapss')
        assert config is NCMAPSS_CONFIG

        with pytest.raises(ValueError):
            get_dataset_config('unknown')

    def test_validate_compatibility_success(self):
        """Test successful compatibility validation."""
        result = validate_dataset_model_compatibility(
            CMAPSS_CONFIG,
            model_sensor_dim=14,
            model_num_conditions=4,
        )
        assert result is True

    def test_validate_compatibility_sensor_mismatch(self):
        """Test sensor dimension mismatch detection."""
        with pytest.raises(ValueError, match="Sensor dimension mismatch"):
            validate_dataset_model_compatibility(
                CMAPSS_CONFIG,
                model_sensor_dim=20,  # Wrong
                model_num_conditions=4,
            )

    def test_validate_compatibility_condition_mismatch(self):
        """Test condition count mismatch detection."""
        with pytest.raises(ValueError, match="Condition count mismatch"):
            validate_dataset_model_compatibility(
                CMAPSS_CONFIG,
                model_sensor_dim=14,
                model_num_conditions=2,  # Too few
            )


class TestCMAPSSDataset:
    """Test CMAPSSDataset class."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_data.npz')
            create_synthetic_cmapss(
                num_engines=10,
                num_sensors=14,
                max_cycles=100,
                min_cycles=50,
                save_path=path,
                seed=42,
            )
            yield path

    def test_dataset_returns_three_values(self, temp_data_path):
        """Test that __getitem__ returns (sequence, rul, condition_id)."""
        dataset = CMAPSSDataset(temp_data_path, condition_id=2)

        sequence, rul, cond = dataset[0]

        assert isinstance(sequence, torch.Tensor)
        assert isinstance(rul, torch.Tensor)
        assert isinstance(cond, torch.Tensor)

        assert sequence.dim() == 2  # [seq_len, sensor_dim]
        assert rul.dim() == 0  # scalar
        assert cond.dim() == 0  # scalar
        assert cond.dtype == torch.long

    def test_condition_id_assignment(self, temp_data_path):
        """Test condition_id is correctly assigned."""
        dataset = CMAPSSDataset(temp_data_path, condition_id=3)

        _, _, cond = dataset[0]
        assert cond.item() == 3

        # All samples should have same condition_id
        for i in range(min(10, len(dataset))):
            _, _, c = dataset[i]
            assert c.item() == 3

    def test_get_sensor_dim(self, temp_data_path):
        """Test get_sensor_dim method."""
        dataset = CMAPSSDataset(temp_data_path)
        assert dataset.get_sensor_dim() == 14

    def test_get_condition_id(self, temp_data_path):
        """Test get_condition_id method."""
        dataset = CMAPSSDataset(temp_data_path, condition_id=2)
        assert dataset.get_condition_id() == 2

    def test_sequence_shape(self, temp_data_path):
        """Test sequence has correct shape."""
        dataset = CMAPSSDataset(temp_data_path, seq_len=30)

        sequence, _, _ = dataset[0]
        assert sequence.shape == (30, 14)

    def test_rul_clipping(self, temp_data_path):
        """Test RUL values are clipped to max_rul."""
        dataset = CMAPSSDataset(temp_data_path, max_rul=125)

        for i in range(min(10, len(dataset))):
            _, rul, _ = dataset[i]
            assert rul.item() <= 125


class TestContinualCMAPSS:
    """Test ContinualCMAPSS class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with multiple sub-datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for sub_id in ['FD001', 'FD003']:
                path = os.path.join(tmpdir, f'{sub_id}_train.npz')
                create_synthetic_cmapss(
                    num_engines=10,
                    save_path=path,
                    seed=int(sub_id[-1]),
                )
            yield tmpdir

    def test_condition_id_per_subdataset(self, temp_data_dir):
        """Test each sub-dataset has unique condition_id."""
        continual = ContinualCMAPSS(
            temp_data_dir,
            sub_datasets=['FD001', 'FD003'],
        )

        # FD001 should have condition_id 0
        dataset_0 = continual.get_task_dataset(0)
        _, _, cond_0 = dataset_0[0]
        assert cond_0.item() == 0

        # FD003 should have condition_id 2
        dataset_1 = continual.get_task_dataset(1)
        _, _, cond_1 = dataset_1[0]
        assert cond_1.item() == 2

    def test_get_condition_id(self, temp_data_dir):
        """Test get_condition_id method."""
        continual = ContinualCMAPSS(
            temp_data_dir,
            sub_datasets=['FD001', 'FD003'],
        )

        assert continual.get_condition_id(0) == 0  # FD001
        assert continual.get_condition_id(1) == 2  # FD003

    def test_get_num_conditions(self, temp_data_dir):
        """Test get_num_conditions method."""
        continual = ContinualCMAPSS(
            temp_data_dir,
            sub_datasets=['FD001', 'FD003'],
        )

        assert continual.get_num_conditions() == 2

    def test_dataloader_returns_three_values(self, temp_data_dir):
        """Test dataloader batches have three values."""
        continual = ContinualCMAPSS(
            temp_data_dir,
            sub_datasets=['FD001'],
            batch_size=4,
        )

        loader = continual.get_task_dataloader(0)
        batch = next(iter(loader))

        assert len(batch) == 3
        x, y, cond = batch

        assert x.shape[0] == 4  # batch_size
        assert y.shape[0] == 4
        assert cond.shape[0] == 4
        assert cond.dtype == torch.long


class TestNCMAPSSDataset:
    """Test NCMAPSSDataset class."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary N-CMAPSS data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'DS02.npz')
            create_synthetic_ncmapss(
                num_engines=10,
                num_sensors=20,
                max_cycles=200,
                min_cycles=100,
                save_path=path,
                seed=42,
            )
            yield path

    def test_dataset_returns_three_values(self, temp_data_path):
        """Test that __getitem__ returns (sequence, rul, condition_id)."""
        dataset = NCMAPSSDataset(temp_data_path, unit_id='DS02')

        sequence, rul, cond = dataset[0]

        assert isinstance(sequence, torch.Tensor)
        assert isinstance(rul, torch.Tensor)
        assert isinstance(cond, torch.Tensor)
        assert cond.dtype == torch.long

    def test_unit_to_condition_mapping(self, temp_data_path):
        """Test unit ID maps to correct condition_id."""
        dataset = NCMAPSSDataset(temp_data_path, unit_id='DS02')
        assert dataset.get_condition_id() == 0

    def test_custom_condition_id(self, temp_data_path):
        """Test custom condition_id override."""
        dataset = NCMAPSSDataset(temp_data_path, unit_id='DS02', condition_id=5)
        assert dataset.get_condition_id() == 5

    def test_get_sensor_dim(self, temp_data_path):
        """Test get_sensor_dim method."""
        dataset = NCMAPSSDataset(temp_data_path)
        assert dataset.get_sensor_dim() == 20


class TestBaseRULDataset:
    """Test base dataset classes."""

    def test_wrapper_adds_condition_id(self):
        """Test RULDatasetWrapper adds condition_id."""
        # Create simple dataset that returns 2 values
        class SimpleDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return torch.randn(30, 14), torch.tensor(50.0)

        wrapped = RULDatasetWrapper(SimpleDataset(), condition_id=3)

        seq, rul, cond = wrapped[0]
        assert cond.item() == 3
        assert cond.dtype == torch.long

    def test_concat_dataset(self):
        """Test ConcatRULDataset combines datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two datasets
            path1 = os.path.join(tmpdir, 'data1.npz')
            path2 = os.path.join(tmpdir, 'data2.npz')

            create_synthetic_cmapss(num_engines=5, save_path=path1, seed=1)
            create_synthetic_cmapss(num_engines=5, save_path=path2, seed=2)

            ds1 = CMAPSSDataset(path1, condition_id=0)
            ds2 = CMAPSSDataset(path2, condition_id=1)

            concat = ConcatRULDataset([ds1, ds2])

            assert len(concat) == len(ds1) + len(ds2)
            assert concat.get_sensor_dim() == 14
            assert concat.get_condition_id() == -1  # Mixed

    def test_concat_preserves_condition_ids(self):
        """Test ConcatRULDataset preserves individual condition_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, 'data1.npz')
            path2 = os.path.join(tmpdir, 'data2.npz')

            create_synthetic_cmapss(num_engines=5, save_path=path1, seed=1)
            create_synthetic_cmapss(num_engines=5, save_path=path2, seed=2)

            ds1 = CMAPSSDataset(path1, condition_id=0)
            ds2 = CMAPSSDataset(path2, condition_id=1)

            concat = ConcatRULDataset([ds1, ds2])

            # First dataset samples should have condition_id 0
            _, _, cond = concat[0]
            assert cond.item() == 0

            # Second dataset samples should have condition_id 1
            _, _, cond = concat[len(ds1)]
            assert cond.item() == 1


class TestModelConfigFromDataset:
    """Test model config creation from dataset config."""

    def test_rul_model_config_from_dataset(self):
        """Test RULModelConfig.from_dataset_config."""
        from models.rul_model import RULModelConfig

        config = RULModelConfig.from_dataset_config(CMAPSS_CONFIG)

        assert config.sensor_dim == 14
        assert config.num_conditions == 4

    def test_dsa_moe_config_from_dataset(self):
        """Test DSAMoEConfig.from_dataset_config."""
        from models.moe.dsa_moe import DSAMoEConfig

        config = DSAMoEConfig.from_dataset_config(CMAPSS_CONFIG, num_stages=3)

        assert config.sensor_dim == 14
        assert config.num_conditions == 4
        assert config.num_stages == 3

    def test_tcsd_config_from_dataset(self):
        """Test TCSDConfig.from_dataset_config."""
        from models.tcsd.tcsd_module import TCSDConfig

        config = TCSDConfig.from_dataset_config(NCMAPSS_CONFIG)

        assert config.sensor_dim == 20
        assert config.num_conditions == 4


class TestDataLoaderIntegration:
    """Test DataLoader integration with datasets."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_data.npz')
            create_synthetic_cmapss(
                num_engines=20,
                save_path=path,
                seed=42,
            )
            yield path

    def test_dataloader_batch_format(self, temp_data_path):
        """Test DataLoader produces correct batch format."""
        from torch.utils.data import DataLoader

        dataset = CMAPSSDataset(temp_data_path, condition_id=1)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        batch = next(iter(loader))

        assert len(batch) == 3
        x, y, cond = batch

        assert x.shape == (8, 30, 14)
        assert y.shape == (8,)
        assert cond.shape == (8,)
        assert (cond == 1).all()

    def test_dataloader_with_trainer_format(self, temp_data_path):
        """Test DataLoader output matches ContinualTrainer expectations."""
        from torch.utils.data import DataLoader

        dataset = CMAPSSDataset(temp_data_path, condition_id=2)
        loader = DataLoader(dataset, batch_size=4)

        for batch in loader:
            # ContinualTrainer.training_step expects (x, y, condition_id)
            x, y, condition_id = batch

            # Verify types and shapes
            assert x.dtype == torch.float32
            assert y.dtype == torch.float32
            assert condition_id.dtype == torch.long

            # Verify can be moved to device
            device = torch.device('cpu')
            x = x.to(device)
            y = y.to(device)
            condition_id = condition_id.to(device)

            break  # Just test first batch


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
